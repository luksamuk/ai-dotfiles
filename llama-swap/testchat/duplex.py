"""Modo duplex voz->voz (transporte chat+TTS): sliding window + mock tools.

Implementacao separada do main.py para o loop de dois passes com:
  - historico persistente no cliente (modelo e' stateless)
  - sliding window por orcamento de tokens (audio ~150 tok/s, LiquidCodec)
  - round-trip das ferramentas de teste (MOCK_TOOLS), confiando no parse do
    tool call pelo llama-server (chat template do GGUF LFM3 renderiza as
    tools no system prompt e o server devolve message.tool_calls pronto)

Dependencias (console, BASE_URL, tools, mock_response) sao injetadas via
configure() para evitar import circular com main.py.
"""
import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import wave

DUPLEX_WINDOW_TOKENS = 24000     # orcamento do historico (32K de contexto - folga p/ tools/saida)
DUPLEX_AUDIO_TOK_PER_SEC = 150   # LiquidCodec: 12 tokens / 80 ms (conservador p/ entrada)
DUPLEX_MAX_TOOL_ROUNDS = 4

# Tag de tool call do LFM3 - montada por partes (nao emitir o literal
# completo em texto de agente: quebra o stream). Na rota chat/completions o
# server parseia via template do GGUF (so avisamos se vier cru). Na rota
# realtime WS o server NAO parseia tool calls - o cliente parseia
# (parse_tool_calls abaixo).
_LFM3_TC_OPEN = "<" + "tool_" + "call" + ">"
_LFM3_TC_CLOSE = "</" + "tool_" + "call" + ">"
_TOOL_BLOCK_RE = re.compile(re.escape(_LFM3_TC_OPEN) + r"\s*(.*?)\s*"
                            + re.escape(_LFM3_TC_CLOSE), re.S)

# Cap do resultado de tool que volta como user item (duplex): fala falada.
_TOOL_RESULT_MAX_CHARS = 2500

_cfg = {}


def configure(console, base_url, tools, mock_response):
    """Injecao de dependencias (main.py chama no boot)."""
    _cfg.update(console=console, base_url=base_url,
                tools=tools, mock_response=mock_response)


def print_tools_summary(active):
    """Resumo (em ingles, idioma do system prompt do modelo de voz) das
    ferramentas de teste na entrada do modo duplex."""
    c = _cfg["console"]
    c.print("[cyan]Test tools available:[/]")
    c.print("[dim]   get_weather  - current weather by city[/]")
    c.print("[dim]   calculator   - math expressions[/]")
    c.print("[dim]   get_time     - date/time by timezone[/]")
    if active:
        c.print("[dim]   Responses are mocked - just ask by voice (PT or English).[/]")
    else:
        c.print("[dim]   (Not executed on this transport - use --duplex-tts for the chat+TTS lane.)[/]")


def _msg_tokens(msg: dict) -> int:
    """Estimativa grosseira de tokens de uma mensagem do historico."""
    content = msg.get("content")
    if isinstance(content, str):
        return max(1, len(content) // 3)
    total = 0
    for part in content or []:
        if isinstance(part, dict):
            ptype = part.get("type")
            if ptype == "input_audio":
                total += int(part.get("_tok") or 0)
            elif ptype == "text":
                total += max(1, len(part.get("text") or "") // 3)
    return total


def _trim_window(messages: list, budget: int = DUPLEX_WINDOW_TOKENS):
    """Sliding window: solta TURNOS completos mais antigos (user + tudo que
    veio depois dele) ate a estimativa caber no orcamento. Turno nunca sai
    pela metade - tool round-trips (assistant tool_calls + results) ficam
    colados ao user que os originou. O ultimo turno fica sempre."""
    dropped = 0

    def total() -> int:
        return sum(_msg_tokens(m) for m in messages)

    while True:
        starts = [i for i, m in enumerate(messages)
                  if isinstance(m, dict) and m.get("role") == "user"]
        if len(starts) < 2 or total() <= budget:
            break
        del messages[starts[0]:starts[1]]
        dropped += starts[1] - starts[0]
    return dropped, total()


def _payload_messages(messages: list) -> list:
    """Copia o historico para o payload, tirando chaves privadas (_tok)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            clean = []
            for part in content:
                if isinstance(part, dict):
                    clean.append({k: v for k, v in part.items()
                                  if not k.startswith("_")})
            out.append({**m, "content": clean})
        else:
            out.append(m)
    return out


def _warn_unparsed_tool_xml(console, content: str):
    """XML cru na resposta = server nao parseiou o tool call (template do
    GGUF desatualizado?). Nao tentamos parsear: so avisamos."""
    if content and _LFM3_TC_OPEN in content:
        console.print("[yellow]Resposta com bloco de tool call nao parseado pelo server "
                      "- confira o chat template do GGUF.[/]")


def tools_instructions() -> str:
    """System prompt da receita oficial (doc Tool Use / Audio Capabilities)
    para spoken tool calling no LFM3: modo interleaved + lista de tools."""
    lines = "\n".join(json.dumps(t["function"], ensure_ascii=False)
                      for t in _cfg["tools"])
    return ("Respond with function calls, and interleaved text and audio.\n\n"
            "Return tool calls for the requested actions, in the order "
            "requested. Do not claim that an action has completed without "
            "a tool result.\n"
            "List of tools:\n<tools>\n" + lines + "\n</tools>")


def parse_tool_calls(text: str) -> list:
    """Parseia blocos de tool call do LFM3 no texto acumulado da resposta.
    Necessario apenas na rota realtime WS: o server nao emite eventos de
    tool call (so deltas de texto/audio), o parse e' do lado do cliente."""
    calls = []
    for block in _TOOL_BLOCK_RE.finditer(text or ""):
        body = block.group(1)
        name_m = re.search(r"<function=([\w.\-]+)>", body)
        if not name_m:
            continue
        args = {}
        for pm in re.finditer(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>",
                              body, re.S):
            args[pm.group(1)] = pm.group(2)
        calls.append({"name": name_m.group(1), "arguments": args})
    return calls


def strip_tool_blocks(text: str) -> str:
    """Remove os blocos de tool call do texto EXIBIDO (e, portanto, da fala
    transcrita): o round-trip ja mostra 'Tool call: name(args)' separado."""
    return _TOOL_BLOCK_RE.sub("", text or "")


def tool_result_user_text(calls: list) -> str:
    """Executa as tools e empacota os resultados como texto de user item --
    o protocolo realtime do llama.cpp (PR #31269) so aceita
    conversation.item.create de role user, entao o resultado da tool
    viaja como texto do 'usuario' (harness de teste: aceitavel).

    Cabecalho explicito + delimitador entre chamadas: sem isso o 2B
    ecoa o resultado verbatim no fim da resposta, colado na ultima palavra.
    Cap por resultado: fala falada nao consome 4KB de texto. """
    mock_response = _cfg["mock_response"]
    parts = []
    for c in calls:
        result = mock_response(c["name"], c["arguments"])
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        result = result.strip()
        if len(result) > _TOOL_RESULT_MAX_CHARS:
            result = result[:_TOOL_RESULT_MAX_CHARS] + "\n[... truncated]"
        parts.append(f'>>> tool: {c["name"]}\narguments: '
                     f'{json.dumps(c["arguments"], ensure_ascii=False)}\n'
                     f'result:\n{result}\n<<<')
    header = ("[Tool results below, delimited by >>> tool: / <<<. Use them as "
              "context to answer. Do NOT repeat them verbatim in your speech; "
              "answer naturally, briefly.]\n\n")
    return header + "\n\n".join(parts)


def run_chat_tts_loop(lane_id):
    """Transporte llama.cpp (dois passes por turno):
    1. chat completions (audio-in, tools de teste, historico + sliding window)
    2. /tts (texto) -> fala 24 kHz -> pw-play

    O historico vive aqui no cliente (o modelo e' stateless): a cada turno a
    conversa inteira e' re-enviada; passou do orcamento de tokens, os turnos
    mais antigos saem inteiros (sliding window, ver _trim_window).
    """
    console = _cfg["console"]
    base_url = _cfg["base_url"]
    tools = _cfg["tools"]
    mock_response = _cfg["mock_response"]
    console.print("[dim]Transporte llama.cpp (chat + /tts, dois passes).[/]")
    console.print("[cyan]Tools active on this transport - just ask (PT or English).[/]")

    # Sobe a lane pela swap (spawn) e acha a porta direta do llama-server
    urllib.request.urlopen(urllib.request.Request(
        f"{base_url}/chat/completions", data=json.dumps(
        {"model": lane_id, "messages": [{"role": "system", "content": "ping"}],
         "max_tokens": 1}).encode(),
        headers={"Content-Type": "application/json"}), timeout=300)
    running = json.loads(urllib.request.urlopen(
        f"{base_url[:-len('/v1')]}/running", timeout=10).read().decode())["running"]
    entry = next((m for m in running if m.get("model") == lane_id), None)
    if entry is None:
        console.print("[red]lane nao esta em /running.[/]")
        return
    from urllib.parse import urlparse
    port = urlparse(entry.get("proxy") or "").port
    http_base = f"http://127.0.0.1:{port}/v1"

    system_prompt = ("You are a helpful voice assistant. You hear the user's "
                     "spoken request. Respond with text only. "
                     "Always respond in Brazilian Portuguese.")
    messages: list = [{"role": "system", "content": system_prompt}]
    wav_in = "/tmp/testchat_duplex_in.wav"
    while True:
        console.print()
        console.print("[green]Gravando... (Enter para enviar)[/]")
        rec = subprocess.Popen(["pw-record", wav_in])
        try:
            input()
        except EOFError:
            rec.terminate()
            console.print("[dim]stdin fechado - encerrando duplex.[/]")
            return
        except KeyboardInterrupt:
            rec.terminate()
            console.print("[dim]Modo duplex encerrado (Ctrl+C).[/]")
            return
        rec.terminate()
        try:
            rec.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rec.kill()
        if not os.path.exists(wav_in) or os.path.getsize(wav_in) < 1024:
            console.print("[yellow]Gravacao vazia.[/]")
            continue
        # duracao do audio -> custo em tokens (sliding window)
        try:
            with wave.open(wav_in, "rb") as f:
                audio_secs = f.getnframes() / max(1, f.getframerate())
        except Exception:
            audio_secs = os.path.getsize(wav_in) / 96000  # ~48 kHz s16 mono
        b64 = base64.b64encode(open(wav_in, "rb").read()).decode()
        messages.append({"role": "user", "content": [
            {"type": "input_audio",
             "input_audio": {"data": b64, "format": "wav"},
             "_tok": int(audio_secs * DUPLEX_AUDIO_TOK_PER_SEC)},
        ]})

        dropped, est = _trim_window(messages)
        if dropped:
            console.print(f"[dim]Sliding window: {dropped // 2} turno(s) antigo(s) "
                          f"saiu/sairam do contexto (estimado ~{est} tokens).[/]")

        chat_text = ""
        try:
            tool_rounds = 0
            while True:  # tool round-trip (ReAct simplificado)
                console.print("[dim]Passe 1: transcricao/resposta...[/]")
                req = urllib.request.Request(
                    f"{http_base}/chat/completions",
                    data=json.dumps({"model": "default",
                                     "messages": _payload_messages(messages),
                                     "max_tokens": 600, "tools": tools,
                                     "tool_choice": "auto"}).encode(),
                    headers={"Content-Type": "application/json"})
                r = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
                msg = r["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []
                if tool_calls and tool_rounds < DUPLEX_MAX_TOOL_ROUNDS:
                    tool_rounds += 1
                    messages.append({"role": "assistant",
                                     "content": msg.get("content") or "",
                                     "tool_calls": tool_calls})
                    for tc in tool_calls:
                        fn = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        console.print(f"[magenta]Tool call: {fn}"
                                      f"({json.dumps(args, ensure_ascii=False)})[/]")
                        result = mock_response(fn, args)
                        messages.append({"role": "tool",
                                         "tool_call_id": tc.get("id") or "call_0",
                                         "content": result})
                    continue  # devolve os resultados ao modelo
                chat_text = strip_tool_blocks((msg.get("content") or "")).strip()
                if not tool_calls:
                    _warn_unparsed_tool_xml(console, chat_text)
                break
            if not chat_text and tool_rounds >= DUPLEX_MAX_TOOL_ROUNDS:
                chat_text = "(esgotei as rodadas de tool call sem resposta final)"
            messages.append({"role": "assistant", "content": chat_text})
            console.print(f"[bold]Resposta: {chat_text}[/]")
            console.print("[dim]Passe 2: voz...[/]")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/tts",
                data=json.dumps({"input": chat_text, "voice": "female"}).encode(),
                headers={"Content-Type": "application/json"})
            r2 = urllib.request.urlopen(req, timeout=600)
            wav_out = r2.read()
            out = "/tmp/testchat_duplex_reply.wav"
            open(out, "wb").write(wav_out)
            subprocess.run(["pw-play", out])
            console.print(f"[dim]Audio: {len(wav_out)/48000:.1f}s tocados.[/]")
        except urllib.error.HTTPError as e:
            console.print(f"[red]HTTP {e.code}: {e.read()[:200]}[/]")
        except KeyboardInterrupt:
            console.print("[dim]Modo duplex encerrado.[/]")
            return
        except Exception as e:
            console.print(f"[red]Erro no turno duplex: {str(e)[:200]}[/]")