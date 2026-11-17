---
name: window-driving
description: "Guia prático para manipular janelas de apps com o Cua Driver MCP em desktops Linux multi-monitor (Hyprland/Wayland e X11): descobrir janelas, ler árvores AT-SPI, converter coordenadas entre captura/janela/desktop, escolher a rota certa de clique (element token vs pixel vs desktop real) e verificar resultados. Use quando precisar clicar, navegar ou ler UI de aplicativos reais, especialmente quando o clique em background falha, quando há dois monitores, ou quando o tamanho da captura não bate com o tamanho da janela."
---

# Manipulação de janelas no desktop (aprendizado em produção)

> Status 2026-11-05: validada em exploração real do VFB no Zen Browser/Hyprland.

Consolidação do que funcionou de verdade ao explorar o Virtual Fly Brain no
Zen Browser em um Hyprland 0.56 com dois monitores. Vale para qualquer app:
navegadores Firefox/Chromium, nativos GTK/Qt, overlays e popups.

## 1. Descoberta: nunca presuma onde a janela está

1. `list_windows()` — nomes, pids, `window_id`, bounds (`LARGURAxALTURA+X+Y`).
   Filtre por `app` ou `pid` para não afogar no ruído.
2. `get_desktop_state(screenshot_out_file=...)` — PNG do layout composto de
   TODOS os monitores + lista das janelas visíveis em coordenadas globais.
3. Em Wayland, `hyprctl clients`/`monitors` (via bash) revela qual monitor e
   workspace cada janela habita e qual monitor está focado — útil quando a
   janela existe mas está "longe" (outro workspace/monitor) e os cliques de
   foco falham por isso.
4. Janela em outro monitor/workspace não é erro: os cliques de escopo desktop
   continuam funcionando, porque a coordenada é global do layout.

## 2. Ler a janela antes de agir (invariante)

- `get_window_state(pid, window_id)` é OBRIGATÓRIO antes de qualquer clique
  (invariante snapshot→ação). Devolve árvore AT-SPI + captura.
- **Cada nova leitura invalida os tokens anteriores** (`s0000002a:11` estilo).
  Releia imediatamente antes do ato; nunca clique token antigo.
- Árvores grandes (apps web) truncam. Use:
  - `query:"texto"` para projetar só o ramo alvo (+ancestrais);
  - `timeout_ms:5000`, `max_elements` maior;
  - `tree_format:"elements"` quando precisar de frames exatos.
- Confirme SEMPRE cruzando árvore e imagem: a árvore mente em superfícies
  exóticas; a imagem dá o chão real.

## 3. Conversão de coordenadas — o segredo de multi-monitores/resize

A captura da janela vem **redimensionada** (ex.: janela 1906×1042 → imagem
1450×793). Os cliques por janela usam os pixels DA CAPTURA ENTREGUE; os
cliques por desktop usam pixels globais do layout. Dois frames diferentes:

```
fator        = largura_real_da_janela / largura_da_captura   (≈1.31 no exemplo)
pixel_janela = pixel_da_captura × fator
pixel_desktop= origem_da_janela (bounds de list_windows) + pixel_janela
```

Regras:
- **Nunca recalcule a origem de cor** quando a janela pode ter se movido ou
  redimensionado: releia `list_windows` a cada rodada de navegação.
- O `zoom(pid, window_id, x1..y2)` usa o MESMO frame da captura da janela —
  é a verificação barata de que a conta está certa, e um clique depois do
  zoom pode usar `from_zoom:true` com coordenadas lidas no recorte.
- `capture_id` também amarra um PNG específico como fonte de coordenadas.
- Monitores separados NÃO mudam a receita: bounds de `list_windows` e o frame
  de `get_desktop_state` já são no espaço composto; some a origem da janela e
  pronto (no exemplo, origem (7,31) → campo em captura (718,319) → desktop
  (951,450) em um layout de 4480 px de largura).

## 4. Escolha da rota de clique (árvore de decisão testada)

1. **Elemento existe na árvore e tem ação** → `click(element_token=...)`,
   `delivery_mode:"background"`. Clica sem roubar foco e funcionou até em
   menus/itens de app web. Se o elemento não tem ação, o driver sobe pelo
   ancestral (`clickAncestor`) — confie nisso.
2. **Superfície fora da árvore (canvas/grafos/custom)** → clique real por
   desktop: `click(x, y, target={"kind":"desktop","display_id":"primary"})`
   apenas com esse target (não misture `scope`/`pid`/`window_id` — rejeita).
   Efeito colateral ÚTIL: o clique real **concede foco** à janela.
   Isto funcionou no Hyprland via plugin cua; em X11, `pid+x/y` background
   também é opção.
3. **Cliques por janela em background com pixels** → normalmente recusados em
   Wayland sem backend focus-free ("the requested target has no focus-free
   input backend"). Não insista: migre para a rota desktop.
4. **`delivery_mode:"foreground"`** → só se recusado, e em Hyprland pode falhar
   por si (`production Hyprland input plugin is unavailable`). A rota desktop
   resolve o mesmo problema inclusive dando foco.
5. **Arraste** (`drag` com `target` desktop + `coordinate_frame:"desktop"`)
   funciona para mover caixas flutuantes dentro de apps web; se o app não
   mover, re-observe e tente outro controle.

## 5. Sessões: o label mata o fluxo

- Sessões nomeadas (`start_session("x")`) são úteis para marcar um trabalho,
  mas **encerram por idle** e a próxima chamada com aquele label falha com
  `session 'x' has ended`. Isso derrubou o fluxo várias vezes em produção.
- Recomendação forte: **trabalhe sem `session`** (sessão implícita é estável)
  e só crie labels para trabalhos curtos e verificados.
- Se ver "session has ended", simplesmente repita a chamada sem o label.

## 6. Verifique o efeito, sempre

- Clique não é sucesso. Padrão mínimo: screenshot/zoom antes e depois
  (`screenshot_out_file` para não truncar em base64) + re-ler árvore com
  `query` pelo texto que deveria aparecer.
- `verify_state(pid, window_id, expect=[...])` para asserções determinísticas.
- `unknown`/silêncio nunca é sucesso; imagem e árvore provam o estado.

## 7. Tabela de falhas → ação (memória de campo)

| Erro observado | Causa real | Ação |
|---|---|---|
| `session 'x' has ended` | sessão nomeada expirou por idle | repetir sem label / `start_session` |
| `Background delivery is not available ... focus-free input backend` | pixel de janela sem foco, sem backend de fundo | clique real por desktop (dá foco) |
| `foreground_unavailable: Hyprland input plugin is unavailable` | plugin de entrada do compositor ausente | clique real por desktop |
| `target cannot be combined with legacy scope/pid/window_id` | abuso de parâmetros com `target` | usar só `target` + x/y |
| Tokens `stale` / `Invalidated snapshots` | nova leitura da mesma janela | reler e usar token fresco |
| Árvore `truncated` / parcial | orçamento de nós/tempo | `query` + `timeout_ms:5000` |
| `surface_identity_unproven` | captura sem atestação de identidade | não usar esses pixels; seguir pela árvore/reler |
| Texto `set_value`/`type_text` não comete | widgets web (react-select etc.) resistem a escrita AT-SPI | (digitação: ver nota abaixo) rotas alternativas do app (links que preenchem formulários, CLI do app, seleção primária) |

## 8. Digitação (pendente — consolidar depois)

Nesta máquina a injeção de teclado falhou por todos os caminhos testados
(libei/portal sem RemoteDesktop no Hyprland, `send_key_state` no-op, sem
/dev/uinput). A lição que já vale: **prefira fluxos do app que não exijam
digitação** — links tipo "Add [term] to ... Query" que preenchem campos,
abrir URLs por CLI do próprio navegador (nova aba na instância em execução),
listas clicáveis em vez de autocomplete. Detalhes da digitação serão
consolidados em outra rodada.

## 9. Receita completa verificada (Virtual Fly Brain como caso real)

1. `list_windows` → achar janela do VFB (`app=zen`) e o `pid/window_id`.
2. `get_window_state` com `query:"Circuit Browser"` → token do item de menu.
3. Clice por token (abre menu Tools → Circuit Browser).
4. Abrir página de um neurônio com conectividade em nova aba via CLI do
   navegador (`zen <url do geppetto com ?id=...&i=...>`).
5. Na Term Info, clicar "Add [neurônio] to Circuit Browser Query" (link que
   preenche o **primeiro campo vazio** da Connectivity query; clicar de novo
   com outro neurônio em foco preenche o segundo).
6. **RUN QUERY** só habilita com os dois neurônios preenchidos.
7. Ícone ❐ do cabeçalho do painel maximiza a visualização do grafo.
8. Cada passo: screenshot + verificação visual imediata (padrão da seção 6).
