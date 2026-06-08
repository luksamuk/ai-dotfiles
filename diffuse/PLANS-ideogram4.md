# Plano de Integração: Ideogram 4 no `diffuse`

> Criado em: 2026-06-07
> Status: **PENDENTE** — aguardando limpeza de SSD (56 GB livre, precisa ~14 GB)

## Contexto

O [Ideogram 4](https://huggingface.co/ideogram-ai/ideogram-4-fp8) é o primeiro modelo open-weight de geração de imagem da Ideogram (9.3B params, DiT treinado do zero). GGUFs oficiais disponíveis via `leejet/ideogram-4-GGUF` (leejet é o autor do stable-diffusion.cpp).

**Licença**: `ideogram-4-non-commercial` — uso pessoal OK, comercial proibido.

## Componentes Necessários

| Componente | Origem HF | Arquivo | Tamanho Q4_K |
|---|---|---|---|
| Transformer condicionado | `leejet/ideogram-4-GGUF` | `ideogram4-Q4_0.gguf` | ~5.8 GB |
| Transformer incondicional | `leejet/ideogram-4-GGUF` | `ideogram4_uncond-Q4_0.gguf` | ~5.8 GB |
| Text encoder (Qwen3-4B) | GGUF separado | `Qwen3VL-8B-Instruct-Q4_K_M.gguf`* | ~2.5 GB |
| VAE (Flux2 32ch) | ComfyUI ou HF | `flux2_ae.safetensors` | ~0.17 GB |
| `sd-cli` binary | Compilar de `leejet/stable-diffusion.cpp` | N/A | ~15 MB |
| **Total em disco** | | | **~14.3 GB** |

\* O text encoder exato (Qwen3 vs Qwen3-VL) precisa ser confirmado pelo model card oficial ou pelo `sd-cli --help` do stable-diffusion.cpp com suporte a Ideogram4.

## Alternativas de Quantização

| Quant | Transformer cond. | Transformer uncond. | Total aprox. | Qualidade esperada |
|---|---|---|---|---|
| Q2_K | 3.6 GB | 3.6 GB | ~10 GB | Péssima, "extremely VRAM handicapped" |
| IQ4_NL | 5.8 GB | 5.8 GB | ~14 GB | Boa |
| Q4_K | 5.8 GB | 5.8 GB | ~14.3 GB | Boa (recomendada) |
| Q5_0 | 6.9 GB | 6.9 GB | ~16.5 GB | Muito boa |
| Q6_K | 8.0 GB | 8.0 GB | ~18.6 GB | Excelente |
| Q8_0 | 10.1 GB | 10.1 GB | ~23 GB | Near-lossless |

**Recomendação para 6 GB VRAM**: Q4_K com offload parcial para RAM (mesmo padrão do Bonsai).

## Arquitetura de Integração

### Novo backend_type: `sd_cpp`

O `generate.py` atual suporta `backend_type: "gemlite"`. Adicionar `"sd_cpp"` para chamar o `sd-cli` como subprocesso.

```
Prompt (JSON estruturado)
    │
    ▼
sd-cli (subprocess)
    ├── --diffusion-model ideogram4-Q4_0.gguf
    ├── --uncond-diffusion-model ideogram4_uncond-Q4_0.gguf
    ├── --llm Qwen3-4B-Q4_K_M.gguf
    ├── --vae flux2_ae.safetensors
    ├── -p '{ ... JSON ... }'
    │
    ▼
Image (PNG)
```

### Mudanças no Código

#### 1. `generate.py` — MODELS registry

```python
"ideogram4-q4": {
    "backend_id": "ideogram4-q4-sd-cpp",
    "hf_repo": "leejet/ideogram-4-GGUF",
    "dir": "ideogram-4-Q4_0",
    "backend_type": "sd_cpp",
    "bits": "4-bit",
    "description": "Ideogram 4 Q4_0 — 9.3B DiT, structured JSON prompts, best-in-class text rendering",
},
"ideogram4-q5": {
    "backend_id": "ideogram4-q5-sd-cpp",
    "hf_repo": "leejet/ideogram-4-GGUF",
    "dir": "ideogram-4-Q5_0",
    "backend_type": "sd_cpp",
    "bits": "5-bit",
    "description": "Ideogram 4 Q5_0 — 9.3B DiT, higher quality at 16.5 GB disk",
},
```

#### 2. `generate.py` — Novo loader `load_pipeline_sd_cpp()`

```python
def load_pipeline_sd_cpp(model_name: str) -> tuple:
    """Load a stable-diffusion.cpp backed pipeline (subprocess wrapper)."""
    import shutil

    model_info = MODELS[model_name]
    model_root = require_model_dir(model_name)

    sd_cli = shutil.which("sd-cli") or str(SCRIPT_DIR / "bin" / "sd-cli")
    if not Path(sd_cli).exists():
        raise FileNotFoundError(f"sd-cli not found at {sd_cli}. Run: diffuse build-sd-cpp")

    # Build sd-cli argument list
    sd_cpp_args = [
        sd_cli,
        "--diffusion-model", str(model_root / "ideogram4-Q4_0.gguf"),
        "--uncond-diffusion-model", str(model_root / "ideogram4_uncond-Q4_0.gguf"),
        "--llm", str(model_root / "text_encoder" / "*.gguf"),  # glob or exact
        "--vae", str(model_root / "vae" / "flux2_ae.safetensors"),
    ]

    return SdCppPipeline(sd_cli_path=sd_cli, model_root=model_root), 0.0
```

#### 3. `SdCppPipeline` wrapper class

```python
class SdCppPipeline:
    """Wrapper around sd-cli subprocess for Ideogram 4 generation."""

    def __init__(self, sd_cli_path: str, model_root: Path):
        self.sd_cli = sd_cli_path
        self.model_root = model_root
        self.last_peak_memory_mb = 0.0

    def generate_png(self, prompt, seed, steps, height, width):
        """Call sd-cli and return PNG bytes."""
        # Build command, run subprocess, capture output PNG
        # Parse sd-cli stdout for timing/memory stats
        ...
```

#### 4. `run.sh` — Novo comando `build-sd-cpp`

```bash
cmd_build-sd-cpp() {
    local sd_cpp_dir="${SCRIPT_DIR}/vendor/stable-diffusion.cpp"
    local bin_dir="${SCRIPT_DIR}/bin"

    if [[ ! -d "$sd_cpp_dir" ]]; then
        log_info "Cloning stable-diffusion.cpp..."
        git clone https://github.com/leejet/stable-diffusion.cpp.git "$sd_cpp_dir"
    fi

    cd "$sd_cpp_dir"
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

    mkdir -p "$bin_dir"
    cp sd-cli "$bin_dir/sd-cli"
    log_info "sd-cli built and installed to $bin_dir/sd-cli"
}
```

#### 5. `download-model.sh` — Novo comando `download ideogram4`

Baixar os GGUFs do `leejet/ideogram-4-GGUF`, o text encoder Qwen3, e a VAE Flux2 usando `huggingface-cli`.

#### 6. `run.sh` — Atualizar `cmd_status`

Adicionar verificação dos componentes Ideogram 4: sd-cli binary, GGUFs, text encoder, VAE.

### Prompting do Ideogram 4

O Ideogram 4 usa prompts **JSON estruturado**, diferente do Bonsai (texto livre):

```json
{
  "high_level_description": "A sunset over mountains with a lake reflection",
  "style_description": {
    "aesthetics": "photorealistic landscape photography",
    "lighting": "golden hour, warm tones, soft shadows",
    "color_palette": ["#F4A460", "#87CEEB", "#2E4057"]
  }
}
```

O `diffuse` CLI precisa aceitar tanto prompt texto simples (converte pra JSON wrapper) quanto JSON direto.

### Fluxo de Geração

1. `--evict-llm` → evict modelos LLM do llama-swap (libera VRAM)
2. Carrega sd-cli com os pesos GGUF em offload parcial (GPU+RAM)
3. Roda inferência (cond + uncond classifier-free guidance)
4. Salva PNG no CWD do caller
5. Descarrega modelo da VRAM

### Previsão de Performance (RTX 3050 6GB + RAM offload)

- Cold start: ~60-120s (carregamento dos GGUFs com offload)
- Geração 512×512: estimativa ~2-5 min (Q4_K com offload)
- Geração 1024×1024: estimativa ~5-15 min (Q4_K com offload)
- Essas são estimativas conservadoras — precisa medir na prática

## ⚠️ Cautelas e Riscos

1. **Licença não-comercial** — OK para uso pessoal, MAS não pode ser usado para gerar imagens comerciais
2. **6GB VRAM limit** — Q4_K não cabe inteiro em VRAM, precisa de offload pra RAM
3. **sd.cpp precisa compilar com suporte Ideogram4** — verificar se o branch mais recente suporta a arquitetura
4. **Text encoder** — precisa confirmar qual exato (Qwen3-4B vs Qwen3-VL-4B) e o formato GGUF compatível
5. **ComfyUI** — o uploader (`stduhpf`) disse explicitamente que não há garantia de funcionar no ComfyUI ainda
6. **Os GGUFs do `leejet` são canônicos** — preferir os dele em vez dos do `stduhpf`
7. **sd-cli argumentos** — verificar exatamente os flags necessários (podem ter mudado desde o exemplo do model card)

## Ordem de Implementação

1. ✅ Pesquisa completa do modelo e infraestrutura
2. ⬜ Compilar `sd-cli` do `stable-diffusion.cpp` e testar gerando uma imagem manualmente
3. ⬜ Baixar GGUFs Q4_K (cond + uncond) + text encoder + VAE
4. ⬜ Adicionar `backend_type: "sd_cpp"` no `generate.py`
5. ⬜ Implementar `SdCppPipeline` wrapper
6. ⬜ Adicionar `download ideogram4` no `download-model.sh`
7. ⬜ Adicionar `build-sd-cpp` no `run.sh`
8. ⬜ Atualizar `cmd_status`
9. ⬜ Testar geração end-to-end com `--evict-llm`
10. ⬜ Benchmark: tempo de geração, VRAM usage, qualidade vs Bonsai
11. ⬜ Atualizar SKILL.md do diffuse com documentação completa

## Referências

- Modelo oficial: https://huggingface.co/ideogram-ai/ideogram-4-fp8
- GGUFs canônicos: https://huggingface.co/leejet/ideogram-4-GGUF
- GGUFs alternativos: https://huggingface.co/stduhpf/ideogram-4-gguf
- ComfyUI node: https://huggingface.co/Comfy-Org/Ideogram-4
- Repositório: https://github.com/ideogram-oss/ideogram4
- Blog post: https://ideogram.ai/blog/ideogram-4.0/
- stable-diffusion.cpp: https://github.com/leejet/stable-diffusion.cpp