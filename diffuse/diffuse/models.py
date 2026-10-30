"""Model registry — all supported models and their configurations."""
from __future__ import annotations

# ── Model registry ─────────────────────────────────────────────────────────
# Each entry defines:
#   backend_type: which backend module handles generation
#   bits: quantization description
#   dir: model directory under MODELS_DIR
#   description: human-readable description
#   enhance_model: default LLM for prompt enhancement
#   enhance_type: "qwen21" | "vision" | "agate" | "moss" (natural language;
#                 "ideogram" é legado JSON, sem entry viva hoje)
#   default_size: (width, height) tuple, optional
#   hf_files: list of {repo, files, subdir?, rename?, hf_path?} for auto-download
#   components: list of {name, path, size_gb} for diffuse list display
#   category: "image" or "video"

MODELS: dict[str, dict] = {
    # ── Image generation ────────────────────────────────────────────────────
    # Bonsai Image 4B Ternary (gemlite + HQQ kernels, subprocess venv)
    "ternary-gemlite": {
        "backend_id": "bonsai-ternary-gemlite",
        "dir": "bonsai-image-4B-ternary-gemlite",
        "backend_type": "bonsai",
        "category": "image",
        "bits": "1.58-bit",
        "description": "Bonsai Image 4B Ternary — 1.58-bit gemlite, 95% of FP16 quality, fast T2I",
        "default_size": (512, 512),
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "vision",
        "components": [
            {"name": "Bonsai 4B Ternary transformer (gemlite int2)", "path": "bonsai-image-4B-ternary-gemlite/transformer-gemlite-int2", "size_gb": 1.5},
            {"name": "Qwen3 text encoder (HQQ 4-bit)", "path": "bonsai-image-4B-ternary-gemlite/text_encoder-hqq-4bit", "size_gb": 2.8},
            {"name": "FLUX.2 VAE (AutoencoderKLFlux2)", "path": "bonsai-image-4B-ternary-gemlite/vae", "size_gb": 0.17},
        ],
    },
    "qwen-image-2.1": {
        "backend_id": "qwen-image-2.1-wan2gp",
        "dir": "qwen-image-2.1-UC-wan2gp",
        "backend_type": "qwen21_wan2gp",
        "lora_root": "qwen21",
        "category": "image",
        "bits": "INT8 ConvRot (abenzerps UC repack)",
        "description": "Qwen-Image 2.1 7B UNCENSORED — edit/outpaint/RGBA/4K nativos do Wan2GP, TE Qwen3-VL-8B abliterated convrot. --turbo = LoRA distill sobre o base (passada 2)",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "qwen21",
        "default_size": (1024, 1024),
        "components": [
            {"name": "qwen-image-2.1-UC-int8_convrot.safetensors (DiT)", "path": "Wan2GP/ckpts/qwen-image-2.1-UC-int8_convrot.safetensors", "size_gb": 7.26},
            {"name": "qwen3vl_8b_int8_convrot.safetensors (TE, abliterated)", "path": "Wan2GP/ckpts/qwen3vl_8b_int8_convrot.safetensors", "size_gb": 9.35},
            {"name": "qwen_image_2.1_vae_bf16.safetensors (VAE)", "path": "Wan2GP/ckpts/qwen_image_21/qwen_image_2.1_vae_bf16.safetensors", "size_gb": 0.68},
        ],
    },
    # Qwen-Image 2.1 Viggle Turbo REMOVIDO 15/out (user: "nós removemos ele")
    # — entry do diffuse registry fora; fragment do swap em _disabled/.
    # Qwen-Image 2.1 TURBO OFICIAL (Alibaba distill, 09/10) — 8 steps, cfg=1.0.
    # MIGRADO 16/out ao Wan2GP (backend wan2gp_backend): o path sd.cpp (Turbo
    # Q4_K_M da abenzerps + custom sigmas sd-cli) foi aposentado junto — o
    # WanGP carrega o turbo int8_convrot com denoise em GPU (profile 5) e TE
    # Sakura GGUF local (remap blk.* nativo do handler). Réguas no
    # references/qwen-image-2.1.md.
    "qwen-image-2.1-turbo": {
        "backend_id": "qwen-image-2.1-turbo-wan2gp",
        "dir": "qwen21-turbo-wan2gp",  # weights live in WanGP ckpts/, not here
        "backend_type": "qwen21_turbo_wan2gp",
        "lora_root": "qwen21",
        "category": "image",
        "bits": "INT8 ConvRot (DDM repack) — official Alibaba turbo checkpoint",
        "description": "Qwen-Image 2.1 Turbo OFICIAL — 8 steps, cfg=1.0, TE Sakura uncensored, T2I + editing via Wan2GP",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "qwen21",
        "default_size": (1024, 1024),
        "components": [
            {"name": "qwen_image_21_turbo_7B_int8_convrot.safetensors (DiT, official turbo)", "path": "Wan2GP/ckpts/qwen_image_21_turbo_7B_int8_convrot.safetensors", "size_gb": 6.76},
            {"name": "Sakura-TextEncoder-Qwen3VL-8B-Uncensored-8bit GGUF (TE, abliterated)", "path": "Wan2GP/ckpts/Qwen3-VL-8B-Instruct/Sakura-TextEncoder-Qwen3VL-8B-Uncensored-8bit-8.11GiB.gguf", "size_gb": 8.11},
            {"name": "qwen_image_21_vae.safetensors (VAE, F32 DDM repack)", "path": "Wan2GP/ckpts/qwen_image_21/qwen_image_21_vae.safetensors", "size_gb": 1.26},
        ],
    },
    # Agate preview 001 (Logolabs) — 260M thinker-steered FCDM, pure PyTorch backend.
    # Fixed 256×256 output (architecture limit), 50 steps cfg 3.0. Fast: ~8s first run
    # (CUDA graph warmup), ~2-4s after. Niche: flat logos/icons, spatial composition drafts.
    # Pipeline code ships in the repo (agate/*.py, imported via sys.path from the model dir).
    "agate-preview-001": {
        "backend_id": "agate-preview-001",
        "dir": "agate-preview-001",
        "backend_type": "agate",
        "category": "image",
        "bits": "BF16 (native, no quant)",
        "description": "Agate 001 — 260M FCDM, fixed 256×256, GenEval 0.550, flat logos/icons + spatial drafts in seconds",
        "default_size": (256, 256),
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "agate",
        "hf_files": [
            {"repo": "Logolabs/agate-preview-001",
             "files": ["config.json", "generator.safetensors",
                        "text_encoder/model.safetensors", "text_encoder/tokenizer.json",
                        "text_encoder/tokenizer_config.json", "text_encoder/config.json",
                        "text_encoder/special_tokens_map.json",
                        "agate/__init__.py", "agate/pipeline.py", "agate/text_encoder.py",
                        "agate/fcdm_t2i.py", "agate/fcdm_thinker2.py", "agate/fcdm_planner.py"]},
            {"repo": "stabilityai/sd-vae-ft-mse",
             "files": ["config.json", "diffusion_pytorch_model.safetensors"],
             "subdir": "vae"},
        ],
        "components": [
            {"name": "generator.safetensors (FCDM-T2 thinker+UNet)", "path": "agate-preview-001/generator.safetensors", "size_gb": 0.76},
            {"name": "Ettin-68M text encoder (ModernBERT)", "path": "agate-preview-001/text_encoder/model.safetensors", "size_gb": 0.27},
            {"name": "SD-VAE-ft-MSE decoder", "path": "agate-preview-001/vae/diffusion_pytorch_model.safetensors", "size_gb": 0.33},
            {"name": "agate/ pipeline code (imported from model dir)", "path": "agate-preview-001/agate", "size_gb": 0.0},
        ],
    },
    # Agate preview 003 (Logolabs) — 261M thinker-steered FCDM multi-res (512 native).
    # QIB 32.6 (vs 001 28.2, SD1.5 29.1), GenEval 0.554. Prompt pipeline embutido
    # (normalize/spell). API superset do 001: pipe(prompt, seed, steps, cfg, resolution=...).
    "agate-preview-003": {
        "backend_id": "agate-preview-003",
        "dir": "agate-preview-003",
        "backend_type": "agate",
        "category": "image",
        "bits": "BF16 (native, no quant)",
        "description": "Agate 003 — 261M FCDM multi-res 512/256, QIB 32.6, flat logos/icons + spatial drafts",
        "default_size": (512, 512),
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "agate",
        "hf_files": [
            {"repo": "Logolabs/agate-preview-003",
             "files": ["config.json", "generator.safetensors",
                        "text_encoder/model.safetensors", "text_encoder/tokenizer.json",
                        "text_encoder/tokenizer_config.json", "text_encoder/config.json",
                        "text_encoder/special_tokens_map.json",
                        "agate/__init__.py", "agate/pipeline.py", "agate/text_encoder.py",
                        "agate/fcdm_t2i.py", "agate/fcdm_thinker2_mr.py", "agate/fcdm_planner.py",
                        "agate/marking.py", "agate/prompt_norm.py"]},
            {"repo": "stabilityai/sd-vae-ft-mse",
             "files": ["config.json", "diffusion_pytorch_model.safetensors"],
             "subdir": "vae"},
        ],
        "components": [
            {"name": "generator.safetensors (FCDM-T2-MR thinker+UNet)", "path": "agate-preview-003/generator.safetensors", "size_gb": 0.74},
            {"name": "Ettin-68M text encoder (ModernBERT)", "path": "agate-preview-003/text_encoder/model.safetensors", "size_gb": 0.26},
            {"name": "SD-VAE-ft-MSE decoder", "path": "agate-preview-003/vae/diffusion_pytorch_model.safetensors", "size_gb": 0.33},
            {"name": "agate/ pipeline code (imported from model dir)", "path": "agate-preview-003/agate", "size_gb": 0.0},
        ],
    },
    "flux-3": {
        "backend_id": "flux3-image-klein-wan2gp",
        "dir": "flux-3-image-klein",
        "backend_type": "flux3_wan2gp",
        "lora_root": "flux3_klein",
        "category": "image",
        "bits": "INT8 ConvRot (DDM repack)",
        "description": "FLUX 3 Klein — 5.6B destilado (16 steps cfg 1.0), best-in-class text-render, edição c/ até 10 refs, via Wan2GP; SFW",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "vision",
        "default_size": (1024, 1024),
        "components": [
            {"name": "flux3_image_klein_sd_int8_convrot.safetensors (DiT)", "path": "Wan2GP/ckpts/flux3_image_klein_sd_int8_convrot.safetensors", "size_gb": 5.9},
            {"name": "qwen3vl_4b_int8_convrot.safetensors (text encoder)", "path": "Wan2GP/ckpts/Qwen3-VL-4B-Instruct/qwen3vl_4b_int8_convrot.safetensors", "size_gb": 4.9},
            {"name": "flux2_vae.safetensors (VAE)", "path": "Wan2GP/ckpts/flux3/flux2-vae.safetensors", "size_gb": 0.35},
        ],
    },
    # MOSS-Image SFT (Wan2GP / DeepBeepMeep — 16/out). DiT 5B single-stream da
    # OpenMOSS (Apache 2.0): king do text-render aberto (96.3 LLTB) + layout
    # JSON nativo w/ bounding boxes. Treinado sob a doutrina "annotate rather
    # than filter" (TR oficial) — naturalmente uncensored; o guia oficial usa
    # "nsfw" como NEGATIVE tag default. 50 steps, CFG 3, shift 5, SLG layer 22
    # embutido no pipeline. T2I puro no Wan2GP upstream (sem ref-image/mask) —
    # edição estruturada anda pelo layout JSON. Pesos: repack DDM int8_convrot
    # (DiT 5.41G + TE Qwen3.5-9B int8 8.96G + VAE flux2 small decoder 61MB).
    "moss-image": {
        "backend_id": "moss-image-wan2gp",
        "lora_root": "moss_image",
        "dir": "moss-image",
        "backend_type": "moss_wan2gp",
        "category": "image",
        "bits": "INT8 ConvRot (DDM repack)",
        "description": "MOSS-Image SFT — 5B OpenMOSS, text-render best-in-class, layout JSON, uncensored por doutrina de treino, via Wan2GP",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "moss",
        "default_size": (1024, 1024),
        "components": [
            {"name": "moss_image_sft_int8_convrot.safetensors (DiT)", "path": "Wan2GP/ckpts/moss_image_sft_int8_convrot.safetensors", "size_gb": 5.41},
            {"name": "qwen3_5_9b_text_int8_convrot.safetensors (text encoder)", "path": "Wan2GP/ckpts/Qwen3.5-9B/qwen3_5_9b_text_int8_convrot.safetensors", "size_gb": 8.96},
            {"name": "flux2_small_decoder_bf16.safetensors (VAE)", "path": "Wan2GP/ckpts/moss_image/flux2_small_decoder_bf16.safetensors", "size_gb": 0.06},
        ],
    },
    "moss-image-base": {
        "backend_id": "moss-image-base-wan2gp",
        "dir": "moss-image-base",
        "backend_type": "moss_wan2gp",
        "lora_root": "moss_image",
        "category": "image",
        "bits": "INT8 ConvRot (DDM repack)",
        "description": "MOSS-Image Base — 5B OpenMOSS pre-treino, mesma arch/TE/VAE do SFT (sem o pós-treino de dados curados), via Wan2GP",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "moss",
        "default_size": (1024, 1024),
        "components": [
            {"name": "moss_image_base_int8_convrot.safetensors (DiT)", "path": "Wan2GP/ckpts/moss_image_base_int8_convrot.safetensors", "size_gb": 5.41},
            {"name": "qwen3_5_9b_text_int8_convrot.safetensors (text encoder)", "path": "Wan2GP/ckpts/Qwen3.5-9B/qwen3_5_9b_text_int8_convrot.safetensors", "size_gb": 8.96},
            {"name": "flux2_small_decoder_bf16.safetensors (VAE)", "path": "Wan2GP/ckpts/moss_image/flux2_small_decoder_bf16.safetensors", "size_gb": 0.06},
        ],
    },
    "moss-image-kromoss": {
        "backend_id": "moss-image-kromoss-wan2gp",
        "dir": "moss-image-kromoss",
        "backend_type": "moss_wan2gp",
        "lora_root": "moss_image",
        "category": "image",
        "bits": "INT8 ConvRot (repack do lodestones no proprio repo do Kromoss)",
        "description": "Kromoss — full fine-tune do MOSS-Image Base @1024px (fix de blurriness, checkpoint 7000 EMA), mesma arch/TE/VAE/pipeline do SFT, via Wan2GP",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "moss",
        "default_size": (1024, 1024),
        "components": [
            {"name": "kromoss_int8_convrot.safetensors (DiT)", "path": "Wan2GP/ckpts/kromoss_int8_convrot.safetensors", "size_gb": 5.41},
            {"name": "qwen3_5_9b_text_int8_convrot.safetensors (text encoder)", "path": "Wan2GP/ckpts/Qwen3.5-9B/qwen3_5_9b_text_int8_convrot.safetensors", "size_gb": 8.96},
            {"name": "flux2_small_decoder_bf16.safetensors (VAE)", "path": "Wan2GP/ckpts/moss_image/flux2_small_decoder_bf16.safetensors", "size_gb": 0.06},
        ],
    },
    "moss-imagine": {
        "backend_id": "moss-imagine-wan2gp",
        "dir": "moss-imagine",
        "backend_type": "moss_wan2gp",
        "lora_root": "moss_image",
        "category": "image",
        "bits": "INT8 ConvRot (realrebelai, merge rank-32 sobre o SFT)",
        "description": "MOSS-Imagine — adaptacao do SFT com dataset gerado pelo Grok Imagine (look fotografico/cinematografico/editorial), mesma arch/TE/VAE/pipeline do SFT; tensores bf16 bit-identicos ao SFT da casa, qkv fundido (layout Comfy-Org), via Wan2GP",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "moss",
        "default_size": (1024, 1024),
        "components": [
            {"name": "moss_imagine_int8_convrot.safetensors (DiT)", "path": "Wan2GP/ckpts/moss_imagine_int8_convrot.safetensors", "size_gb": 5.41},
            {"name": "qwen3_5_9b_text_int8_convrot.safetensors (text encoder)", "path": "Wan2GP/ckpts/Qwen3.5-9B/qwen3_5_9b_text_int8_convrot.safetensors", "size_gb": 8.96},
            {"name": "flux2_small_decoder_bf16.safetensors (VAE)", "path": "Wan2GP/ckpts/moss_image/flux2_small_decoder_bf16.safetensors", "size_gb": 0.06},
        ],
    },
    "moss-image-photoreal": {
        "backend_id": "moss-image-photoreal-wan2gp",
        "dir": "moss-image-photoreal",
        "backend_type": "moss_wan2gp",
        "lora_root": "moss_image",
        "category": "image",
        "bits": "INT8 ConvRot (kimrf / Civitai 3021669 — merge sobre o SFT, layout Comfy-Org qkv fundido)",
        "description": "MOSS Photographic Realism — merge da comunidade (kimrf) sobre o MOSS-Image SFT com foco photo/lighting; BF16 norm/TE bit-identicos ao SFT da casa, delta de camada ~1.5-2.5% (régua SFT-Base). Mesma arch/TE/VAE/pipeline do SFT, via Wan2GP. Fonte Civitai: baixada à mão (auto-download do wgp não funciona).",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "moss",
        "default_size": (1024, 1024),
        "components": [
            {"name": "moss_image_photoreal_int8_convrot.safetensors (DiT)", "path": "Wan2GP/ckpts/moss_image_photoreal_int8_convrot.safetensors", "size_gb": 5.41},
            {"name": "qwen3_5_9b_text_int8_convrot.safetensors (text encoder)", "path": "Wan2GP/ckpts/Qwen3.5-9B/qwen3_5_9b_text_int8_convrot.safetensors", "size_gb": 8.96},
            {"name": "flux2_small_decoder_bf16.safetensors (VAE)", "path": "Wan2GP/ckpts/moss_image/flux2_small_decoder_bf16.safetensors", "size_gb": 0.06},
        ],
    },
    "fibo2-edit-turbo": {
        "backend_id": "fibo2-sdcpp",
        "dir": "fibo2",
        "backend_type": "fibo_sdcpp",
        "transformer": "fibo2_edit_turbo_int8_convrot.safetensors",
        "cfg_scale": 1.0,
        "category": "image",
        "bits": "INT8 ConvRot",
        "description": "BRIA Fibo 2.0 Edit Turbo — edicao multi-referencia em 4 passos (CFG 1), TE Qwen3-VL-4B, VAE FLUX.2, via sd.cpp (arch fibo2 nativa)",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "fibo",
        "default_size": (1024, 1024),
        "components": [
            {"name": "fibo2_edit_turbo_int8_convrot.safetensors (DiT)", "path": "fibo2/diffusion_models", "size_gb": 8.41},
            {"name": "Qwen3VL-4B-Instruct-Q4_K_M.gguf (text encoder)", "path": "fibo2/te", "size_gb": 2.50},
            {"name": "flux2-vae.safetensors (VAE FLUX.2)", "path": "fibo2/vae", "size_gb": 0.34},
        ],
    },
    # fibo2-dev REMOVIDO 24/out: A/B com seed e caption VGL congelados deu 263s (20 passos,
    # CFG 5) contra 39s do edit-turbo (4 passos, CFG 1) — qualidade igual ou melhor no turbo,
    # so perdendo aderencia literal ao campo lighting. Transformer de 8,4 GB deletado.
    # Unica vantagem do dev seria negative prompt (CFG 1 desliga guidance); se precisar, rebaixar.

    # Anima (CircleStone Labs × Comfy Org, 30/10/2026) — 2B T2I anime/illustration.
    # sd-cpp arch nativa; pesos em ~/.llama-models/anima (layout de frota — um jogo
    # de pesos serve o diffuse sd-cli E os fragments sd-server do llama-swap).
    # Uncensored por default (tier "Unbound" no wiki uncensored-image-gen; os safety
    # tags booru DIRIGEM, não bloqueiam). Receitas do card do criador: turbo CFG 1
    # @8-12 steps; base/aesthetic/preview 30-50 steps CFG 4-6 (aesthetic melhor no 3).
    # Resoluções 512²-1536². Sampler default: er_sde (neutro, flat, sharp).

    "anima-turbo-v1.1": {
        "backend_id": "anima-sdcpp",
        "dir": "anima-turbo-v1.1",
        "backend_type": "anima_sdcpp",
        "category": "image",
        "bits": "BF16 (turbo-v1.1, sem quant GGUF upstream)",
        "description": "Anima Turbo v1.1 — 2B anime T2I CircleStone×Comfy; distilled 10 steps CFG 1, er_sde; uncensored (Unbound). RECOMENDADO p/ começar",
        "default_size": (832, 1216),
        "cfg_scale": 1.0,
        "sample_steps": 10,
        "sampler": "er_sde",
        "weights_subdir": "turbo-v1.1",
        "diffusion_file": "anima-turbo-v1.1.safetensors",
        "te_file": "common/te/Qwen3-0.6B-Base.Q8_0.gguf",
        "vae_file": "common/vae/qwen_image_vae.safetensors",
        "components": [
            {"name": "anima-turbo-v1.1.safetensors (DiT 2B Turbo v1.1)", "path": "~/.llama-models/anima/turbo-v1.1/anima-turbo-v1.1.safetensors", "size_gb": 4.18},
            {"name": "qwen3-0.6B-base TE Q8_0 [shared]", "path": "~/.llama-models/anima/common/te/Qwen3-0.6B-Base.Q8_0.gguf", "size_gb": 0.64},
            {"name": "qwen_image VAE [shared]", "path": "~/.llama-models/anima/common/vae/qwen_image_vae.safetensors", "size_gb": 0.25},
        ],
    },
    "anima-aesthetic-v1.1": {
        "backend_id": "anima-sdcpp",
        "dir": "anima-aesthetic-v1.1",
        "backend_type": "anima_sdcpp",
        "category": "image",
        "bits": "BF16 (aesthetic-v1.1, sem quant GGUF upstream)",
        "description": "Anima Aesthetic v1.1 — 2B anime fine-tune de consistência; 40 steps CFG 3, er_sde; melhor default art style",
        "default_size": (832, 1216),
        "cfg_scale": 3.0,
        "sample_steps": 40,
        "sampler": "er_sde",
        "weights_subdir": "aesthetic-v1.1",
        "diffusion_file": "anima-aesthetic-v1.1.safetensors",
        "te_file": "common/te/Qwen3-0.6B-Base.Q8_0.gguf",
        "vae_file": "common/vae/qwen_image_vae.safetensors",
        "components": [
            {"name": "anima-aesthetic-v1.1.safetensors (DiT 2B Aesthetic v1.1)", "path": "~/.llama-models/anima/aesthetic-v1.1/anima-aesthetic-v1.1.safetensors", "size_gb": 4.18},
            {"name": "qwen3-0.6B-base TE Q8_0 [shared]", "path": "~/.llama-models/anima/common/te/Qwen3-0.6B-Base.Q8_0.gguf", "size_gb": 0.64},
            {"name": "qwen_image VAE [shared]", "path": "~/.llama-models/anima/common/vae/qwen_image_vae.safetensors", "size_gb": 0.25},
        ],
    },
    "anima-base-v1.0": {
        "backend_id": "anima-sdcpp",
        "dir": "anima-base-v1.0",
        "backend_type": "anima_sdcpp",
        "category": "image",
        "bits": "GGUF Q8_0 (Abiray)",
        "description": "Anima Base v1.0 — base máxima flexibilidade/diversidade, 40 steps CFG 5, er_sde; O CANVAS PRA TREINAR LoRA",
        "default_size": (832, 1216),
        "cfg_scale": 5.0,
        "sample_steps": 40,
        "sampler": "er_sde",
        "weights_subdir": "base-v1.0",
        "diffusion_file": "anima-base-v1.0-Q8_0.gguf",
        "te_file": "common/te/Qwen3-0.6B-Base.Q8_0.gguf",
        "vae_file": "common/vae/qwen_image_vae.safetensors",
        "components": [
            {"name": "anima-base-v1.0-Q8_0.gguf (DiT 2B Base v1.0)", "path": "~/.llama-models/anima/base-v1.0/anima-base-v1.0-Q8_0.gguf", "size_gb": 2.24},
            {"name": "qwen3-0.6B-base TE Q8_0 [shared]", "path": "~/.llama-models/anima/common/te/Qwen3-0.6B-Base.Q8_0.gguf", "size_gb": 0.64},
            {"name": "qwen_image VAE [shared]", "path": "~/.llama-models/anima/common/vae/qwen_image_vae.safetensors", "size_gb": 0.25},
        ],
    },
    "anima-preview3": {
        "backend_id": "anima-sdcpp",
        "dir": "anima-preview3",
        "backend_type": "anima_sdcpp",
        "category": "image",
        "bits": "GGUF Q8_0 (Bedovyy)",
        "description": "Anima Preview3-base — próxima geração (modelo do workflow Preview oficial da Comfy); 40 steps CFG 5, er_sde",
        "default_size": (832, 1216),
        "cfg_scale": 5.0,
        "sample_steps": 40,
        "sampler": "er_sde",
        "weights_subdir": "preview3",
        "diffusion_file": "anima-preview3-base-Q8_0.gguf",
        "te_file": "common/te/Qwen3-0.6B-Base.Q8_0.gguf",
        "vae_file": "common/vae/qwen_image_vae.safetensors",
        "components": [
            {"name": "anima-preview3-base-Q8_0.gguf (DiT 2B Preview3-base)", "path": "~/.llama-models/anima/preview3/anima-preview3-base-Q8_0.gguf", "size_gb": 2.28},
            {"name": "qwen3-0.6B-base TE Q8_0 [shared]", "path": "~/.llama-models/anima/common/te/Qwen3-0.6B-Base.Q8_0.gguf", "size_gb": 0.64},
            {"name": "qwen_image VAE [shared]", "path": "~/.llama-models/anima/common/vae/qwen_image_vae.safetensors", "size_gb": 0.25},
        ],
    }

}
