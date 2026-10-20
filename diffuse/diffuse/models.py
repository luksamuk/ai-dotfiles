"""Model registry — all supported models and their configurations."""
from __future__ import annotations

# ── Model registry ─────────────────────────────────────────────────────────
# Each entry defines:
#   backend_type: which backend module handles generation
#   bits: quantization description
#   dir: model directory under MODELS_DIR (or ~/.llama-models/ for hidream)
#   description: human-readable description
#   enhance_model: default LLM for prompt enhancement
#   enhance_type: "ideogram" (JSON) or "vision" (natural language)
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
    # Z-Image-Turbo (diffusers ZImagePipeline + GGUF, subprocess in bonsai-venv)
    "z-image-turbo-q4": {
        "backend_id": "z-image-turbo-q4-sd-cpp",
        "dir": "z-image-turbo-q4",
        "backend_type": "zimage_sd_cpp",
        "category": "image",
        "bits": "Q3_K GGUF (auto-selected for 6GB VRAM)",
        "description": "Z-Image-Turbo — 6B S3-DiT, 8 NFE, sub-second on H800, uncensored, bilingual text",
        "default_size": (1024, 1024),
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "vision",
        "components": [
            {"name": "z_image_turbo GGUF (DiT)", "path": "z-image-turbo-q4/z_image_turbo-Q3_K.gguf", "size_gb": 3.0},
            {"name": "Qwen3-4B-Instruct-2507 Q4_K_M GGUF (text encoder)", "path": "z-image-turbo-q4/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "size_gb": 2.4},
            {"name": "Flux VAE", "path": "z-image-turbo-q4/pipeline/vae/diffusion_pytorch_model.safetensors", "size_gb": 0.32},
        ],
    },
    # HiDream-O1-Image-Dev SDNQ (transformers + accelerate CPU offload)
    "hidream-sdnq": {
        "dir": "HiDream-O1-Image-Dev-SDNQ-last8",
        "backend_type": "hidream",
        "category": "image",
        "bits": "4-bit SDNQ (uint4+svd-r32 last8-odown-bf16)",
        "description": "HiDream-O1-Image-Dev SDNQ — 8B unified (T2I + editing + IP), ~3min/2048² on 6GB VRAM",
        "default_size": (1024, 1024),
        "hidream_repo": "~/git/HiDream-O1-Image",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "vision",
        "components": [
            {"name": "Qwen3-VL 8B SDNQ (unified DiT + text encoder)", "path": "~/.llama-models/HiDream-O1-Image-Dev-SDNQ-last8", "size_gb": 7.3},
        ],
    },
    # Ideogram 4 (sd-cli / stable-diffusion.cpp)
    # Models auto-download from HuggingFace on first use (no pre-download needed).
    "ideogram4-q4": {
        "backend_id": "ideogram4-q4-sd-cpp",
        "dir": "ideogram-4-Q4_0",
        "backend_type": "sd_cpp",
        "category": "image",
        "bits": "4-bit",
        "description": "Ideogram 4 Q4_0 — 9.3B DiT, structured JSON prompts, best-in-class text rendering",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "ideogram",
        "default_size": (1024, 1024),
        "hf_files": [
            {"repo": "leejet/ideogram-4-GGUF", "files": ["ideogram4-Q4_0.gguf", "ideogram4_uncond-Q4_0.gguf"]},
            {"repo": "unsloth/Qwen3-VL-8B-Instruct-GGUF", "files": ["Qwen3-VL-8B-Instruct-Q4_K_M.gguf"],
             "rename": {"Qwen3-VL-8B-Instruct-Q4_K_M.gguf": "Qwen3VL-8B-Instruct-Q4_K_M.gguf"}},
            {"repo": "black-forest-labs/FLUX.2-dev", "files": ["ae.safetensors"], "subdir": "vae",
             "rename": {"ae.safetensors": "flux2-vae.safetensors"}},
        ],
        "components": [
            {"name": "ideogram4-Q4_0.gguf (DiT)", "path": "ideogram-4-Q4_0/ideogram4-Q4_0.gguf", "size_gb": 5.0},
            {"name": "ideogram4_uncond-Q4_0.gguf (uncond DiT)", "path": "ideogram-4-Q4_0/ideogram4_uncond-Q4_0.gguf", "size_gb": 5.0},
            {"name": "Qwen3VL-8B-Instruct-Q4_K_M.gguf (text encoder)", "path": "ideogram-4-Q4_0/Qwen3VL-8B-Instruct-Q4_K_M.gguf", "size_gb": 4.7},
            {"name": "flux2-vae.safetensors (VAE)", "path": "ideogram-4-Q4_0/vae/flux2-vae.safetensors", "size_gb": 0.2},
        ],
    },
    # Qwen-Image 2.1 (sd-cli / stable-diffusion.cpp — day-0 support, sd.cpp 137f740)
    # 7B DiT, 32 blocks, hidden 4096. T2I + native editing (up to 10 reference images) + RGBA.
    # Text encoder is Qwen3-VL-8B (NOT the Qwen2.5-VL-7B of Qwen-Image 1.0).
    # The VAE is NOT interchangeable with Qwen-Image 1.0 or Wan 2.2 — the name
    # conversion path differs (convert_diffusers_to_original_wan_vae(qwen_image_2_1=true)).
    # Dimensions MUST be multiples of 32.
    # Qwen-Image 2.1 BASE — APOSENTADO 09/10 (user: "vamos aposentar o antigo").
    # DiT qwen-image-2.1-Q4_K_M.gguf DELETED; weights re-installable from
    # abenzerps/Qwen-Image-2.1-Uncensored-GGUF (Q4_K_M 4.60 GB). Entry kept in the
    # registry on purpose (code stays; weights gone) — loader missing-error is the
    # intended guardrail. The old --pruna flag was removed (base retired; its
    # LoRA extracts p_qwen_5/8step deleted with it).
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
    # Mage-Flow-Edit-Turbo (sd-cli / stable-diffusion.cpp — instruction-based image editing)
    # 4B NR-MMDiT, 4 steps, cfg=1.0 (Turbo). Uses Qwen3-VL-4B as text+vision encoder.
    # GGUF from gguf-org, VAE included. Only edit-turbo variant (no T2I GGUF yet).
    "mageflow-edit-turbo": {
        "backend_id": "mageflow-edit-turbo-sd-cpp",
        "dir": "mageflow-edit-turbo",
        "backend_type": "mageflow_sd_cpp",
        "category": "image",
        "bits": "NVFP4 GGUF",
        "description": "Mage-Flow-Edit-Turbo — 4B NR-MMDiT, 4-step instruction-based image editing, no masks needed",
        "enhance_model": "qwen3.6-35b-a3b-heretic",
        "enhance_type": "vision",
        "hf_files": [
            {"repo": "gguf-org/mageflow-gguf",
             "files": ["mageflow-edit-turbo-nvfp4.gguf", "pig_mageflow_vae_fp32-f16.gguf"]},
            {"repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
             "files": ["Qwen3VL-4B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-4B-Instruct-F16.gguf"]},
        ],
        "components": [
            {"name": "mageflow-edit-turbo-nvfp4.gguf (DiT)", "path": "mageflow-edit-turbo/mageflow-edit-turbo-nvfp4.gguf", "size_gb": 2.3},
            {"name": "pig_mageflow_vae_fp32-f16.gguf (VAE)", "path": "mageflow-edit-turbo/pig_mageflow_vae_fp32-f16.gguf", "size_gb": 0.33},
            {"name": "Qwen3VL-4B-Instruct-Q4_K_M.gguf (text encoder)", "path": "mageflow-edit-turbo/Qwen3VL-4B-Instruct-Q4_K_M.gguf", "size_gb": 2.4},
            {"name": "mmproj-Qwen3VL-4B-Instruct-F16.gguf (vision encoder)", "path": "mageflow-edit-turbo/mmproj-Qwen3VL-4B-Instruct-F16.gguf", "size_gb": 0.8},
        ],
    },
    # FLUX 3 Image Klein (Wan2GP / DeepBeepMeep — 16/out). 5.6B BFL destilado:
    # 16 steps, cfg 1.0. T2I + edição c/ até 10 refs ordenadas (KI = main ref
    # dimensiona o output). Pipeline mmgp Profile 5 offload bloco a bloco — o
    # mesmo regime do Ming/H3 na 6GB — denoise em GPU (o sd_cpp não engolava o
    # fp8r de 6.07 GB num graph 1-segment e caía pra CPU a 67 s/it). NSFW
    # suprimido no treino (red team BFL): casa usa qwen-image-2.1 pra NSFW e
    # Klein pra texto-fino SFW/edição. Pesos em ~/git/Wan2GP/models/flux3/
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
}