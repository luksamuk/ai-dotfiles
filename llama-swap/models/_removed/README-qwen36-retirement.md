# Aposentadoria qwen3.6-35b-a3b-heretic — 02/nov/2026

Fragmens movidos pra ca (build-config.py nao empacota _removed/). GGUFs deletados:
~/.llama-models/qwen36-heretic-apex-mini/ (Qwen3.6-35B-A3B-uncensored-heretic-APEX-I-Mini.gguf 14G + mmproj 861M).

Razao (dono): ornith-1.5-35b-heretic cobre o papel de enhance/prompting; vocabularios
IDENTICOS medidos nas headers GGUF (tokenizer pre=qwen35, 248.320 tokens, bos/eos
248044/248046, embedding 2048, arch qwen35moe, ctx nativo 262144). Qwen3.6 nao tem
vocabularios maior que ornith 1.5 - a diferenca e o treino.

REINSTALL (se um dia voltar):
1. Base: HF Qwen/Qwen3.6-35B-A3B (mirror CI: positron-ai/Qwen3.6-35B-A3B)
2. Quant: APEX-I-Mini de mudler (era o que vivia em qwen36-heretic-apex-mini/)
3. mv fragmentos de volta pra models/, descomentar q36 no config-footer.yaml
   (var + evict_costs), recolocar q36 no set heavy, rodar build-config.py,
   deploy pro runtime, restart llama-swap
