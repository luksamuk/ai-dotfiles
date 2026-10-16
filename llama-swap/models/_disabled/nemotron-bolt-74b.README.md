# Bolt 74B Experimentos - APOSENTADO 15/out/2026
- Veredito: decode 4.87 t/s no mainline v0.7.0 -ncmoe 88 (a única rota que sobe) < critério 10 t/s
- vanilla -ncmoe: NÃO CABE no 6GB (dense 4.8GB+compute > 5.1GB free; fit aborta c/ tensor overrides)
- cafe -ssd: check_tensor_dims rejeita packing NVIDIA (240 vs 496 declarados)
- GGUF deletado 15/out (23.7GB recuperados). Re-download: nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
- Tweet de referência (3060 12GB = 24 t/s): x.com/DogukanUrker/status/2110768114556101028
- Fragmento funcionou E2E no mainline; se voltar com GPU maior: usar receita mainline, NÃO cafe (packing)
