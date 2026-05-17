# Troubleshooting

### Out of Memory (OOM) Errors

If you see CUDA OOM errors:

1. Check which model is loaded: `llama-swap-cli ps`
2. Unload it: `llama-swap-cli unload <model_id>`
3. Try a smaller context: reduce `--ctx-size` or `--fit-ctx`
4. Ensure no other GPU processes are running (Wayland, browser, etc.)

### Model Not Loading

1. Check logs: `journalctl --user -u llama-swap -f`
2. Verify the GGUF path in config matches the actual file: `ls ~/.llama-models/*.gguf`
3. Verify the binary path: `ls ~/git/llama.cpp/build/bin/llama-server`
4. Try loading the model directly (without llama-swap) to isolate the issue:
   ```bash
   ~/git/llama.cpp/build/bin/llama-server --model ~/.llama-models/Qwen3.5-4B-UD-Q3_K_XL.gguf --port 8080
   ```

### Slow Inference

- **MoE models**: Ensure you're using ik_llama.cpp (pinned memory for expert offload)
- **Dense models on ik**: Switch to upstream llama.cpp — ik is slower for dense
- **Check GPU utilization**: `nvidia-smi` — if GPU% is low, the model may be CPU-side
- **Context size**: Larger contexts require more KV cache. Try reducing `--ctx-size`