all: glm translategemma mistral gpt qwen3-coder nomic

glm:
	ollama create "glm-4.7-flash:q4_K_M-64k" -f glm-4.7-flash.modelfile
	ollama rm "glm-4.7-flash:q4_K_M"

lfm:
	ollama create "lfm2.5-thinking:1.2b-32k" -f lfm2.5-thinking.modelfile
	ollama rm "lfm2.5-thinking:1.2b"

translategemma:
	ollama create "translategemma:12b-32k" -f translategemma.modelfile
	ollama rm "translategemma:12b"

mistral:
	ollama create "mistral-small3.2:24b-32k" -f mistral-small3.2.modelfile
	ollama rm "mistral-small3.2:24b"

devstral:
	ollama create "devstral-small-2:24b-32k" -f devstral-small-2.modelfile
	ollama rm "devstral-small-2:24b"

gpt:
	ollama create "gpt-oss:20b-32k" -f gpt-oss.modelfile
	ollama rm "gpt-oss:20b"

qwen3-coder:
	ollama create "qwen3-coder:30b-64k" -f qwen3-coder.modelfile
	ollama rm "qwen3-coder:30b"

qwen3-coder-next:
	ollama create "qwen3-coder-next:q4_K_M-32k" -f qwen3-coder-next.modelfile
	ollama rm "qwen3-coder-next:q4_K_M"

nomic:
	ollama pull "nomic-embed-text-v2-moe:latest"

zephyr:
	ollama create "zephyr:7b-32k" -f zephyr.modelfile
	ollama rm "zephyr:7b"

glm-ocr:
	ollama pull "glm-ocr:bf16"

extra: lfm zephyr devstral
super: qwen3-coder-next
