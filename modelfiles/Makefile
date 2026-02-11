.PHONY: all extra super \
        glm-4.7-flash lfm2.5-thinking translategemma mistral-small3.2 \
        devstral-small-2 gpt-oss qwen3-coder qwen3-coder-next nomic zephyr \
        glm-ocr llava smollm3 llama3.2 pepe

# must-have models
all: glm-4.7-flash translategemma mistral-small3.2 gpt-oss qwen3-coder nomic \
     devstral-small-2 pepe

# extra models
extra: lfm2.5-thinking zephyr llava smollm3 llama3.2 sead

# super models that you probably can't even run
super: qwen3-coder-next

# generic macro for building targets:
# $(call BUILD,<model-name>,<modelfile>)
define BUILD
	@set -e; \
	base=$$(awk 'NR==1 && $$1=="FROM" {print $$2}' "$(2)"); \
	test -n "$$base"; \
	echo "==> create $(1)"; \
	ollama create "$(1)" -f "$(2)"; \
	echo "==> rm base $$base"; \
	ollama rm "$$base"
endef

glm-4.7-flash:
	$(call BUILD,glm-4.7-flash:q4_K_M-64k,glm-4.7-flash.modelfile)

lfm2.5-thinking:
	$(call BUILD,lfm2.5-thinking:1.2b-32k,lfm2.5-thinking.modelfile)

translategemma:
	$(call BUILD,translategemma:12b-32k,translategemma.modelfile)

mistral-small3.2:
	$(call BUILD,mistral-small3.2:24b-32k,mistral-small3.2.modelfile)

devstral-small-2:
	$(call BUILD,devstral-small-2:24b-64k,devstral-small-2.modelfile)

gpt-oss:
	$(call BUILD,gpt-oss:20b-64k,gpt-oss.modelfile)

qwen3-coder:
	$(call BUILD,qwen3-coder:30b-64k,qwen3-coder.modelfile)

qwen3-coder-next:
	$(call BUILD,qwen3-coder-next:q4_K_M-32k,qwen3-coder-next.modelfile)

zephyr:
	$(call BUILD,zephyr:7b-32k,zephyr.modelfile)

llava:
	$(call BUILD,llava:7b-32k,llava-7b.modelfile)

smollm3:
	$(call BUILD,smollm3:Q8_0-64k,smollm3-3b.modelfile)

sead:
	$(call BUILD,sead:14b-32k,sead-14b.modelfile)

llama3.2:
	$(call BUILD,llama3.2:3b-32k,llama3.2.modelfile)

pepe:
	$(call BUILD,pepe:8b-64k,assistant-pepe.modelfile)

# pull-only targets

nomic:
	ollama pull "nomic-embed-text-v2-moe:latest"

glm-ocr:
	ollama pull "glm-ocr:bf16"

