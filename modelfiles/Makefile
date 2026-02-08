.PHONY: all glm lfm translategemma mistral devstral gpt \
        qwen3-coder qwen3-coder-next nomic zephyr glm-ocr \
        extra super

all: glm translategemma mistral gpt qwen3-coder nomic

# macro genérica:
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

glm:
	$(call BUILD,glm-4.7-flash:q4_K_M-64k,glm-4.7-flash.modelfile)

lfm:
	$(call BUILD,lfm2.5-thinking:1.2b-32k,lfm2.5-thinking.modelfile)

translategemma:
	$(call BUILD,translategemma:12b-32k,translategemma.modelfile)

mistral:
	$(call BUILD,mistral-small3.2:24b-32k,mistral-small3.2.modelfile)

devstral:
	$(call BUILD,devstral-small-2:24b-32k,devstral-small-2.modelfile)

gpt:
	$(call BUILD,gpt-oss:20b-64k,gpt-oss.modelfile)

qwen3-coder:
	$(call BUILD,qwen3-coder:30b-64k,qwen3-coder.modelfile)

qwen3-coder-next:
	$(call BUILD,qwen3-coder-next:q4_K_M-32k,qwen3-coder-next.modelfile)

# pull-only targets

nomic:
	ollama pull "nomic-embed-text-v2-moe:latest"

glm-ocr:
	ollama pull "glm-ocr:bf16"

# agregados

zephyr:
	$(call BUILD,zephyr:7b-32k,zephyr.modelfile)

extra: lfm zephyr devstral
super: qwen3-coder-next
