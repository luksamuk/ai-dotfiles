.PHONY: all extra super \
        glm-4.7-flash lfm2.5-thinking translategemma mistral-small \
        devstral-small-2 gpt-oss qwen3-coder qwen3-coder-next nomic zephyr \
        glm-ocr llava smollm3 llama3.2 llama3.1 assistant-pepe nemotron-3-nano \
        ministral

# must-have models
all: translategemma glm-ocr llama3.1 \
     ministral qwen3-coder nemotron-3-nano gpt-oss assistant-pepe  \
     cloud-models nomic

# extra models
extra: lfm2.5-thinking mistral-small zephyr llava smollm3 llama3.2 sead \
       devstral-small-2 deepseek-coder-v2 glm-4.7-flash

# other models that you might not need to install
other: glm-4.7-flash

# super models that you probably can't even run
super: qwen3-coder-next

# generic macro for building targets:
# $(call BUILD,<model-name>,<modelfile>)
define BUILD
	@set -e; \
	base=$$(awk 'NR==1 && $$1=="FROM" {print $$2}' "$(2)"); \
	test -n "$$base"; \
	echo "==> create $(1)"; \
	ollama create "$(1)" -f "$(2)";
endef

# generic macro for building targets:
# $(call BUILD,<model-name>,<modelfile>)
define BUILD_DELBASE
	@set -e; \
	base=$$(awk 'NR==1 && $$1=="FROM" {print $$2}' "$(2)"); \
	test -n "$$base"; \
	echo "==> create $(1)"; \
	ollama create "$(1)" -f "$(2)"; \
	echo "==> rm base $$base"; \
	ollama rm "$$base"
endef

glm-4.7-flash:
	$(call BUILD,glm-4.7-flash:q4_K_M,glm-4.7-flash.modelfile)

lfm2.5-thinking:
	$(call BUILD,lfm2.5-thinking:1.2b,lfm2.5-thinking.modelfile)

translategemma:
	$(call BUILD,translategemma:12b,translategemma.modelfile)

mistral-small:
	$(call BUILD,mistral-small:24b,mistral-small.modelfile)

devstral-small-2:
	$(call BUILD,devstral-small-2:24b,devstral-small-2.modelfile)

gpt-oss:
	$(call BUILD,gpt-oss:20b,gpt-oss.modelfile)

qwen3-coder:
	$(call BUILD,qwen3-coder:30b,qwen3-coder.modelfile)

qwen3-coder-next:
	$(call BUILD,qwen3-coder-next:q4_K_M-32k,qwen3-coder-next.modelfile)

zephyr:
	$(call BUILD,zephyr:7b,zephyr.modelfile)

llava:
	$(call BUILD,llava:7b,llava-7b.modelfile)

smollm3:
	$(call BUILD_DELBASE,smollm3:Q8_0,smollm3-3b.modelfile)

sead:
	$(call BUILD_DELBASE,sead:14b,sead-14b.modelfile)

llama3.2:
	$(call BUILD,llama3.2:3b,llama3.2.modelfile)

assistant-pepe:
	$(call BUILD_DELBASE,assistant-pepe:8b,assistant-pepe.modelfile)

deepseek-coder-v2:
	$(call BUILD,deepseek-coder-v2:16b,deepseek-coder-v2.modelfile)

nemotron-3-nano:
	$(call BUILD,nemotron-3-nano:30b,nemotron-3-nano.modelfile)

ministral:
	$(call BUILD,ministral-3:14b,ministral.modelfile)

# pull-only targets

nomic:
	ollama pull "nomic-embed-text-v2-moe:latest"

glm-ocr:
	ollama pull "glm-ocr:bf16"

llama3.1:
	ollama pull "llama3.1:8b"

# cloud targets
cloud-models:
	ollama pull "glm-5:cloud"
	ollama pull "minimax-m2.5:cloud"
	ollama pull "qwen3-coder-next:cloud"
	ollama pull "kimi-k2.5:cloud"

