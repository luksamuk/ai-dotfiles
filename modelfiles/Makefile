.PHONY: all extra super \
        glm-4.7-flash lfm2.5-thinking translategemma mistral-small \
        devstral-small-2 gpt-oss qwen3-coder qwen3-coder-next nomic zephyr \
        glm-ocr llava smollm3 llama3.2 llama3.1 assistant-pepe nemotron-3-nano \
        ministral hermes3 nouscoder hermes4 qwen3.5-27b lfm2 \
        qwen2.5 mistral-nemo hermes4.3 qwen3.5-9b jan-code \
        qwen3.5-abliterated-9b qwen3.5-4b qwen3.5-0.8b qwen3.5 \
        nemotron-cascade-2-30b nanbeige4.1-3b \
        ministral-3b falcon3-3b falcon3-1b \
        hermes2-pro-mistral-7b

# Qwen 3.5 small variants
qwen3.5-small: qwen3.5-9b qwen3.5-4b qwen3.5-2b qwen3.5-0.8b
qwen3.5-small-abliterated: qwen3.5-abliterated-9b qwen3.5-abliterated-4b \
                           qwen3.5-abliterated-2b qwen3.5-abliterated-0.8b

# must-have models
all: essentials
essentials: nomic \
            glm-ocr \
            qwen3.5-small \
            translategemma \
            ministral \
            qwen2.5-coder-7b \
            cloud-models

# extra models
extra: mistral-small zephyr llava smollm3 llama3.2 sead \
       devstral-small-2 deepseek-coder-v2 glm-4.7-flash hermes3 \
       nouscoder hermes4 lfm qwen3.5-9b qwen2.5 mistral-nemo mistral hermes4.3 \
       jan-code nemotron-3-nano nemotron-3-nano-4b gpt-oss qwen3-coder qwen3 \
       lfm2 qwen3.5-27b hermes4 nouscoder

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
# $(call BUILD_DELBASE,<model-name>,<modelfile>)
define BUILD_DELBASE
	@set -e; \
	base=$$(awk '/^FROM / {print $$2; exit}' "$(2)"); \
	test -n "$$base"; \
	echo "==> create $(1)"; \
	ollama create "$(1)" -f "$(2)"; \
	echo "==> rm base $$base"; \
	sleep 2; \
	ollama rm "$$base"
endef

glm-4.7-flash:
	$(call BUILD,glm-4.7-flash:q4_K_M,glm-4.7-flash.modelfile)

translategemma:
	$(call BUILD,translategemma:4b,translategemma.modelfile)

mistral-small:
	$(call BUILD,mistral-small:24b,mistral-small.modelfile)

devstral-small-2:
	$(call BUILD,devstral-small-2:24b,devstral-small-2.modelfile)

gpt-oss:
	$(call BUILD,gpt-oss:20b,gpt-oss.modelfile)

qwen3:
	$(call BUILD,qwen3:8b,qwen3-8b.modelfile)

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

nemotron-3-nano-4b:
	$(call BUILD,nemotron-3-nano:4b,nemotron-3-nano-4b.modelfile)

ministral:
	$(call BUILD,ministral-3:14b,ministral.modelfile)

hermes3:
	$(call BUILD,hermes3:8b,hermes3.modelfile)

nouscoder:
	$(call BUILD_DELBASE,nouscoder:14b,nouscoder.modelfile)

hermes4:
	$(call BUILD_DELBASE,hermes4:14b,hermes4.modelfile)

lfm2:
	$(call BUILD,lfm2:24b,lfm2.modelfile)

qwen2.5:
	$(call BUILD,qwen2.5:7b,qwen2.5.modelfile)

qwen2.5-coder-7b:
	ollama pull qwen2.5-coder:7b
	ollama create qwen2.5-coder:7b -f qwen2.5-coder-7b.modelfile

mistral-nemo:
	$(call BUILD,mistral-nemo:12b,mistral-nemo.modelfile)

hermes4.3:
	$(call BUILD_DELBASE,hermes4.3:36b,hermes4.3.modelfile)

jan-code:
	$(call BUILD_DELBASE,jan-code:4b,jan-code.modelfile)

# Qwen 3.5 Family

qwen3.5-0.8b:
	$(call BUILD,qwen3.5:0.8b,qwen3.5-0.8b.modelfile)

qwen3.5-2b:
	$(call BUILD,qwen3.5:2b,qwen3.5-2b.modelfile)

qwen3.5-4b:
	$(call BUILD,qwen3.5:4b,qwen3.5-4b.modelfile)

qwen3.5-9b:
	$(call BUILD,qwen3.5:9b,qwen3.5-9b.modelfile)

qwen3.5-27b:
	$(call BUILD,qwen3.5:27b,qwen3.5.modelfile)

# Qwen 3.5 Family (Abliterated)

qwen3.5-abliterated-9b:
	$(call BUILD_DELBASE,qwen3.5-abliterated:9b,qwen3.5-9b-abliterated.modelfile)

qwen3.5-abliterated-4b:
	$(call BUILD_DELBASE,qwen3.5-abliterated:4b,qwen3.5-4b-abliterated.modelfile)

qwen3.5-abliterated-2b:
	$(call BUILD_DELBASE,qwen3.5-abliterated:2b,qwen3.5-2b-abliterated.modelfile)

qwen3.5-abliterated-0.8b:
	$(call BUILD_DELBASE,qwen3.5-abliterated:0.8b,qwen3.5-0.8b-abliterated.modelfile)

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

nemotron-cascade-2-30b:
	$(call BUILD,nemotron-cascade-2:30b,nemotron-cascade-2-30b.modelfile)

# Experimental models from other companies
nanbeige4.1-3b:
	$(call BUILD_DELBASE,nanbeige4.1:3b,nanbeige4.1-3b.modelfile)

ministral-3b:
	$(call BUILD,ministral-3:3b,ministral-3b.modelfile)

falcon3-3b:
	$(call BUILD_DELBASE,falcon3:3b,falcon3-3b.modelfile)

falcon3-1b:
	$(call BUILD_DELBASE,falcon3:1b,falcon3-1b.modelfile)

# FC-enabled models from other companies
hermes2-pro-mistral-7b:
	$(call BUILD_DELBASE,hermes2-pro-mistral:7b,hermes2-pro-mistral-7b.modelfile)