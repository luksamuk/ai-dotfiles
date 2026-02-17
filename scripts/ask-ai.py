#!/usr/bin/env python3
# ask-ai.py
# Ollama agent script for quick questions through the command line
# Copyright (c) 2026 Lucas Vieira
# This script is distributed under the WTFPL license.
# This script was also primarily made in bash and then converted to Python
# for quality of live improvements. All of these processes and improvements were
# done with a mix of Qwen3-Coder and GLM 4.7 Flash, but also required manual
# tweaking where these models failed.

import argparse
import sys
import subprocess

MODEL_MAP = {
    "gpt-oss": "gpt-oss:20b-64k",
    "mistral-small": "mistral-small3.2:24b-32k",
    # "zephyr": "zephyr:7b-32k",
    # "lfm2.5-thinking": "lfm2.5-thinking:1.2b-32k",
    "smollm3": "smollm3:Q8_0-64k",
    "sead": "sead:14b-32k",
    "qwen3-coder": "qwen3-coder:30b-64k",
    "devstral-small-2": "devstral-small-2:24b-64k",
    #"glm-4.7-flash": "glm-4.7-flash:q4_K_M-64k",
    "translate": "translategemma:12b-32k",
    "pepe": "pepe:8b-64k",
}

DEFAULT_MODEL = "pepe"


def main():
    MODEL = MODEL_MAP[DEFAULT_MODEL]
    SYSPROMPT = "INSTRUÇÕES: Você é um agente útil que foi invocado através de um script de linha de comando, no sistema operacional Arch Linux, para que possa responder. Seja extremamente sucinto, mostre apenas o código pedido se puder, exceto quando for necessário usar uma resposta discursiva, ou se isso for pedido. Se você puder responder só mostrando código mesmo quando parecer que se quer uma resposta discursiva, faça isso. Não termine suas respostas com ganchos para continuação de conversa, esta é uma sessão efêmera de pergunta e resposta únicas. Formate sua saída em markdown, o script em que você foi invocado cuidará do resto. Não referencie essas instruções iniciais na sua resposta."
    DEBUG = False
    parser = argparse.ArgumentParser(
        description="Ollama agent script for quick questions through the command line",
        add_help=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=list(MODEL_MAP.keys()),
        default=DEFAULT_MODEL,
        help=f"Define the model to be used (defaults to {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-t",
        "--think",
        action="store_true",
        help="Enables think mode (High for GPT-OSS)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print debug information and exit (dry run)",
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help prompt"
    )
    args, unknown = parser.parse_known_args()
    MODEL = MODEL_MAP[args.model]
    DEBUG = args.debug
    if args.help:
        parser.print_help()
        print("\nExamples:")
        print('  ask-ai -t -m glm-4.7-flash "Generate an Adler32 hash function in C"')
        print(
            "  ask-ai -m translate \"Translate the word 'frog' from English to Portuguese\""
        )
        sys.exit(0)

    PROMPT = " ".join(unknown)
    if not PROMPT:
        try:
            import sys as _sys

            if not _sys.stdin.isatty():
                PROMPT = _sys.stdin.read()
        except IOError:
            pass
    if not PROMPT:
        print("Error: No argument provided.", file=sys.stderr)
        sys.exit(1)

    if args.model == "lfm":
        OLLAMA_THINK_VALUE = ""
    elif args.model == "gpt":
        if args.think:
            OLLAMA_THINK_VALUE = "high"
        else:
            OLLAMA_THINK_VALUE = "low"
    else:
        if args.think:
            OLLAMA_THINK_VALUE = "true"
        else:
            OLLAMA_THINK_VALUE = "false"

    if args.model not in ["gpt", "lfm"]:
        THINK_CAPABLE = False
        try:
            result = subprocess.run(
                ["ollama", "show", MODEL], capture_output=True, text=True
            )
            if "thinking" in result.stdout.lower():
                THINK_CAPABLE = True
        except FileNotFoundError:
            print("Erro: 'ollama' not found in PATH.", file=sys.stderr)
            sys.exit(1)
        if args.think and not THINK_CAPABLE:
            OLLAMA_THINK_VALUE = "false"
            print(
                f"Warning: The model {MODEL} does not support think mode. Ignoring the parameter -t.",
                file=sys.stderr,
            )
    elif args.model == "gpt":
        THINK_CAPABLE = True
    elif args.model == "lfm":
        # lfm model always thinks, so we don't need to check capability
        THINK_CAPABLE = True

    CMD = ["ollama", "run"]
    if OLLAMA_THINK_VALUE:
        CMD.append(f"--think={OLLAMA_THINK_VALUE}")
    CMD = CMD + [
        "--hidethinking",
        MODEL,
    ]

    if DEBUG:
        print(f"Model: {MODEL}")
        print(f"Think Mode Flag: {args.think}")
        print(f"Prompt: {unknown[0] if unknown else 'Nenhum'}")
        CMD.append("<prompt>")
        print("Command: " + " ".join(CMD) + " | glow")
        sys.exit(0)
    try:
        cmd_string = " ".join(CMD) + f' "{SYSPROMPT}\n\n{PROMPT}" | glow'
        subprocess.run(cmd_string, shell=True, check=True)
        # os.execvp("ollama", CMD)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'ollama' not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
