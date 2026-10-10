Patch NAO incluido no v263 upstream (camada diferente do disableKeepAlives).
Reaplicar: clone upstream -> git apply 0001-sse-keepalive-15s.patch -> make build. Sintoma que motiva: UI reset total em reconnect por conexao SSE idle dropada por intermediario.
Backup binario com o patch INTEIRO: ~/.local/bin/llama-swap.v261-sse.bak (rollback instantaneo).
