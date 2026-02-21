# Ollama Latency Optimization (RTX 4080 + 16GB RAM)

Target: **sub-2s end-to-end** response time.

## Run Ollama with optimized flags

```powershell
# Flash attention + full GPU layers (2x faster)
$env:OLLAMA_FLASH_ATTENTION = "1"
ollama serve
```

Or create `OLLAMA_FLASH_ATTENTION=1` in your environment before starting Ollama.

## Per-model context (num_ctx)

| Model | num_ctx | Use |
|-------|---------|-----|
| DictaLM 2.0 | 8192 | Planner, Reflector, Personality |
| Qwen3:4b | 4096 | Tool calls, self-evolution, JSON |

JARVIS passes `num_ctx` via API. For Modelfile overrides:

```
FROM aminadaven/dictalm2.0-instruct:q5_K_M
PARAMETER num_ctx 8192
```

## Optional: vLLM or exllama2 for Qwen (2x faster)

For maximum speed, run Qwen3:4b via vLLM or exllama2 instead of Ollama. Requires separate setup.
