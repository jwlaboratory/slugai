# Host

Code that runs on hosts, ie, the macbooks, gaming pcs, GPUs, etc.

# Files

profile.py:
Profiles the current host for disk speed, network speed, compute. Useful so we know how much work to shard this node and rebalancing
- Run using 
- Test using


model.py
Runs a range of layers of a model with an option to run using weight streaming
- Run using
- Test using




Setup, from the repo root (needs MLX; older versions break mlx-lm's batch caches):

```
uv venv .venv && uv pip install -p .venv/bin/python "mlx>=0.32.3" "mlx-lm>=0.32" pytest
```

# Files

model.py:
- Run using `python -m host.model mlx-community/Qwen3-8B-4bit --layers 0:18,18:36 --resident 8`. It loads
  the shards on this Mac, generates greedily through them and prints the text, ms per token and peak memory.
  - `model`: a Hugging Face repo id (downloaded to the HF cache on first use; set `HF_HOME` to move it) or a local directory
  - `--layers`: each shard's layers `[start, end)`, covering the model in order (default: one shard with every layer)
  - `--resident`: layers each shard keeps in memory; the rest stream from disk (default: all)
  - `--prompt`, `--tokens`: what to generate
- Test using `python -m pytest host/tests/test_model.py`. On tiny Qwen3 (fp16, 4-bit, tied) and Qwen3-Next
  models: any split and any streaming give exactly the logits of one shard holding every layer, the tokens
  match mlx-lm's, and streamed layers aren't held in memory.
- Fits into bigger picture - each host holds one shard: a contiguous block of the model's layers. The first
  shard also holds the embedding and takes token ids; the last holds the final norm and lm_head and returns
  logits; the rest take and return hidden states, which is all that crosses the network. In code:

  ```python
  shard = Shard("mlx-community/Qwen3-8B-4bit", 18, 36, resident=8)  # layers 18-35 + head; 10 streamed
  cache = shard.new_cache(batch)                    # one per micro-batch
  out = shard.forward(hidden, cache, lengths)       # prefill: right-padded prompts, real tokens per row
  out = shard.forward(next_hidden, cache)           # decode step
  ```

  Prefill all but each prompt's last token, then run the last tokens as the first decode step, so the rows
  line up. Streaming lets a host hold more layers than fit in its memory: a streamed layer is read from disk
  while the one before it computes and dropped after use, at the cost of reading it on every step.

profile.py:
- Run using `python -m host.profile mlx-community/Qwen3-8B-4bit --context 256`. It prints the report as
  JSON: `layer_time` (micro-batch size → seconds per layer per decode step), `free_memory` (bytes for weights
  and caches) and `disk_speed` (bytes/s reading the model's files). `--batch-sizes 1,8,32` times fewer sizes.
  About 90 s for Qwen3-8B on an M4 Pro.
- Test using `python -m pytest host/tests/test_profile.py`
- Fits into bigger picture - the profiling helps know how much to shard work to this node. Time per layer is
  a layer's marginal cost: decode steps through the model's first 6 and first 2 layers alternate and the
  median difference is divided by 4, so fixed costs cancel. Steps run with `--context` tokens already cached
  (the job's typical length, capped at 1024), because attention over a longer cache costs more. Free memory
  is what macOS lets the GPU use, minus what's in use and 1 GB of headroom. Disk speed is how fast streamed
  layers can be read.
