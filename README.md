# slugai — slow distributed inference

Throughput-first LLM decoding across a ring of MacBooks over Wi-Fi.

We only care about throughput, not latency. Three MacBooks with different
compute and different network speeds hold a model too big for any one of them
(target: Qwen3-Next-80B-A3B-Instruct, 4-bit, ~45 GB). Each Mac holds a
contiguous block of layers and their KV cache; only hidden states cross the
network (4 KB per sequence per token for Qwen3-Next's 2048-wide bf16 states);
micro-batches circulate the ring so that communication overlaps compute.

```
           hidden states            hidden states
  Mac A  ───────────────▶  Mac B  ───────────────▶  Mac C
  layers 0-31              layers 32-61             layers 62-79 + lm_head
     ▲                                                   │
     └──────────────── token ids (4 B/seq) ◀─────────────┘
```

## Install

On each MacBook (needs MLX and the model files):

```
pip install "slugai[mlx]"
slugai prep-model mlx-community/Qwen3-8B-4bit     # stage model files locally
slugai doctor --model mlx-community/Qwen3-8B-4bit  # check the setup
slugai host                                         # blocks sleep, waits for a server
```

On the server (any machine; may be one of the Macs):

```
pip install slugai
slugai run --model mlx-community/Qwen3-8B-4bit --prompts prompts.jsonl
```

Hosts are found over Bonjour; `--hosts a,b,c` (or `a:47111,b:47111,...`)
skips discovery. Without `--prompts`, jobs use 300 built-in sample prompts —
add `--limit N` for a quick try.

## Commands

| command | what it does |
| --- | --- |
| `slugai host` | start on each MacBook; blocks sleep and waits for the server |
| `slugai run --model M [--prompts P]` | start a job on the server |
| `slugai bench --model M [--prompts P]` | baseline: the same prompts on one Mac holding every layer |
| `slugai prep-model M` | stage model files on this Mac |
| `slugai doctor [--model M]` | check a Mac's setup |

Prompts: a `.txt` with one prompt per line, or `.jsonl` with `{"prompt": ...}`.
Completions go to `<job>.jsonl` (`--output`), a JSON report to `--report`.
Decoding is greedy; instruct models get their chat template applied.

### Server screen

```
slugai · job demo-01 · mlx-community/Qwen3-8B-4bit · 3 hosts
phase 6/7 decode         elapsed  1:12:40   ETA  2:05:10

HOST        STAGE  LAYERS   BUSY  STEP      MEMORY
MacBook-A   1      0-31      97%  1.52 s    35/48 GB
MacBook-B   2      32-61     99%  1.56 s    33/36 GB
MacBook-C   3      62-79     95%  1.49 s    20/24 GB

LINK             USED        CAPACITY    CARRIES
MacBook-A > MacBook-B  0.7 MB/s   48.0 MB/s   hidden states
MacBook-B > MacBook-C  0.7 MB/s   31.0 MB/s   hidden states
MacBook-C > MacBook-A  0.2 KB/s    9.0 MB/s   token ids

41.0 tok/s · 148k tok/h · 61% of peak
1,212 / 3,000 sequences · 4 micro-batches x 64 · 310,272 tokens

[r] rebalance  [s] save report  [q] quit
```

`--no-tui` prints the same numbers as log lines instead. `[q]` (or Ctrl+Q, or
Ctrl-C with `--no-tui`) stops the job and still writes the completions
finished so far; a second Ctrl-C aborts without writing.

## How a job runs

Every job runs the same phases automatically: **discover → profile → place →
load → prefill → decode (rebalancing as needed) → collect**. Nothing is tuned
by hand.

1. **Profile.** Each Mac measures its seconds per layer per decode step at
   several micro-batch sizes (marginal cost of extra layers, so fixed overheads
   cancel) and reports free memory. Every directed link is probed for latency
   (ping-pong) and bandwidth (4 MB transfers).
2. **Place.** With `L_i` layers on Mac i, `t_i` its time per layer, `w` weight
   bytes per layer, `k` KV bytes per layer per token, `s` fixed cache bytes per
   layer per sequence (linear-attention state), `B` sequences in flight, `c`
   the context budget (rounded up to the KV cache's 256-token blocks) and
   `m_i` free memory:

       min over (L_1, L_2, L_3) of max_i L_i * t_i
       s.t. L_i * (w + B * (k * c + s)) <= m_i,  sum_i L_i = L

   solved by dynamic programming for every ring order (the slowest link ends
   up on the return hop, which only carries token ids). `k` and `s` are read
   off the model's own caches, so hybrid models like Qwen3-Next (~2 MB of
   state per linear-attention layer per sequence) are counted correctly. The
   micro-batch size is the largest every link can keep up with, capped so even
   a small job fills one micro-batch per Mac; micro-batches are added until no
   Mac idles or memory runs out.
3. **Load.** Each host loads just its layers (plus the embedding on stage 1,
   the final norm and lm_head on the last stage) and connects to the next host.
4. **Prefill / decode.** The server feeds prompts into free slots; stage 1
   tokenizes and pads a micro-batch, the last stage picks the greedy token,
   streams finished sequences to the server and sends token ids back to
   stage 1. A slot is refilled as soon as all its sequences finish.
5. **Rebalance.** Live per-stage step times double as continuous profiling:
   each Mac's profiled time per layer is scaled by measured / predicted for
   its current layers. If a re-solve at those speeds predicts ≥10 % more
   throughput than the current layout at the same speeds (a Mac slowed down
   from heat or throttling, or was mis-profiled), the server drains the slots
   at the next batch boundary, reloads the stages with the new split and
   resumes. `[r]` forces a check and takes any faster layout.

## Layout

| path | role |
| --- | --- |
| `slugai/cli.py`, `__main__.py` | `slugai` command (also `python -m slugai`) |
| `slugai/placement.py` | layer split, ring order, micro-batch choice, rebalance |
| `slugai/prompts.py`, `data/` | prompt files; the built-in sample prompts |
| `slugai/models.py` | model files on a host (`prep-model`; they live in the Hugging Face cache, set `HF_HOME` to move it) |
| `slugai/backend.py` | runs a range of a real model's layers with MLX: stage, tokenizer, model sizes, profiling |
| `slugai/net/` | `protocol` (frames: JSON header + optional tensor), `transport` (asyncio connections, heartbeats), `discovery` (Bonjour) |
| `slugai/ring/` | `host` (daemon: profile, link probes, load layers, run a stage), `server` (coordinator: discover → collect, rebalance), `dashboard` (Textual screen / plain logs) |
| `slugai/tools/` | `bench` (the single-Mac baseline), `doctor` |
| `tests/` | unit tests, backend tests and end-to-end ring tests on tiny real models built on disk (`tiny_model.py`, `conftest.py`, `helpers.py`) |

### Wire protocol

Frames are `u32 total_len | u32 header_len | JSON header | tensor bytes`. The
header carries `type` and, for tensors, `dtype`/`shape`. Control messages
(server ↔ host): `hello/welcome`, `profile/profile_result`,
`probe_link/probe_result`, `load/loaded`, `start/started`, `feed`,
`outputs`, `slot_free`, `stats`, `unload/unloaded`, `heartbeat`, `error`,
`bye`. Ring messages (host ↔ host):
`peer_hello`, `hidden` (per slot, in the model's dtype; bf16 travels as its
raw 16 bits), `tokens` (int32, last stage → stage 1),
`link_probe/link_probe_ack`.

### Backend

Built on mlx-lm so every architecture it supports works without per-model
code: the model is loaded lazily, `model.model.layers` is sliced to the
stage's range and only those weights are evaluated. Middle stages swap the
embedding and final norm for identities so hidden states flow straight
through. KV caches are mlx-lm's batch caches (`BatchKVCache`, and
`ArraysCache` for the linear-attention layers of hybrid models such as
Qwen3-Next), which handle padding and per-sequence RoPE offsets. Hidden
states stay in the model's dtype between Macs, so a ring produces exactly the
tokens one Mac would. Needs mlx ≥ 0.32.3 and mlx-lm ≥ 0.32; older versions
break the batch caches.

## Development

```
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev,mlx]" transformers
.venv/bin/python -m pytest
```

The tests build tiny random Qwen3 (fp16 and 4-bit) and Qwen3-Next (linear
attention + MoE) models on disk in milliseconds and run everything on them:
the ring end to end with real sockets (it must reproduce mlx-lm's greedy
tokens exactly), rebalancing, the dashboard, the baseline and the CLI. They
need MLX: Apple silicon, or on Linux the CPU build
(`pip install "mlx[cpu]" mlx-lm transformers`). Without MLX only the
placement, protocol, CLI, `doctor` and `prep-model` tests run; the rest are
skipped.

To try a real ring on one Mac, start three hosts on different ports and point
a job at them, then check the tokens against `slugai bench` with the same
batch size:

```
for p in 47201 47202 47203; do slugai host --port $p --name mac$p --no-advertise & done
slugai run --model mlx-community/Qwen3-8B-4bit --hosts 127.0.0.1:47201,127.0.0.1:47202,127.0.0.1:47203 \
    --limit 120 --max-tokens 64 --micro-batch 32 --num-micro-batches 3 --no-tui
slugai bench --model mlx-community/Qwen3-8B-4bit --limit 120 --max-tokens 64 --batch 32
```

Each host reports the whole Mac's free memory, so pin the micro-batches as
above: left to itself, placement would over-commit the shared memory.

## Prior work

- [FlexGen](https://arxiv.org/abs/2303.06865): throughput-first inference on one machine by streaming weights from disk.
- [GPipe](https://arxiv.org/abs/1811.06965): micro-batch pipelining, the scheduling idea behind the ring.
- [TD-Pipe](https://arxiv.org/abs/2506.10470): separate prefill and decode phases to remove pipeline bubbles.
- [Helix](https://arxiv.org/abs/2406.01566): layer placement on uneven GPUs and links as a max-flow problem.
- [Petals](https://arxiv.org/abs/2209.01188) and [exo](https://github.com/exo-explore/exo): layer pipelines across consumer machines, tuned for interactive use rather than throughput.
