# slugai — slow distributed inference

Throughput-first LLM decoding across a ring of MacBooks over Wi-Fi.

We only care about throughput, not latency. Three MacBooks with different
compute and different network speeds hold a model too big for any one of them
(target: Qwen3-Next-80B-A3B-Instruct, 4-bit, ~45 GB). Each Mac holds a
contiguous block of layers and their KV cache; only hidden states cross the
network (~16 KB per sequence per token); micro-batches circulate the ring so
that communication overlaps compute.

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

`slowring` is accepted as an alias of `slugai`. Hosts are found over Bonjour;
`--hosts a:47111,b:47111,c:47111` skips discovery.

## Commands

| command | what it does |
| --- | --- |
| `slugai host` | start on each MacBook; blocks sleep and waits for the server |
| `slugai run --model M --prompts P` | start a job on the server |
| `slugai bench single\|streaming --model M --prompts P` | run a baseline on the same prompts |
| `slugai prep-model M` | stage model files on this Mac |
| `slugai profile --model M` | profile one Mac (time per layer, memory) |
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

[p] pause  [r] rebalance  [s] save report  [q] quit
```

`--no-tui` prints the same numbers as log lines instead.

## How a job runs

Every job runs the same phases automatically: **discover → profile → place →
load → prefill → decode (rebalancing as needed) → collect**. Nothing is tuned
by hand.

1. **Profile.** Each Mac measures its seconds per layer per decode step at
   several micro-batch sizes (marginal cost of extra layers, so fixed overheads
   cancel) and reports free memory. Every directed link is probed for latency
   (ping-pong) and bandwidth (4 MB transfers).
2. **Place.** With `L_i` layers on Mac i, `t_i` its time per layer, `w` weight
   bytes per layer, `k` KV bytes per layer per token, `B` sequences in flight,
   `c` the context budget and `m_i` free memory:

       min over (L_1, L_2, L_3) of max_i L_i * t_i
       s.t. L_i * (w + k * B * c) <= m_i,  sum_i L_i = L

   solved by dynamic programming for every ring order (the slowest link ends
   up on the return hop, which only carries token ids). The micro-batch size
   is the largest every link can keep up with; micro-batches are added until
   no Mac idles or KV memory runs out.
3. **Load.** Each host loads just its layers (plus the embedding on stage 1,
   the final norm and lm_head on the last stage) and connects to the next host.
4. **Prefill / decode.** The server feeds prompts into free slots; stage 1
   tokenizes and pads a micro-batch, the last stage picks the greedy token,
   streams finished sequences to the server and sends token ids back to
   stage 1. A slot is refilled as soon as all its sequences finish.
5. **Rebalance.** Live per-stage step times double as continuous profiling.
   If a Mac slows down (heat, throttling) and a re-solve predicts ≥10 % more
   throughput, the server drains the slots at the next batch boundary,
   reloads the stages with the new split and resumes. `[r]` forces a check.

## Layout

| module | role |
| --- | --- |
| `slugai/protocol.py` | length-prefixed frames: JSON header + optional fp16 tensor |
| `slugai/transport.py` | asyncio connections, heartbeats |
| `slugai/discovery.py` | Bonjour (zeroconf) advertise/browse |
| `slugai/placement.py` | layer split, ring order, micro-batch choice, rebalance |
| `slugai/backend.py` | `Stage`/`Backend` interface; numpy fake backend for tests |
| `slugai/mlx_backend.py` | MLX backend built on mlx-lm's model classes and batch caches |
| `slugai/host.py` | host daemon: profile, link probes, load layers, run a stage, stream stats |
| `slugai/server.py` | coordinator: discover, profile, place, feed prompts, collect, rebalance |
| `slugai/dashboard.py` | Textual dashboard / plain log fallback |
| `slugai/bench.py` | baselines: `single` (one Mac, all layers) and `streaming` (FlexGen-style) |
| `slugai/doctor.py`, `models.py`, `profiling.py` | setup checks, model staging, profiling CLI |

### Wire protocol

Frames are `u32 total_len | u32 header_len | JSON header | tensor bytes`. The
header carries `type` and, for tensors, `dtype`/`shape`. Control messages
(server ↔ host): `hello/welcome`, `profile/profile_result`,
`probe_link/probe_result`, `load/loaded`, `start`, `feed`, `outputs`,
`slot_free`, `stats`, `pause/resume`, `unload`, `heartbeat`, `error`, `bye`.
Ring messages (host ↔ host): `peer_hello`, `hidden` (fp16 tensor, per slot),
`tokens` (int32, last stage → stage 1), `link_probe/link_probe_ack`.

### MLX backend

Built on mlx-lm so every architecture it supports works without per-model
code: the model is loaded lazily, `model.model.layers` is sliced to the
stage's range and only those weights are evaluated. Middle stages swap the
embedding and final norm for identities so hidden states flow straight
through. KV caches are mlx-lm's batch caches (`BatchKVCache`, and
`ArraysCache` for the linear-attention layers of hybrid models such as
Qwen3-Next), which handle padding and per-sequence RoPE offsets.

## Development

```
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

The suite runs without MLX: a numpy fake backend (a small causal transformer
with real KV caches and padding masks) stands in for the model, so the ring,
scheduling, rebalancing and placement are exercised end to end on one machine
with real sockets. `slugai host --backend fake` runs the same thing by hand.

The MLX backend has its own tests that build tiny Qwen3 (fp16 and 4-bit) and
Qwen3-Next (linear attention + MoE) models on disk and check that the 3-stage
ring reproduces mlx-lm's greedy tokens exactly. They run on Apple silicon, or
on Linux with the CPU build: `pip install "mlx[cpu]" mlx-lm transformers`.

## Prior work

- [FlexGen](https://arxiv.org/abs/2303.06865): throughput-first inference by streaming weights; the basis of `bench streaming`.
- [GPipe](https://arxiv.org/abs/1811.06965): micro-batch pipelining, the scheduling idea behind the ring.
- [TD-Pipe](https://arxiv.org/abs/2506.10470): separate prefill and decode phases to remove pipeline bubbles.
- [Helix](https://arxiv.org/abs/2406.01566): layer placement on uneven GPUs and links as a max-flow problem.
- [Petals](https://arxiv.org/abs/2209.01188) and [exo](https://github.com/exo-explore/exo): layer pipelines across consumer machines, tuned for interactive use rather than throughput.
