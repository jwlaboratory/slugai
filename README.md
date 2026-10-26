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
slugai host                                         # wait for a server
```

On the server (any machine; may be one of the Macs):

```
pip install slugai
slugai run --model mlx-community/Qwen3-8B-4bit --prompts prompts.jsonl
```

`slowring` is accepted as an alias of `slugai`.

## Commands

| command | what it does |
| --- | --- |
| `slugai host` | start on each MacBook; blocks sleep and waits for the server |
| `slugai run --model M --prompts P` | start a job on the server |
| `slugai bench single\|streaming --model M --prompts P` | run a baseline on the same prompts |
| `slugai prep-model M` | stage model files on this Mac |
| `slugai profile --model M` | profile one Mac (time per layer, memory) |
| `slugai doctor` | check a Mac's setup |

Every job runs the same phases automatically: **discover → profile → place →
load → prefill → decode (rebalancing as needed) → collect**. Nothing is tuned
by hand: the server measures each Mac's time per layer and free memory and
every link's bandwidth and latency, then solves

    min over (L_1, L_2, L_3) of max_i L_i * t_i
    s.t. L_i * (w + k * B * c) <= m_i,  sum_i L_i = L

by brute force (about 3,000 splits × 6 ring orders), puts the slowest link on
the return hop (which only carries token ids), and picks a micro-batch size
every link can keep up with.

## Layout

| module | role |
| --- | --- |
| `slugai/protocol.py` | length-prefixed frames: JSON header + optional fp16 tensor |
| `slugai/transport.py` | asyncio connections, heartbeats |
| `slugai/discovery.py` | Bonjour (zeroconf) advertise/browse |
| `slugai/placement.py` | layer split, ring order, micro-batch choice |
| `slugai/profiling.py` | per-Mac compute profile, link probes |
| `slugai/backend.py` | stage execution: MLX backend, numpy fake backend for tests |
| `slugai/host.py` | host daemon: profile, load layers, run a stage, stream stats |
| `slugai/server.py` | coordinator: discover, profile, place, feed prompts, collect |
| `slugai/dashboard.py` | Textual dashboard |
| `slugai/bench.py` | baselines |

## Development

```
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

The test suite runs without MLX: a numpy "fake" backend stands in for the
model so the ring, scheduling and placement logic are exercised end to end on
one machine (`slugai host --backend fake`).
