# Running slugai on two MacBooks

Two MacBooks split one model's layers between them: each holds half, and
micro-batches circulate between the two so both stay busy. What we measure is
throughput, the total tokens per second across all prompts.

Start with Qwen3-8B. It fits on one Mac, so you can compare the pair against a
single Mac. Then try the 80B model: if it doesn't fit in the two Macs' memory,
the layers that don't fit stream from disk.

## What you need

- Two Apple-silicon MacBooks (M1 or later).
- Python 3.10 or newer on each. Check with `python3 --version`; stock macOS
  ships 3.9, so if that's what you see, run `brew install python`.
- One network where the Macs can reach each other directly. Many campus and
  office Wi-Fi networks block traffic between devices; if the hosts can't find
  each other, use a home router or a phone hotspot.
- Both Macs plugged in, lids open, for the whole job. The host keeps the Mac
  awake, but closing the lid still puts it to sleep.
- About 5 GB of free disk for the test model (about 45 GB for the 80B model).

## 1. Install (on both Macs)

Create a fresh environment. `--clear` replaces any older install; both Macs
must run the same version.

```bash
python3 -m venv --clear ~/slugai-venv
```

```bash
~/slugai-venv/bin/pip install "slugai[mlx] @ git+https://github.com/jwlaboratory/slugai.git"
```

## 2. Download the model and check the setup (on both Macs)

```bash
~/slugai-venv/bin/slugai prep-model mlx-community/Qwen3-8B-4bit
```

```bash
~/slugai-venv/bin/slugai doctor --model mlx-community/Qwen3-8B-4bit
```

Every line should say `ok`. Write down each Mac's "GB recommended working set"
number; you need it for the 80B model.

## 3. Start a host (on both Macs)

```bash
~/slugai-venv/bin/slugai host
```

Leave it running. If macOS asks whether Python may accept incoming connections,
click **Allow**. The host prints its address, for example
`listening on 192.168.1.23:47111`; note it in case discovery fails.

## 4. Run a job (on either Mac, in a second terminal)

```bash
~/slugai-venv/bin/slugai run --model mlx-community/Qwen3-8B-4bit --min-hosts 2 --limit 300 --max-tokens 256 --output ring.jsonl --report ring.json
```

- `--min-hosts 2` tells the server to wait for two hosts (the default is three).
- Without `--prompts` it uses 300 built-in sample prompts. To use your own, pass
  `--prompts my.jsonl` (one `{"prompt": ...}` per line) or a `.txt` file with
  one prompt per line.
- If it can't find both hosts, replace `--min-hosts 2` with
  `--hosts 192.168.1.23,192.168.1.24`, using the addresses from step 3.

The job runs through discover → profile → place → load → prefill → decode →
collect. The `placement:` line shows how the layers were split and the
micro-batch size, for example `A:0-17  B:18-35  micro-batch 128 x 3`.
Completions go to `ring.jsonl` and the numbers to `ring.json`.

On the dashboard, `[q]` stops the job and still saves every finished
completion, `[s]` saves a report, and `[r]` checks whether moving layers would
help. Over SSH, add `--no-tui`; Ctrl-C then stops and saves.

## 5. Compare with one Mac

When the job is done (the hosts go idle and can keep running), run the same
prompts on one Mac. Set `--batch` to the micro-batch from the `placement:` line.

```bash
~/slugai-venv/bin/slugai bench --model mlx-community/Qwen3-8B-4bit --limit 300 --max-tokens 256 --batch 128 --report bench.json
```

Throughput:

```bash
python3 -c "import json; print('two Macs', round(json.load(open('ring.json'))['tok_s'], 1), 'tok/s | one Mac', round(json.load(open('bench.json'))['tok_s'], 1), 'tok/s')"
```

Correctness: with the same batch size, the two runs should produce the same
tokens.

```bash
python3 -c "import json; r = [json.loads(l)['token_ids'] for l in open('ring.jsonl')]; b = [json.loads(l)['token_ids'] for l in open('bench.jsonl')]; print(sum(x == y for x, y in zip(r, b)), '/', len(r), 'completions identical')"
```

On two identical Macs expect every completion to match. If the Macs have
different chips, a few completions may diverge after some tokens, because the
GPUs round slightly differently; that's not a bug.

What to expect: with two similar Macs on decent Wi-Fi, the planner predicts
about twice one Mac's throughput at the same batch size, since each Mac
computes half the layers and three micro-batches keep both busy. That's a
prediction; it hasn't been measured on real hardware yet, so record what you
get in the table at the end.

## 6. The 80B model (optional)

If the two "recommended working set" numbers from step 2 add up to well over
45 GB (for example a 36 GB and a 48 GB MacBook, about 27 + 36 GB), the model
fits in memory. If not, slugai streams the layers that don't fit: they stay on
disk and are read for every step, FlexGen-style. The `placement:` line shows
how many, e.g. `pro:0-32 (32 from disk)`. It runs, but slowly, because reading
weights from disk every step costs far more than keeping them. On a MacBook Pro
(19 GB) plus a MacBook Air (12.7 GB) the planner predicts roughly 15-30 tok/s,
depending on how fast each disk reads; a fast SSD matters more than anything
else. Close other apps first, since the planner counts the GPU memory limit,
not what other apps are using.

On both Macs (about 45 GB each; the hosts can keep running):

```bash
~/slugai-venv/bin/slugai prep-model mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit
```

Then on either Mac:

```bash
~/slugai-venv/bin/slugai run --model mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit --min-hosts 2 --max-tokens 256 --output ring80b.jsonl --report ring80b.json
```

There's no one-Mac comparison here: no single Mac can run it.

## Troubleshooting

- **"found 1 host(s), need 2"**: discovery didn't see both hosts. Check both
  are running, then use `--hosts` with their addresses. If that fails too, the
  network is blocking traffic between devices.
- **Connections refused or timing out**: the macOS firewall is blocking Python.
  Allow it in System Settings → Network → Firewall, or click Allow when asked.
- **"host … is busy with job …"**: an earlier server is still connected. Stop
  it, or restart that host with Ctrl-C and step 3.
- **Different versions**: if one Mac was installed earlier, redo step 1 on both.
- **Testing with both hosts on one Mac** works for checking correctness, but the
  throughput numbers mean nothing: the two hosts share one GPU.

To stop a host, press Ctrl-C in its terminal.

## Results

| date | Macs (chip, memory) | model | network | two Macs tok/s | one Mac tok/s | identical completions |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |
