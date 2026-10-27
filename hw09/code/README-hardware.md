# HW09: Hardware Check

Measure a **largest successful tested model for a fixed workload and budget** on
your own Windows, macOS, or Linux computer. Model size earns no extra credit.
The question PDF gives the submission requirements; this guide supplies commands.

## Setup

Use Python 3.10 or newer in an isolated environment. The following commands avoid
shell activation and execution-policy changes. Substitute `py` for `python` on
Windows if that is your installed Python launcher.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
# Install the appropriate PyTorch wheel using https://pytorch.org/get-started/locally/
.venv\Scripts\python.exe -m pip install "transformers>=4.46,<6" "accelerate>=1.0,<2" "peft>=0.15,<1" psutil
.venv\Scripts\python.exe hardware_check.py --mode fp32 --dry-run
```

macOS/Linux (bash/zsh):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Install the appropriate PyTorch wheel using https://pytorch.org/get-started/locally/
.venv/bin/python -m pip install 'transformers>=4.46,<6' 'accelerate>=1.0,<2' 'peft>=0.15,<1' psutil
.venv/bin/python hardware_check.py --mode fp32 --dry-run
```

For a CPU or supported macOS wheel, the PyTorch selector may give the plain
`python -m pip install torch` command. For NVIDIA hardware use its command for
your OS and supported CUDA runtime. A CPU-only installation works for the core
assignment. Do not install a different GPU driver just for this problem.

In the commands below, `PY` means the environment's Python executable:
`.venv\Scripts\python.exe` in PowerShell or `.venv/bin/python` in bash/zsh.
Replace `PY` with that path; `PY` is not a command or variable assignment.
Each command is one line and works in either shell after that replacement.
Record exact installed versions (the results include them); dependencies and
kernel availability vary by platform. `--dry-run` needs only Python and prints
the plan without installing packages, downloading weights, or validating a backend.

## Run the small baseline

Download the **base** (not `-Instruct`) 135M checkpoint once. The three aliases are
`135m`, `360m`, and `1.7b`, resolving to `HuggingFaceTB/SmolLM2-*` models. There is
no dataset download and no remote model code. Only tokenizer/configuration files
and SafeTensors weights are downloaded.

```text
PY hardware_check.py --mode download --model 135m
PY hardware_check.py --mode fp32 --model 135m --device cpu
PY hardware_check.py --mode int8 --model 135m --device cpu
PY hardware_check.py --mode disk --model 135m --device cpu
PY hardware_check.py --mode full --model 135m --device cpu
PY hardware_check.py --mode lora --model 135m --device cpu
```

The download result gives the resolved commit SHA. For strict repeatability,
add `--revision THE_COMMIT_SHA` to each subsequent command. The five benchmark
modes load exclusively from the local cache; missing files produce a logged
error instead of triggering a download. No training checkpoint is written.

If supported, also run `fp32`, `full`, and `lora` with `--device cuda` (NVIDIA on
Windows/Linux) or `--device mps` (Apple GPU on macOS). Keep the CPU FP32 control
when comparing with CPU INT8. Do not interpret a difference between GPU FP32 and
CPU INT8 as the effect of quantization alone. If MPS encounters an unsupported
operator, log it and use the CPU path; this harness does not silently enable MPS
CPU fallback. Intel/AMD accelerators can use the CPU baseline; an independently
documented native backend is an optional extension.

All runs use seed 0, four CPU threads by default, batch size 1, a fixed text truncated
to exactly 128 tokens, eager attention, and no AMP or compilation. Inference
does one untimed one-token warm-up, then greedily generates exactly 16 tokens
with a KV cache. Timed inference includes prefill and generation; it is **not**
a decode-only tokens/s measurement. Training performs two AdamW updates at
`1e-4`, without KV cache, gradient accumulation, or gradient checkpointing. The
two finite losses and completed optimizer steps test feasibility, not quality.

## Quantization and offload mechanisms

- `int8` dynamically packs `nn.Linear` weights into CPU INT8 modules. Embeddings,
  layer norms, and other tensors remain FP32. The result reports the selected
  quantized engine, number of packed modules, and packed linear-weight bytes
  (excluding metadata). It checks that at least one module was quantized. Native
  INT8 kernels may be unavailable; an exact unsupported/error result is acceptable.
  Conversion begins with an FP32 model in RAM, so this mode can fail during
  loading even if the eventual quantized model would fit. It does not claim
  training support or universal fourfold model memory savings.
- `disk` uses `accelerate.disk_offload` to memory-map FP32 weights and stage
  them to the CPU for computation. It writes a revision-specific subdirectory
  under `hardware-offload/` and reports its size. This small mechanism check
  initially loads the model into RAM. It **does not** establish that a checkpoint
  larger than RAM can be loaded. CPU execution is explicit. OS page-cache hits
  can conceal physical disk I/O; record this limitation.
- `cpu-offload` uses `accelerate.cpu_offload` to stage FP32 weights from host RAM
  to CUDA a layer at a time. It is an inference mode. These inference hooks are
  not used by any training mode.
- `lora` freezes the FP32 base and trains rank-8 query/value adapters
  (`alpha=16`, no dropout, no bias). An alternate model must have `q_proj` and
  `v_proj` modules, or you must declare and document your target-module change.
- `--activation-offload` uses PyTorch's `torch.autograd.graph.save_on_cpu` for
  saved training tensors on CUDA. It stages saved activations to host RAM and
  brings them back when autograd needs them. It does not offload optimizer
  states or base weights. On a tiny input its speed/memory benefit may be small.

On a supported CUDA system, compare a small inference run and a small training run:

```text
PY hardware_check.py --mode cpu-offload --model 135m --device cuda
PY hardware_check.py --mode lora --model 135m --device cuda --activation-offload
```

For optional QLoRA, first check the **stable installed version's**
[bitsandbytes installation support](https://huggingface.co/docs/bitsandbytes/installation).
Its CPU/Apple/Intel/AMD support and CUDA capabilities vary with releases; this
particular harness implements QLoRA only on CUDA. Do not assume that an Apple
CPU wheel implies MPS kernels. If supported, install `bitsandbytes` in the same
environment and run:

```text
PY -m pip install bitsandbytes
PY hardware_check.py --mode qlora --model 135m --device cuda
```

This mode uses a frozen NF4 4-bit base with double quantization, FP32 compute,
and the same rank-8 adapters, plus two complete updates. It uses an explicit
single-device map `{"": 0}`, **never inference `device_map="auto"` for training**.
It does not imply that packed 4-bit base weights are directly fine-tuned.
If this optional backend is unsupported, log that fact and explain the estimates
without trying to repair or compile the backend.

## Resource bounds and the ladder

The supervisor starts a fresh child process for each invocation, records its
stderr in a per-run log, and appends JSON to `hardware-results.jsonl`. It samples
worker/child RSS every 0.1 s and stops the worker at the first of:

- 180 seconds by default, including load/setup; benchmarks are capped at 180
  seconds, while a separate download can use `--timeout 600` for up to 600 seconds;
- worker RSS exceeding 70% of initially available host RAM, or system available
  RAM falling below 1 GiB;
- more than 8 GiB in the assignment's cache and offload directories.

You can lower limits with `--timeout`, `--ram-gib`, and `--disk-gib`; RAM cannot
be raised beyond the default host reserve. The monitor is best-effort sampling,
not an OS memory quota: brief spikes can be missed. GPU memory is bounded by the
device allocator and OOM handling, not the host RSS cap. Stop the experiment if
other work makes the machine unresponsive. Keep the combined benchmark wall
time to 30 minutes yourself; the script does not track a course-wide session.

Before download, the script conservatively adds the declared matching file sizes
to existing cache/offload bytes and checks free space with a 1 GiB reserve.
This can overestimate cache reuse. If a cached model is already available, skip
`download`. On Windows without symlink support, Hugging Face can store duplicate
files; actual cached bytes are monitored. The CPU disk demo can add roughly
`4P` bytes, so include it in your storage planning. No cache/offload files are
automatically deleted.

Repeat the feasible modes at `360m`, then `1.7b`, one model at a time, reusing
downloads. Apply the PDF's memory estimates **before** attempting the next rung.
Stop a mode at its first OOM, timeout, RAM/disk budget stop, unsupported backend,
or predicted budget violation. Do not repeat a harmful run just to obtain OOM.
If all three sizes fit and time/storage remain, you may declare a longer
increasing ladder of ungated compatible models. Pass its exact Hub ID using
`--model`, use a fixed commit SHA, and document architecture changes. The harness
uses eager attention and PEFT query/value module names; arbitrary model families
may need a documented adaptation. The assignment does not require downloading
multi-billion-parameter models beyond the default ladder.

## Reading the results

Use one row per mode/model/device. Include success or the exact stopping status;
`ram_budget` and `disk_budget` are supervised stops, distinct from allocator OOM.
`worker_exit` can mean an external process kill; do not relabel it OOM without
evidence. The JSON contains the requested model/revision, resolved revision,
versions, timings, fixed workload, trainable count, generation or losses,
sampled peak RSS, and CUDA allocator peaks or an MPS current sample.

`load_setup_seconds` includes local loading, quantization/offload setup, tokenizing,
and adapter preparation. The process RSS peak covers imports/loading/warm-up/work.
CUDA peak allocated/reserved statistics cover the worker after CUDA initialization;
they exclude driver allocations and other processes. An MPS current allocation
sample is not a peak. Do not sum host RSS and MPS allocations as independent
physical memory on Apple unified memory.

In QLoRA, Transformers' parameter-count routine may apply quantization-specific
counting; for the memory formulas use the same checkpoint's FP32 parameter count
from its baseline/configuration. Adapter parameter counts are separately reported.
A checksum difference for the first trainable tensor is diagnostic only, not a
proof of the number of updated tensors. A row succeeds only after generation or
both optimizer updates complete, not merely when weights load.

Report a separate largest successful tested size for FP32 inference, quantized
inference, offloaded inference, FP32 full fine-tuning, FP32 LoRA, and optional
QLoRA. If nothing succeeds for a mode, state that and provide its measured error
or estimate. Give the next attempted/estimated rung, stopping reason, and budget.
This is a reproducible lower bound for this workload, not a global maximum or a
language-quality benchmark.

## Official references

- [SmolLM2 model card and available sizes](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch dynamic quantization API](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html)
- [Accelerate CPU/disk offload APIs](https://huggingface.co/docs/accelerate/package_reference/big_modeling)
- [Accelerate inference dispatch and its memory limits](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)
- [PyTorch saved-tensor CPU offload](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu)
- [PEFT quantization and QLoRA preparation](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PEFT LoRA API](https://huggingface.co/docs/peft/package_reference/lora)
- [bitsandbytes platform installation matrix](https://huggingface.co/docs/bitsandbytes/installation)
