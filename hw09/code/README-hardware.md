# HW09: Hardware Check

You must understand your hardware to do a good job on the project. Set up a local
Python environment, run the supplied fixed workloads, and find the largest
**successfully tested** model size for each mode within the experiment's budget.
Only `hardware-results.csv` is submitted. The runner writes it automatically; no
written answers, screenshots, environment export, or edited code are required.
Model size earns no extra credit.

## 1. Set up the environment

Use Python **3.11 or 3.12**. Download `hardware_check.py` into a new working folder.
The commands below avoid shell activation and execution-policy changes.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch
.venv\Scripts\python.exe -m pip install "transformers==5.21.0" "accelerate==1.15.0" "peft==0.21.0" psutil
```

macOS/Linux (bash or zsh):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install 'transformers==5.21.0' 'accelerate==1.15.0' 'peft==0.21.0' psutil
```

If using Python 3.11 on Windows, replace `py -3.12` with `py -3.11`. On macOS/Linux,
check `python3 --version` before creating the environment. For NVIDIA CUDA on
Windows/Linux, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
in place of the plain `torch` installation line: **replace the selector's `pip`
or `pip3` with this environment's Python path followed by `-m pip`**. Keep its
package names and `--index-url` exactly as supplied. Thus packages are installed
inside `.venv`, even though it is not activated. Do not change GPU drivers for
this assignment. CPU works on all three platforms; Apple Silicon can use MPS.

In all commands below, replace `PY` with `.venv\Scripts\python.exe` on Windows
or `.venv/bin/python` on macOS/Linux. `PY` is a placeholder, not a shell command.

```text
PY hardware_check.py --mode setup --device auto
```

This checks imports, chooses CUDA, MPS, or CPU, executes a small matrix
multiplication, and records the OS, CPU, RAM, and accelerator in the CSV.
If the selected accelerator is unsupported, rerun with `--device cpu` and use
`--device cpu` in the next command too. Bring the generated error log to course
staff if the CPU setup or smallest CPU baseline cannot complete.

## 2. Launch the fixed stress test

```text
PY hardware_check.py --suite --device auto --download
```

The default ladder selects `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`,
`Qwen/Qwen3.5-4B`, and `Qwen/Qwen3.5-9B`, in that order. It then independently
tests `inclusionAI/Ling-3.0-tiny` (alias `ling`) and
`JetBrains/Mellum2.1-12B-A2.5B-Thinking` (alias `mellum`).
The runner explicitly loads the Qwen models' **text
backbones** with `Qwen3_5ForCausalLM`; no image/video inputs or vision training
are used. Ling and Mellum use native `AutoModelForCausalLM` support in the
pinned Transformers version, with eager expert execution and no remote code.
The script includes the Linux-tested Ling compatibility fix: it normalizes the
checkpoint's integer `kda_lower_bound` to a float and restores its low-rank query
projections. Ling defaults to the reviewed checkpoint
`9a98e35fe1c9ee255f78dd64771c7ae15a799481`; the compatibility path requires that
revision and Transformers 5.21.0. It rejects missing, unexpected, or mismatched
weights and an incorrect parameter count before benchmarking. Loading diagnostics
are saved locally in a `.loading.json` file; no extra helper download is needed.
Mellum workers automatically set `HF_DEACTIVATE_ASYNC_LOAD=1` to load checkpoint
shards sequentially and reduce temporary host-RAM use. The FP32 workload is unchanged.
`--download` permits model/tokenizer downloads as each size becomes
needed. Downloads use SafeTensors and `trust_remote_code=False`, and their time
is excluded from benchmark timing. The resolved checkpoint commit is recorded.
The requested revision (such as `main`) is cached too, so the individual offline
commands below work after downloading.
Benchmark workers use cached files only, with Hugging Face offline mode enabled.

The suite runs the following configurations in fresh processes. A mode advances
to the next Qwen size only after success at its current size. Ling and Mellum
each start with all applicable modes, even if a Qwen mode or another model's
download failed. All families share the same time and storage budgets.
Use `--models ling mellum` to run only the two additional models.

Ling has about **7.9B total / 1.3B active** parameters; Mellum has
**12B total / 2.5B active**. Active parameters describe computation per token,
not the amount of weight storage. Their original checkpoints occupy roughly
15.8 GB and 24.3 GB, respectively, before FP32 conversion. Retaining earlier
checkpoints can exhaust the default 32 GiB cache/scratch budget; that is a
recorded storage limit. Increase `--disk-gib` only to fit your available storage.

| Mode in CSV | Concrete workload |
| --- | --- |
| `fp32` | FP32 inference on the selected device. If this is a GPU, a separate CPU FP32 control is also run. |
| `int8` | CPU inference after dynamically packing linear weights into INT8. This also provides a quantized path on Macs; it runs on CPU, not MPS. |
| `disk` | FP32 CPU inference with weights memory-mapped from disk through Accelerate. |
| `full` | Two FP32 AdamW updates of all text-model parameters on the selected device. |
| `lora` | Two AdamW updates of rank-8 adapters on all supported linear layers, with the FP32 base frozen. |
| `cpu-offload` (CUDA only) | FP32 inference with CPU weights staged to CUDA as needed. |
| `lora+activation-offload` (CUDA only) | The same adapter training while saved activations are offloaded to CPU. |

All modes use seed 0, batch size 1, four CPU threads, and exactly 128 tokens of
fixed text. Inference does one untimed one-token warm-up, then greedily generates
exactly 16 new tokens with a cache. Tokens/s includes both prompt processing and
generation. Training performs exactly two optimizer steps at learning rate
`1e-4`, with no AMP, gradient accumulation, or KV cache. Labels are the unshifted
input token IDs; the causal loss shifts them internally. LoRA uses rank 8,
alpha 16, zero dropout, and `target_modules="all-linear"`. A successful row
requires completed generation or both finite-loss training steps.

INT8 reports its engine and actual packed modules in the local JSON. Other
parameters remain FP32, including fused MoE expert tensors that are not
`nn.Linear` modules. Thus this INT8 mode does not quantize every MoE weight.
It first loads an FP32 model, so it measures the limits
of this concrete loading path as well as execution. Disk offload also first
loads FP32 weights in RAM; it is not a demonstration of loading a model larger
than RAM. Its per-run scratch files are removed after the worker exits. The
model download cache is retained. Both offload inference modes preload
`Qwen3_5GatedDeltaNet` and `BailingHybridDeltaAttention` weights because their
convolution weights are accessed directly. Training does not use inference
dispatch hooks. LoRA uses PEFT's supported targets; the JSON records the actual
trainable parameter count, and not every model uses identical adapter targets.

The suite reaches 9B only for modes that succeeded at 4B and while benchmark
time and storage remain. Use `--models 0.8b 2b 4b` to request a shorter ladder.

For optional CUDA QLoRA, install a supported stable
[bitsandbytes wheel](https://huggingface.co/docs/bitsandbytes/installation) with
`PY -m pip install bitsandbytes`, then add `--with-qlora` to the suite command.
This adds NF4 linear-weight adapter training with FP32 compute and two complete
updates. It uses a single CUDA device, not inference `device_map="auto"`.
Nonlinear-module weights, including unsupported fused experts, remain FP32.
QLoRA and accelerator modes are optional when unsupported. An unsupported
backend is a reported limit, not an instruction to compile native kernels.

## 3. Submit the automatically generated CSV

Submit **only `hardware-results.csv`**. It records the machine's RAM/accelerator,
model and resolved revision, mode/device/precision, success or stopping status,
time, throughput or seconds/step, sampled host memory, and CUDA peak allocations
where available. Keep the generated `.jsonl` and `-logs/` folder locally for
troubleshooting; they are not required for submission.

Use the largest successful row for each mode as the measured hardware limit for
this fixed workload. If all attempted sizes succeed, that is a valid completed
ladder: do not force a failure. A timeout, allocator OOM, unsupported backend,
or budget stop is also a valid result. Exit code **0** means all requested runs
completed; **2** means a resource/backend limit was recorded; **1** means a setup
or runner error that must be fixed and rerun before submission. Check the CSV:
`missing_dependency`, `error`, and `worker_exit` are not hardware limits.
`worker_exit` means the worker exited abnormally or returned no valid result.
Self-tests return 1 for any non-success, including resource limits.

The suite allows at most **30 minutes of benchmark worker time**, with at most
**180 seconds per configuration including loading**. Downloads are separately
supervised, up to 30 minutes per size. `--timeout` and `--total-minutes` can lower
these limits. By default, one RSS cap is set to at most 95% of available
host RAM at invocation start and held fixed across that suite. The assignment's
cache plus scratch is limited to 32 GiB; at least 1 GiB of
system RAM and disk remain reserved. Change `--disk-gib` to the storage budget
available on your machine, or lower `--ram-gib` to keep more RAM free.
After each worker, the runner sleeps for at least 5 seconds, then waits up to
30 seconds total for available RAM to recover to 90% of its initial value.
These pauses are excluded from benchmark timing; recovery details stay in the
local JSONL. The live 1 GiB RAM reserve remains enforced if recovery is incomplete.
Download/offload storage is checked before writing. The monitor is best-effort
sampling, not an OS quota. Stop if the machine becomes unresponsive.
Every numeric budget must be finite and positive; `NaN` and infinite limits are
rejected before any worker starts. On Linux, CPU identification falls back to
`/proc/cpuinfo` when the platform library returns an empty name.

Host RSS is sampled every 0.1 seconds and includes imports/loading/warm-up/work.
CUDA peaks are allocator peaks, excluding the driver and other processes. MPS
records an end sample in the CSV, not a peak; the CSV labels
this limitation. Apple GPU memory shares physical memory with the host, so do
not add these values as independent physical memories. This exercise measures
successful tested sizes under stated limits, not a universal maximum model size.

For a single rerun, use `--mode fp32|int8|disk|full|lora|cpu-offload|qlora`,
`--model 0.8b|2b|4b|9b|ling|mellum`, and the actual `--device`. INT8/disk require CPU;
CPU-offload/QLoRA require CUDA. `--activation-offload` can be added to CUDA
training modes. `--revision COMMIT_SHA` fixes a single-model run to the recorded
revision. `--dry-run` prints a plan without validating dependencies or hardware.
`--self-test` runs tiny random Qwen3.5 checks offline and writes a separate test
CSV; those checks do not count as student model measurements.

Official references: [Qwen3.5 models](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny),
[Mellum2.1 Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking),
[sequential checkpoint loading](https://huggingface.co/docs/transformers/en/reference/environment_variables#model-loading),
[Transformers text-only class](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[dynamic INT8](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html),
[Accelerate offload](https://huggingface.co/docs/accelerate/package_reference/big_modeling),
[saved-tensor CPU offload](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu),
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
