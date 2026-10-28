# HW09: Hardware Check

You must understand your hardware to do a good job on the project. Set up a local
Python environment, run the supplied fixed workloads, and find the largest
**successfully tested** Qwen3.5 size for each mode within the experiment's budget.
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
PY hardware_check.py --suite --models 0.8b 2b 4b --device auto --download
```

The aliases select `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, and
`Qwen/Qwen3.5-4B`, in that order. The runner explicitly loads their **text
backbones** with `Qwen3_5ForCausalLM`; no image/video inputs or vision training
are used. `--download` permits model/tokenizer downloads as each size becomes
needed. Downloads use SafeTensors and `trust_remote_code=False`, and their time
is excluded from benchmark timing. The resolved checkpoint commit is recorded.
Benchmark workers use cached files only, with Hugging Face offline mode enabled.

The suite runs the following configurations in fresh processes. A mode advances
to the next size only after success at its current size.

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
parameters remain FP32. It first loads an FP32 model, so it measures the limits
of this concrete loading path as well as execution. Disk offload also first
loads FP32 weights in RAM; it is not a demonstration of loading a model larger
than RAM. Its per-run scratch files are removed after the worker exits. The
model download cache is retained. Both offload inference modes preload
`Qwen3_5GatedDeltaNet` weights because its convolution weights are accessed
directly. Training does not use inference dispatch hooks.

If time and storage remain, adding `9b` to `--models` is optional. To avoid
repeating earlier sizes, run an individual mode at 9B only after that mode
succeeded at 4B. For example:

```text
PY hardware_check.py --mode download --model 9b
PY hardware_check.py --mode fp32 --model 9b --device auto
```

For optional CUDA QLoRA, install a supported stable
[bitsandbytes wheel](https://huggingface.co/docs/bitsandbytes/installation) with
`PY -m pip install bitsandbytes`, then add `--with-qlora` to the suite command.
This adds NF4 frozen-base adapter training with FP32 compute and two complete
updates. It uses a single CUDA device, not inference `device_map="auto"`.
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
or budget stop is also a valid result. A `missing_dependency` status requires completing the setup commands and rerunning; it is not a hardware limit. The CSV distinguishes these statuses;
`worker_exit` only means the worker ended without returning a result.

The suite allows at most **30 minutes of benchmark worker time**, with at most
**180 seconds per configuration including loading**. Downloads are separately
supervised, up to 30 minutes per size. `--timeout` and `--total-minutes` can lower
these limits. By default, the worker's RSS is limited to at most 70% of available
host RAM and the assignment's cache plus scratch to 32 GiB; at least 1 GiB of
system RAM and disk remain reserved. Change `--disk-gib` to the storage budget
available on your machine, or lower `--ram-gib` to keep more RAM free.
Download/offload storage is checked before writing. The monitor is best-effort
sampling, not an OS quota. Stop if the machine becomes unresponsive.

Host RSS is sampled every 0.1 seconds and includes imports/loading/warm-up/work.
CUDA peaks are allocator peaks, excluding the driver and other processes. MPS
records an end sample in the CSV, not a peak; the CSV labels
this limitation. Apple GPU memory shares physical memory with the host, so do
not add these values as independent physical memories. This exercise measures
successful tested sizes under stated limits, not a universal maximum model size.

For a single rerun, use `--mode fp32|int8|disk|full|lora|cpu-offload|qlora`,
`--model 0.8b|2b|4b|9b`, and the actual `--device`. INT8/disk require CPU;
CPU-offload/QLoRA require CUDA. `--activation-offload` can be added to CUDA
training modes. `--revision COMMIT_SHA` fixes a single-model run to the recorded
revision. `--dry-run` prints a plan without validating dependencies or hardware.
`--self-test` runs tiny random Qwen3.5 checks offline and writes a separate test
CSV; those checks do not count as student model measurements.

Official references: [Qwen3.5 models](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[Transformers text-only class](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[dynamic INT8](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html),
[Accelerate offload](https://huggingface.co/docs/accelerate/package_reference/big_modeling),
[saved-tensor CPU offload](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu),
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
