#!/usr/bin/env python3
"""HW09 bounded local LLM feasibility probe. See README-hardware.md.

No network is used by benchmark modes; download is an explicit separate mode.
Each invocation supervises a fresh worker. --dry-run needs only Python.
"""
import argparse
import contextlib
import fnmatch
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

GIB = 2 ** 30
MODELS = {s.lower(): "HuggingFaceTB/SmolLM2-" + s
          for s in ("135M", "360M", "1.7B")}
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.tiktoken",
            "tokenizer.*", "merges.txt", "vocab.*"]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["download", "fp32", "int8", "disk",
                   "cpu-offload", "full", "lora", "qlora"], required=True)
    p.add_argument("--model", default="135m", help="135m, 360m, 1.7b, or Hub ID")
    p.add_argument("--revision", default="main", help="commit SHA recommended")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--cache", default="hardware-cache")
    p.add_argument("--offload", default="hardware-offload")
    p.add_argument("--output", default="hardware-results.jsonl")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--ram-gib", type=float, help="worker RSS cap; default <=70%% available")
    p.add_argument("--disk-gib", type=float, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--activation-offload", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker-result", help=argparse.SUPPRESS)
    a = p.parse_args()
    a.model = MODELS.get(a.model.lower(), a.model)
    if min(a.timeout, a.disk_gib, a.threads) <= 0 or (a.ram_gib is not None and a.ram_gib <= 0):
        p.error("time, storage, threads and RAM limits must be positive")
    if a.mode in ("int8", "disk") and a.device != "cpu":
        p.error("int8 and disk use --device cpu; compare with FP32 CPU")
    if a.mode in ("cpu-offload", "qlora") and a.device != "cuda":
        p.error("this harness implements cpu-offload and qlora on CUDA only")
    if a.activation_offload and (a.device != "cuda" or a.mode not in ("full", "lora", "qlora")):
        p.error("activation offload requires a CUDA training mode")
    return a


def folder_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*")
               if p.is_file() and not p.is_symlink()) if Path(path).exists() else 0


def versions():
    result = {"python": platform.python_version(), "os": platform.platform(),
              "cpu": platform.processor()}
    for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "psutil"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def worker(a):
    from huggingface_hub import HfApi, snapshot_download
    if a.mode == "download":
        info = HfApi().model_info(a.model, revision=a.revision, files_metadata=True)
        wanted = [x for x in info.siblings if any(fnmatch.fnmatch(x.rfilename, pat) for pat in PATTERNS)]
        if any(x.size is None for x in wanted):
            raise RuntimeError("Unknown download size: inspect model files before downloading")
        needed = sum(x.size for x in wanted)
        if needed + folder_bytes(a.cache) + folder_bytes(a.offload) > a.disk_gib * GIB:
            raise RuntimeError("Conservative download estimate exceeds storage budget; reuse cache or stop")
        if shutil.disk_usage(Path(a.cache).resolve().parent).free < needed + GIB:
            raise RuntimeError("Insufficient free disk space with 1 GiB reserve")
        start = time.perf_counter()
        path = snapshot_download(a.model, revision=info.sha, cache_dir=a.cache,
                                 allow_patterns=PATTERNS, max_workers=2)
        return {"status": "ok", "snapshot": path, "revision": info.sha,
                "download_seconds": time.perf_counter() - start,
                "conservative_download_bytes": needed}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise NotImplementedError("CUDA unavailable in this PyTorch installation")
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise NotImplementedError("MPS unavailable in this PyTorch installation")
    if a.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()

    def sync():
        if a.device == "cuda":
            torch.cuda.synchronize()
        elif a.device == "mps":
            torch.mps.synchronize()

    start = time.perf_counter()
    snapshot = snapshot_download(a.model, revision=a.revision, cache_dir=a.cache,
                                 local_files_only=True, allow_patterns=PATTERNS)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                              trust_remote_code=False)
    kw = dict(local_files_only=True, trust_remote_code=False,
              use_safetensors=True, torch_dtype=torch.float32,
              attn_implementation="eager")
    if a.mode == "qlora":
        from transformers import BitsAndBytesConfig
        kw.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32),
            device_map={"": 0})  # explicit single training device, never "auto"
    model = AutoModelForCausalLM.from_pretrained(snapshot, **kw)
    count = model.num_parameters()
    # Build exactly 128 tokens from a public, fixed string; no dataset required.
    text = "Attention combines information from earlier tokens. " * 128
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :128]
    if ids.shape[1] != 128:
        raise RuntimeError("Tokenizer did not produce 128 input tokens")
    result = {"revision": Path(snapshot).name, "parameter_count": count,
              "input_tokens": 128, "batch_size": 1, "seed": 0,
              "device": a.device, "precision": "FP32", "quantized_linear_modules": 0}
    if a.mode == "int8":
        engines = torch.backends.quantized.supported_engines
        engine = next((x for x in ("x86", "fbgemm", "qnnpack", "onednn") if x in engines), None)
        if engine is None:
            raise NotImplementedError("No supported CPU dynamic INT8 engine: " + str(engines))
        torch.backends.quantized.engine = engine
        model.eval()
        torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear},
                                              dtype=torch.qint8, inplace=True)
        qmods = [m for m in model.modules()
                 if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
        if not qmods:
            raise NotImplementedError("No nn.Linear modules were quantized in this architecture")
        result.update(precision="INT8 linear weights; remaining tensors FP32",
                      quantized_linear_modules=len(qmods), quantized_engine=engine,
                      packed_linear_weight_bytes=sum(m.weight().numel() * m.weight().element_size() for m in qmods))
    elif a.mode == "disk":
        from accelerate import disk_offload
        # Small mechanism demonstration: loading itself still needs host RAM.
        subdir = Path(a.offload) / (Path(snapshot).name + "-fp32")
        subdir.mkdir(parents=True, exist_ok=True)
        disk_offload(model, offload_dir=str(subdir), execution_device=torch.device("cpu"))
        result["offload_directory"] = str(subdir.resolve())
        result["offload_bytes"] = folder_bytes(subdir)
    elif a.mode == "cpu-offload":
        from accelerate import cpu_offload
        cpu_offload(model, execution_device=torch.device("cuda:0"))
        result["weight_placement"] = "CPU weights staged layerwise to CUDA"
    elif a.mode != "qlora":
        model.to(a.device)
    if a.mode in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if a.mode == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            result["precision"] = "frozen NF4 base / FP32 compute and adapters"
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
            bias="none", target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    ids = ids.to(a.device)
    mask = torch.ones_like(ids)
    result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sync()
    result["load_setup_seconds"] = time.perf_counter() - start
    training = a.mode in ("full", "lora", "qlora")
    if training:
        model.train()
        model.config.use_cache = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=1e-4, foreach=False)
        losses = []
        before = next(p for p in model.parameters() if p.requires_grad).detach().float().sum().item()
        sync()
        start = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.activation_offload else contextlib.nullcontext()
            with ctx:
                loss = model(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
            optimizer.step()
            losses.append(loss.item())
        sync()
        elapsed = time.perf_counter() - start
        after = next(p for p in model.parameters() if p.requires_grad).detach().float().sum().item()
        result.update(losses=losses, optimizer_steps=2, workload_seconds=elapsed,
                      seconds_per_step=elapsed / 2, activation_offload=a.activation_offload,
                      first_trainable_tensor_sum_change=after - before)
    else:
        model.eval()
        gen = dict(do_sample=False, use_cache=True,
                   pad_token_id=tokenizer.eos_token_id or 0)
        with torch.inference_mode():
            model.generate(input_ids=ids, attention_mask=mask, min_new_tokens=1, max_new_tokens=1, **gen)
            sync()
            start = time.perf_counter()
            out = model.generate(input_ids=ids, attention_mask=mask, min_new_tokens=16, max_new_tokens=16, **gen)
            sync()
            elapsed = time.perf_counter() - start
        if out.shape[1] - ids.shape[1] != 16:
            raise RuntimeError("Generation did not finish exactly 16 new tokens")
        result.update(generated_tokens=16, workload_seconds=elapsed,
                      tokens_per_second=16 / elapsed,
                      generated_text=tokenizer.decode(out[0, 128:]))
    if a.device == "cuda":
        result.update(accelerator=torch.cuda.get_device_name(),
                      cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    elif a.device == "mps":
        result["mps_current_allocated_bytes_not_peak"] = torch.mps.current_allocated_memory()
    result["status"] = "ok"
    return result


def main():
    a = arguments()
    if a.dry_run:
        print(json.dumps({"plan": vars(a), "workload": {"input_tokens": 128,
              "new_tokens": 16, "optimizer_steps": 2}, "network": a.mode == "download",
              "note": "No model, dependency, or backend validation was performed."}, indent=2))
        return
    if a.worker_result:
        try:
            result = worker(a)
        except Exception as exc:
            traceback.print_exc()
            message = str(exc)
            status = "unsupported" if isinstance(exc, (ImportError, NotImplementedError)) else "error"
            if "out of memory" in message.lower():
                status = "oom"
            result = {"status": status, "error": type(exc).__name__ + ": " + message}
        Path(a.worker_result).write_text(json.dumps(result), encoding="utf-8")
        return

    import psutil
    available = psutil.virtual_memory().available
    ram_limit = min(a.ram_gib * GIB if a.ram_gib else available * .7,
                    available * .7, max(0, available - GIB))
    if ram_limit < .25 * GIB:
        raise SystemExit("Insufficient available RAM with 1 GiB reserve; report budget stop")
    outpath = Path(a.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    logdir = outpath.parent / (outpath.stem + "-logs")
    logdir.mkdir(exist_ok=True)
    run_id = str(time.time_ns())
    result_path = logdir / (run_id + ".json")
    stderr_path = logdir / (run_id + ".log")
    deadline = min(600 if a.mode == "download" else 180, a.timeout)
    base = {"model": a.model, "requested_revision": a.revision, "mode": a.mode,
            "requested_device": a.device, "versions": versions(),
            "host_total_bytes": psutil.virtual_memory().total,
            "initial_available_ram_bytes": available, "rss_limit_bytes": ram_limit,
            "timeout_seconds": deadline, "stderr_log": str(stderr_path.resolve())}
    start = time.monotonic()
    peak = 0
    stopped = None
    with stderr_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                *sys.argv[1:], "--worker-result", str(result_path.resolve())], stdout=log, stderr=log)
        psproc = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                processes = [psproc] + psproc.children(recursive=True)
                rss = sum(p.memory_info().rss for p in processes if p.is_running())
                peak = max(peak, rss)
                if time.monotonic() - start > deadline:
                    stopped = "timeout"
                elif rss > ram_limit or psutil.virtual_memory().available < GIB:
                    stopped = "ram_budget"
                elif folder_bytes(a.cache) + folder_bytes(a.offload) > a.disk_gib * GIB:
                    stopped = "disk_budget"
                if stopped:
                    for p in reversed(processes):
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                    proc.wait()
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(.1)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"status": "worker_exit", "exit_code": proc.returncode}
    if stopped:
        result.update(status=stopped, error="Supervisor stopped worker at declared resource limit")
    base.update(result, sampled_peak_rss_bytes=peak,
                rss_sampling_interval_seconds=.1, total_worker_seconds=time.monotonic() - start,
                cache_bytes=folder_bytes(a.cache), total_offload_bytes=folder_bytes(a.offload))
    line = json.dumps(base)
    with outpath.open("a", encoding="utf-8") as output:
        output.write(line + "\n")
    print(json.dumps(base, indent=2))


if __name__ == "__main__":
    main()
