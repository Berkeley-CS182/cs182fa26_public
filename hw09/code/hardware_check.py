#!/usr/bin/env python3
"""Bounded Qwen3.5 hardware probe. Submit hardware-results.csv; see README-hardware.md.

Only --mode download or --suite --download uses the network. Every configuration
runs in a fresh child process. Detailed JSON/logs stay local and need not be submitted.
"""
import argparse
import contextlib
import copy
import csv
import fnmatch
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

GIB = 2 ** 30
MODELS = {s.lower(): "Qwen/Qwen3.5-" + s for s in ("0.8B", "2B", "4B", "9B")}
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.tiktoken", "tokenizer.*", "merges.txt", "vocab.*"]
MODES = ["setup", "download", "fp32", "int8", "disk", "cpu-offload", "full", "lora", "qlora"]
PRELOAD = ["Qwen3_5GatedDeltaNet"]
LIMIT_STATUSES = {"oom", "timeout", "budget_stop", "unsupported"}
CSV_FIELDS = ["os", "cpu", "host_ram_gib", "accelerator", "accelerator_memory_gib", "model", "revision",
              "mode", "device", "precision", "status", "worker_seconds", "tokens_per_second", "seconds_per_step",
              "peak_host_rss_gib", "device_allocated_gib", "device_memory_measurement",
              "ram_limit_gib", "disk_budget_gib", "timeout_seconds", "error"]


class BudgetStop(RuntimeError):
    pass


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--suite", action="store_true")
    p.add_argument("--models", nargs="+", default=["0.8b", "2b", "4b"])
    p.add_argument("--model", default="0.8b", help="0.8b, 2b, 4b, 9b, or official Qwen3.5 Hub ID")
    p.add_argument("--revision", default="main", help="fixed commit SHA for a single model")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--download", action="store_true", help="permit suite downloads as each size is needed")
    p.add_argument("--cache", default="hardware-cache")
    p.add_argument("--offload", default="hardware-offload")
    p.add_argument("--output", default="hardware-results.csv")
    p.add_argument("--timeout", type=float, default=180, help="seconds per benchmark, including loading; capped at 180")
    p.add_argument("--total-minutes", type=float, default=30, help="suite benchmark budget; capped at 30; downloads excluded")
    p.add_argument("--download-timeout", type=float, default=1800)
    p.add_argument("--ram-gib", type=float, help="worker RSS cap; default at most 70%% of available RAM")
    p.add_argument("--disk-gib", type=float, default=32, help="combined assignment cache and scratch budget")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--activation-offload", action="store_true")
    p.add_argument("--with-qlora", action="store_true", help="optional CUDA NF4 adapter training in suite")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true", help="offline tiny random Qwen3.5 integration checks")
    p.add_argument("--worker-config", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker_config:
        return a
    if sum(bool(x) for x in (a.mode, a.suite, a.self_test)) != 1:
        p.error("choose exactly one of --mode, --suite, or --self-test")
    a.model = MODELS.get(a.model.lower(), a.model)
    a.models = [MODELS.get(x.lower(), x) for x in a.models]
    if min(a.timeout, a.total_minutes, a.download_timeout, a.disk_gib, a.threads) <= 0 or (a.ram_gib is not None and a.ram_gib <= 0):
        p.error("time, storage, threads, and RAM limits must be positive")
    if a.suite and len(a.models) > 1 and a.revision != "main":
        p.error("a commit SHA belongs to one model; use --model and --mode for fixed-revision individual runs")
    if a.mode in ("int8", "disk") and a.device not in ("cpu", "auto"):
        p.error("int8 and disk use --device cpu")
    if a.mode in ("cpu-offload", "qlora") and a.device not in ("cuda", "auto"):
        p.error("cpu-offload and qlora require CUDA")
    if a.activation_offload and (a.device not in ("cuda", "auto") or a.mode not in ("full", "lora", "qlora")):
        p.error("activation offload requires a CUDA training mode")
    a.tiny = False
    return a


def folder_bytes(path):
    total = 0
    for item in Path(path).rglob("*") if Path(path).exists() else []:
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            pass
    return total


def versions():
    result = {"python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor()}
    for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "psutil"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def resolve_device(requested):
    import torch
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def hardware():
    import psutil
    import torch
    result = {"os": platform.platform(), "cpu": platform.processor(),
              "host_ram_gib": psutil.virtual_memory().total / GIB,
              "accelerator": "none", "accelerator_memory_gib": ""}
    if torch.cuda.is_available():
        result.update(accelerator=torch.cuda.get_device_name(0),
                      accelerator_memory_gib=torch.cuda.get_device_properties(0).total_memory / GIB)
    elif torch.backends.mps.is_available():
        result["accelerator"] = "Apple MPS (shared host memory)"
        if hasattr(torch.mps, "recommended_max_memory"):
            result["accelerator_memory_gib"] = torch.mps.recommended_max_memory() / GIB
    return result


def result_exit_code(statuses, self_test=False):
    """0: all completed; 2: measured limit; 1: setup/runner failure."""
    if any(s != "ok" and (self_test or s not in LIMIT_STATUSES) for s in statuses):
        return 1
    return 2 if any(s != "ok" for s in statuses) else 0


def wait_for_memory_recovery(initial_available):
    import psutil
    start = time.monotonic()
    # Windows may release a terminated worker's CUDA/host allocations later.
    time.sleep(5)
    available = psutil.virtual_memory().available
    while available < .9 * initial_available and time.monotonic() - start < 30:
        time.sleep(min(1, max(0, 30 - (time.monotonic() - start))))
        available = psutil.virtual_memory().available
    return {"recovery_seconds": time.monotonic() - start,
            "recovery_complete": available >= .9 * initial_available,
            "available_ram_after_gib": available / GIB}


def worker(a):
    from huggingface_hub import HfApi, snapshot_download
    if a.mode == "download":
        api = HfApi()
        # Cache the requested branch -> SHA mapping for later offline reruns.
        resolved = api.resolve_revision(a.model, revision=a.revision, cache_dir=a.cache)
        info = api.model_info(a.model, revision=resolved.resolved, files_metadata=True)
        wanted = [x for x in info.siblings if any(fnmatch.fnmatch(x.rfilename, pat) for pat in PATTERNS)]
        if any(x.size is None for x in wanted):
            raise BudgetStop("Unknown model-file size; cannot check storage budget")
        # Count only missing files for this resolved revision, including partial downloads conservatively.
        try:
            cached = Path(snapshot_download(a.model, revision=info.sha, cache_dir=a.cache,
                                           local_files_only=True, allow_patterns=PATTERNS))
        except Exception:
            cached = None
        needed = sum(x.size for x in wanted if cached is None or not (cached / x.rfilename).is_file())
        if needed + folder_bytes(a.cache) + folder_bytes(a.offload) > a.disk_gib * GIB:
            raise BudgetStop("Download estimate exceeds --disk-gib; reduce the ladder or use a larger disk budget")
        if shutil.disk_usage(Path(a.cache).resolve()).free < needed + GIB:
            raise BudgetStop("Insufficient free disk space with 1 GiB reserve")
        start = time.perf_counter()
        path = snapshot_download(a.model, revision=info.sha, cache_dir=a.cache, allow_patterns=PATTERNS, max_workers=2)
        return {"status": "ok", "snapshot": path, "revision": info.sha,
                "download_seconds": time.perf_counter() - start, "estimated_download_bytes": needed}

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
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

    if a.mode == "setup":
        import accelerate
        import peft
        x = torch.ones((32, 32), device=a.device)
        if not torch.all(x @ x == 32).item():
            raise RuntimeError("Matrix multiplication check failed")
        sync()
        return {"status": "ok", "device": a.device, "precision": "FP32", "matmul_check": True, **hardware()}

    start = time.perf_counter()
    if a.tiny:
        from transformers import Qwen3_5TextConfig
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
            linear_value_head_dim=8, layer_types=["linear_attention"] * 3 + ["full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0,
                             "mrope_section": [1, 1, 2]}, eos_token_id=127, pad_token_id=0)
        config._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(config).float()
        ids = torch.arange(128).remainder(120).unsqueeze(0)
        revision = "tiny-random-offline-not-pretrained"
        tokenizer = None
    else:
        snapshot = snapshot_download(a.model, revision=a.revision, cache_dir=a.cache,
                                     local_files_only=True, allow_patterns=PATTERNS)
        revision = Path(snapshot).name
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        kw = dict(local_files_only=True, trust_remote_code=False, use_safetensors=True,
                  dtype=torch.float32, attn_implementation="eager")
        if a.mode == "qlora":
            from transformers import BitsAndBytesConfig
            kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32), device_map={"": 0})
        # Explicit text backbone of a multimodal checkpoint; no vision training.
        model = Qwen3_5ForCausalLM.from_pretrained(snapshot, **kw)
        text = "Attention combines information from earlier tokens. " * 128
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :128]
    if ids.shape[1] != 128:
        raise RuntimeError("Expected exactly 128 input tokens")
    result = {"revision": revision, "parameter_count": model.num_parameters(), "input_tokens": 128,
              "batch_size": 1, "seed": 0, "device": a.device, "precision": "FP32",
              "quantized_linear_modules": 0}
    if a.mode == "int8":
        engines = torch.backends.quantized.supported_engines
        engine = next((x for x in ("x86", "fbgemm", "qnnpack", "onednn") if x in engines), None)
        if engine is None:
            raise NotImplementedError("No CPU dynamic INT8 engine: " + str(engines))
        torch.backends.quantized.engine = engine
        model.eval()
        torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        qmods = [m for m in model.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
        if not qmods:
            raise NotImplementedError("No linear weights were packed into INT8")
        result.update(precision="INT8 linear / FP32 other", quantized_linear_modules=len(qmods), quantized_engine=engine,
                      packed_linear_weight_bytes=sum(m.weight().numel() * m.weight().element_size() for m in qmods))
    elif a.mode == "disk":
        from accelerate import disk_offload
        # Accelerate writes state-dict entries, which can include tied aliases.
        required = sum(p.numel() * p.element_size() for p in model.state_dict().values())
        if required + folder_bytes(a.cache) + folder_bytes(a.offload) > a.disk_gib * GIB:
            raise BudgetStop("Disk offload files would exceed --disk-gib")
        if shutil.disk_usage(Path(a.offload).resolve()).free < required + GIB:
            raise BudgetStop("Insufficient disk space for FP32 offload files and 1 GiB reserve")
        disk_offload(model, offload_dir=a.scratch, execution_device=torch.device("cpu"), preload_module_classes=PRELOAD)
        result["offload_bytes"] = folder_bytes(a.scratch)
    elif a.mode == "cpu-offload":
        from accelerate import cpu_offload
        cpu_offload(model, execution_device=torch.device("cuda:0"), preload_module_classes=PRELOAD)
    elif a.mode != "qlora":
        model.to(a.device)
    if a.mode in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if a.mode == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            result["precision"] = "NF4 frozen base / FP32 adapters+compute"
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
            bias="none", target_modules="all-linear", task_type="CAUSAL_LM"))
    ids = ids.to(a.device)
    mask = torch.ones_like(ids)
    result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sync()
    result["load_setup_seconds"] = time.perf_counter() - start
    if a.mode in ("full", "lora", "qlora"):
        model.train()
        model.config.use_cache = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, foreach=False)
        losses = []
        sync()
        start = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.activation_offload else contextlib.nullcontext()
            with ctx:
                # The causal loss shifts labels internally; never shift them a second time.
                loss = model(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
            optimizer.step()
            losses.append(loss.item())
        sync()
        elapsed = time.perf_counter() - start
        result.update(losses=losses, optimizer_steps=2, workload_seconds=elapsed,
                      seconds_per_step=elapsed / 2, activation_offload=a.activation_offload)
    else:
        model.eval()
        gen = dict(do_sample=False, use_cache=True,
                   pad_token_id=tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0)
        with torch.inference_mode():
            model.generate(input_ids=ids, attention_mask=mask, min_new_tokens=1, max_new_tokens=1, **gen)
            sync()
            start = time.perf_counter()
            out = model.generate(input_ids=ids, attention_mask=mask, min_new_tokens=16, max_new_tokens=16, **gen)
            sync()
            elapsed = time.perf_counter() - start
        if out.shape[1] - ids.shape[1] != 16:
            raise RuntimeError("Generation did not finish exactly 16 new tokens")
        result.update(generated_tokens=16, workload_seconds=elapsed, tokens_per_second=16 / elapsed,
                      generated_token_ids=out[0, 128:].tolist())
    if a.device == "cuda":
        result.update(device_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                      device_memory_measurement="CUDA allocator peak", cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    elif a.device == "mps":
        result.update(device_allocated_gib=torch.mps.current_allocated_memory() / GIB,
                      device_memory_measurement="MPS end sample only; unified RAM",
                      mps_current_allocated_bytes_not_peak=torch.mps.current_allocated_memory())
    result["status"] = "ok"
    return result


def write_result(a, result):
    outpath = Path(a.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.with_suffix(".jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(result) + "\n")
    exists = outpath.exists() and outpath.stat().st_size > 0
    with outpath.open("a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        row = {key: result.get(key, "") for key in CSV_FIELDS}
        row["error"] = str(row["error"]).replace("\n", " ")[:240]
        writer.writerow(row)
    print("{mode:24} {model:23} {device:5} {status}".format(**result), flush=True)


def supervise(a, machine, remaining=None):
    import psutil
    a = copy.copy(a)
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    Path(a.offload).mkdir(parents=True, exist_ok=True)
    outpath = Path(a.output)
    logdir = outpath.parent / (outpath.stem + "-logs")
    logdir.mkdir(parents=True, exist_ok=True)
    run_id = str(time.time_ns())
    a.scratch = str((Path(a.offload) / ("run-" + run_id)).resolve())
    result_path = logdir / (run_id + ".json")
    config_path = logdir / (run_id + "-config.json")
    stderr_path = logdir / (run_id + ".log")
    a.worker_result = str(result_path.resolve())
    config_path.write_text(json.dumps(vars(a)), encoding="utf-8")
    available = psutil.virtual_memory().available
    ram_limit = a.ram_limit_bytes
    limit = min(a.download_timeout, 1800) if a.mode == "download" else min(180, a.timeout)
    if remaining is not None:
        limit = min(limit, max(0, remaining))
    base = {**machine, "model": a.model, "requested_revision": a.revision, "revision": a.revision,
            "mode": a.mode + ("+activation-offload" if a.activation_offload else ""), "device": a.device,
            "precision": "INT8 linear / FP32 other" if a.mode == "int8" else "NF4 frozen base / FP32 adapters+compute" if a.mode == "qlora" else "FP32",
            "versions": versions(), "rss_limit_bytes": ram_limit, "ram_limit_gib": ram_limit / GIB,
            "initial_available_ram_gib": a.initial_available_ram / GIB,
            "available_ram_before_gib": available / GIB,
            "disk_budget_gib": a.disk_gib, "timeout_seconds": limit,
            "stderr_log": str(stderr_path.resolve()), "status": "budget_stop"}
    if ram_limit < .25 * GIB or available < GIB or limit <= 0:
        base["error"] = "Insufficient RAM reserve or remaining suite time"
        write_result(a, base)
        return base
    start = time.monotonic()
    peak = 0
    stopped = None
    disk_check = 0
    with stderr_path.open("w", encoding="utf-8") as log:
        env = os.environ.copy()
        if a.mode != "download":
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-config", str(config_path.resolve())],
                                stdout=log, stderr=log, env=env)
        psproc = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                processes = [psproc] + psproc.children(recursive=True)
                rss = sum(p.memory_info().rss for p in processes if p.is_running())
                peak = max(peak, rss)
                if time.monotonic() - start > limit:
                    stopped = ("timeout", "Worker exceeded time limit including import/load/setup")
                elif rss > ram_limit or psutil.virtual_memory().available < GIB:
                    stopped = ("budget_stop", "Worker RSS cap or 1 GiB available-RAM reserve reached")
                elif time.monotonic() - disk_check > 1:
                    disk_check = time.monotonic()
                    if folder_bytes(a.cache) + folder_bytes(a.offload) > a.disk_gib * GIB:
                        stopped = ("budget_stop", "Assignment disk budget reached")
                    elif min(shutil.disk_usage(a.cache).free, shutil.disk_usage(a.offload).free) < GIB:
                        stopped = ("budget_stop", "1 GiB free-disk reserve reached")
                if stopped:
                    for child in reversed(processes):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    proc.wait()
                    psutil.wait_procs(processes, timeout=5)
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(.1)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or "status" not in result:
            raise ValueError("Missing worker status")
    except (OSError, ValueError):
        result = {"status": "worker_exit", "error": "Missing or invalid worker result; exit code " + str(proc.returncode)}
    if proc.returncode and not stopped:
        result = {"status": "worker_exit", "error": "Worker exit code " + str(proc.returncode)}
    if stopped:
        result.update(status=stopped[0], error=stopped[1])
    base.update(result, peak_host_rss_gib=peak / GIB, rss_sampling_interval_seconds=.1,
                worker_seconds=time.monotonic() - start)
    # Only remove scratch created by this run; never remove the user's model cache.
    scratch = Path(a.scratch).resolve()
    if scratch.exists() and scratch.parent == Path(a.offload).resolve() and scratch.name == "run-" + run_id:
        shutil.rmtree(scratch)
    base.update(wait_for_memory_recovery(a.initial_available_ram))
    write_result(a, base)
    return base


def suite_plan(device, optional_qlora=False):
    plan = [("fp32", device, False), ("int8", "cpu", False), ("disk", "cpu", False),
            ("full", device, False), ("lora", device, False)]
    if device != "cpu":
        plan.insert(1, ("fp32", "cpu", False))
    if device == "cuda":
        plan.extend([("cpu-offload", "cuda", False), ("lora", "cuda", True)])
        if optional_qlora:
            plan.append(("qlora", "cuda", False))
    return plan


def main():
    a = arguments()
    if a.worker_config:
        a = argparse.Namespace(**json.loads(Path(a.worker_config).read_text(encoding="utf-8")))
        try:
            result = worker(a)
        except Exception as exc:
            traceback.print_exc()
            message = str(exc)
            status = "budget_stop" if isinstance(exc, BudgetStop) else "missing_dependency" if isinstance(exc, ImportError) else "unsupported" if isinstance(exc, NotImplementedError) else "error"
            if status == "missing_dependency":
                message += "; install the README dependencies with this environment's Python"
            if "out of memory" in message.lower():
                status = "oom"
            result = {"status": status, "error": type(exc).__name__ + ": " + message}
        Path(a.worker_result).write_text(json.dumps(result), encoding="utf-8")
        return
    if a.dry_run:
        print(json.dumps({"arguments": vars(a), "workload": {"input_tokens": 128, "new_tokens": 16, "optimizer_steps": 2},
              "network_allowed": a.mode == "download" or (a.suite and a.download),
              "note": "Plan only; no dependency, backend, or model validation performed."}, indent=2))
        return
    try:
        import psutil
        a.device = resolve_device(a.device)
        machine = hardware()
        a.initial_available_ram = psutil.virtual_memory().available
        # One comparable RSS cap for this invocation; retain the live 1 GiB reserve.
        a.ram_limit_bytes = min(a.ram_gib * GIB if a.ram_gib else a.initial_available_ram * .7,
                               a.initial_available_ram * .7, max(0, a.initial_available_ram - GIB))
    except ImportError as exc:
        write_result(a, {"os": platform.platform(), "cpu": platform.processor(), "model": a.model,
                        "revision": a.revision, "mode": a.mode or "suite", "device": a.device,
                        "status": "missing_dependency", "error": str(exc) + "; complete README setup with this Python"})
        raise SystemExit("Complete README setup using this environment's Python, then rerun.")
    if a.mode in ("int8", "disk"):
        a.device = "cpu"
    if a.mode in ("cpu-offload", "qlora") and a.device != "cuda":
        raise SystemExit("This mode requires CUDA; use the CPU suite on this machine.")
    if a.activation_offload and a.device != "cuda":
        raise SystemExit("Activation offload requires CUDA")
    if a.self_test:
        a.tiny = True
        a.models = ["tiny-random-Qwen3.5"]
        a.device = "cpu"
        a.download = False
        if a.output == "hardware-results.csv":
            a.output = "hardware-self-test.csv"
    if a.suite or a.self_test:
        plan = suite_plan(a.device, a.with_qlora)
        active = set(plan)
        statuses = []
        remaining = min(30, a.total_minutes) * 60
        for model in a.models:
            if not active or remaining <= 0:
                break
            a.model = model
            if a.download:
                download = copy.copy(a)
                download.mode, download.activation_offload = "download", False
                loaded = supervise(download, machine)
                statuses.append(loaded["status"])
                if loaded["status"] != "ok":
                    break
                a.revision = loaded["revision"]
            for mode, device, activation in plan:
                if (mode, device, activation) not in active:
                    continue
                run = copy.copy(a)
                run.mode, run.device, run.activation_offload = mode, device, activation
                result = supervise(run, machine, remaining)
                statuses.append(result["status"])
                remaining -= result.get("worker_seconds", 0)
                if result["status"] != "ok":
                    active.remove((mode, device, activation))
            a.revision = "main"
        code = result_exit_code(statuses, a.self_test)
        print("Finished. Results: " + str(Path(a.output).resolve()) +
              (". Fix error rows and rerun before submitting." if code == 1 else ". Submit only this CSV."))
        raise SystemExit(code)
    else:
        result = supervise(a, machine)
        raise SystemExit(result_exit_code([result["status"]]))


if __name__ == "__main__":
    main()
