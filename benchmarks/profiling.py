"""Opt-in profiling. Profiled runs are excluded from performance comparisons."""
from functools import wraps
import json
from pathlib import Path
import subprocess
import threading
from time import perf_counter


class Profiler:

    def __init__(self, llm, job):
        self.llm = llm
        self.mode = job.get("profile_mode", "none")
        self.directory = Path(job.get("profile_dir", "."))
        self.active_steps = job.get("profile_steps", 8)
        self.wait_steps = job.get("profile_wait", 2)
        self.steps = 0
        self.profiler = None
        self.originals = []
        self.stop = threading.Event()
        self.samples = []
        self.error = None

    def wrap(self, obj, name, label):
        if not hasattr(obj, name):
            return
        import torch
        original = getattr(obj, name)
        @wraps(original)
        def forward(*args, **kwargs):
            with torch.profiler.record_function(label), torch.cuda.nvtx.range(label):
                return original(*args, **kwargs)
        self.originals.append((obj, name, original))
        setattr(obj, name, forward)

    def sample_gpu(self, uuid):
        query = "utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu"
        start = perf_counter()
        while not self.stop.is_set():
            try:
                values = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-gpu=" + query,
                    "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT, timeout=5)
                row = dict(time_s=perf_counter() - start)
                for name, value in zip(query.split(","), values.strip().split(",")):
                    try:
                        row[name] = float(value.strip())
                    except ValueError:
                        row[name] = None
                self.samples.append(row)
            except (OSError, subprocess.SubprocessError) as error:
                self.error = str(error)
                break
            self.stop.wait(.25)

    def __enter__(self):
        if self.mode == "none":
            return self
        import torch
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.llm.model_runner
        for obj, name, label in ((self.llm.scheduler, "schedule", "scheduler"),
                (target, "run_model", "target.run_model"),
                (target.model, "forward", "target.forward"),
                (target.model, "compute_logits", "target.logits")):
            self.wrap(obj, name, label)
        decoder = self.llm.speculative
        if decoder is not None:
            batched = getattr(decoder, "batched", None)
            if batched is not None:
                for name in ("propose", "verify", "accept", "commit"):
                    self.wrap(batched, name, "jetspec.batched_" + name)
            self.wrap(decoder, "run_round", "speculative.round")
            self.wrap(decoder, "verify_logits", "target.verify")
            self.wrap(decoder, "_propose", "jetspec.propose_tree")
            drafter = getattr(decoder, "drafter", None)
            if drafter is not None:
                for name in ("propose", "catch_up", "after_verify"):
                    self.wrap(drafter, name, "draft." + name)
            head = getattr(decoder, "head", None)
            if head is not None:
                self.wrap(head, "forward", "jetspec.head")
                self.wrap(head, "store_context", "jetspec.context_store")
            import nanovllm.engine.speculative as speculative
            for name in ("accept_greedy", "accept_random"):
                self.wrap(speculative, name, "sampling." + name)
        if self.mode == "torch":
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=self.wait_steps, warmup=1, active=self.active_steps, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(str(self.directory / "trace.json")),
                record_shapes=True, profile_memory=True)
            self.profiler.__enter__()
        else:
            torch.cuda.cudart().cudaProfilerStart()
        uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        if uuid is not None:
            uuid = str(uuid)
            if not uuid.startswith("GPU-"):
                uuid = "GPU-" + uuid
            self.thread = threading.Thread(target=self.sample_gpu, args=(uuid,), daemon=True)
            self.thread.start()
        else:
            self.thread = None
            self.error = "CUDA did not expose a device UUID; telemetry disabled to avoid sampling another GPU"
        return self

    def step(self):
        self.steps += 1
        if self.profiler is not None:
            self.profiler.step()

    def __exit__(self, exc_type, exc, traceback):
        if self.mode == "none":
            return
        import torch
        try:
            if self.profiler is not None:
                self.profiler.__exit__(exc_type, exc, traceback)
                averages = self.profiler.key_averages()
                (self.directory / "operators.txt").write_text(
                    averages.table(sort_by="self_cuda_time_total", row_limit=100), encoding="utf-8")
                operators = [dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
                    self_device_us=getattr(e, "self_device_time_total", 0)) for e in averages]
                (self.directory / "operators.json").write_text(json.dumps(operators, indent=2), encoding="utf-8")
            else:
                torch.cuda.cudart().cudaProfilerStop()
        finally:
            self.stop.set()
            if self.thread is not None:
                self.thread.join(timeout=6)
            for obj, name, original in reversed(self.originals):
                setattr(obj, name, original)
            status = dict(mode=self.mode, steps=self.steps, telemetry_error=self.error,
                          complete_capture=self.steps >= self.wait_steps + 1 + self.active_steps,
                          trace_written=(self.directory / "trace.json").exists())
            (self.directory / "profile_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            (self.directory / "gpu_samples.json").write_text(json.dumps(self.samples, indent=2), encoding="utf-8")
