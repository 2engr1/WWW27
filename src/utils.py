"""Shared file, hashing, device, and timing utilities."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

NS_NAMESPACE = uuid.UUID("154f289a-fef5-4c99-a2e1-09983f5e6109")

def rows(path):
    """Yield JSON objects from a JSONL file."""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: JSON object required")
                yield value

def dump(path, value):
    """Write a UTF-8 JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def emit(handle, value):
    """Write one JSON object to an open JSONL stream."""
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")

def sha256_file(path):
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def stable_doc_id(original_id):
    """Generate a deterministic UUIDv5 for a source document ID."""
    return str(uuid.uuid5(NS_NAMESPACE, str(original_id)))

def fingerprint(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]

def neural_device(requested):
    """Resolve an explicit neural device; CUDA requests never silently fall back."""
    import torch
    try:
        selected = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("--device must be cpu, cuda, or cuda:N") from exc
    if selected.type == "cpu":
        if selected.index is not None:
            raise ValueError("Use --device cpu without an index")
        print("Neural models: CPU (explicitly selected)", flush=True)
        return "cpu"
    if selected.type != "cuda":
        raise ValueError("--device must be cpu, cuda, or cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. No CPU fallback is performed; use --device cpu explicitly.")
    index = selected.index if selected.index is not None else 0
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(f"GPU cuda:{index} unavailable; visible GPU count: {count}")
    torch.cuda.set_device(index)
    torch.empty(1, device=f"cuda:{index}")
    print(
        f"Neural models: cuda:{index} | {torch.cuda.get_device_name(index)} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    return f"cuda:{index}"

def verify_model_device(model, device, label):
    """Fail early if a neural model is not resident on the requested device."""
    parameters_fn = getattr(model, "parameters", None)
    if not callable(parameters_fn):
        return
    parameters = list(parameters_fn())
    if not parameters:
        return
    import torch
    actual = {str(parameter.device) for parameter in parameters}
    expected = str(torch.device(device))
    if actual != {expected}:
        raise RuntimeError(f"{label} device mismatch: expected {expected}, got {sorted(actual)}")
    print(f"{label} device verified: {expected}", flush=True)

def verify_reranker_device(reranker, device):
    wrapper = getattr(reranker, "model", None)
    model = getattr(wrapper, "model", wrapper)
    if model is not None:
        verify_model_device(model, device, "CrossEncoder")

def synchronize(device):
    """Synchronize queued CUDA work before reading elapsed time."""
    if device and str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)

class Timer:
    """Small monotonic timer with optional CUDA synchronization."""
    def __init__(self, device=None):
        self.device = device
        self.started = None
    def start(self):
        synchronize(self.device)
        self.started = time.perf_counter()
        return self
    def stop_ms(self):
        synchronize(self.device)
        if self.started is None:
            raise RuntimeError("Timer was not started")
        return (time.perf_counter() - self.started) * 1000.0
