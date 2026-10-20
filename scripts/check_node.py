#!/usr/bin/env python
"""Check that the node you were given actually is the GPU you think it is.

Motivation: a VEP-OPSD run decoded at 57-69 ms per token step, i.e. slower per
step than a workstation A6000, on a node whose pod names suggested a shared GPU
pool. Before tuning code, confirm the hardware:

    python scripts/check_node.py
    torchrun --nproc_per_node=8 scripts/check_node.py   # per-rank view

Reports device name, MIG/vGPU hints, memory, a bandwidth probe and a small
matmul throughput probe per visible device.
"""

from __future__ import annotations

import os
import socket
import time

import torch


def probe(device_index: int) -> None:
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    print(f"\n--- cuda:{device_index}")
    print(f"  name              {properties.name}")
    print(f"  total memory      {properties.total_memory / 2**30:.1f} GiB")
    print(f"  multiprocessors   {properties.multi_processor_count}")
    print(f"  capability        {properties.major}.{properties.minor}")
    print(f"  torch build       {torch.__version__} cuda {torch.version.cuda}")
    # A vGPU/MIG slice shows up as a smaller SM count / memory than the physical card.
    if properties.total_memory < 70 * 2**30:
        print("  NOTE: < 70 GiB on an A100-80G package suggests a slice, not a whole card")

    # Bandwidth: one big copy, timed after warm-up.
    bytes_moved = 8 * 2**30
    source = torch.empty(bytes_moved // 4, dtype=torch.float32, device=device_index)
    target = torch.empty_like(source)
    for _ in range(2):
        target.copy_(source)
    torch.cuda.synchronize()
    started = time.perf_counter()
    target.copy_(source)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    print(f"  copy bandwidth    {2 * bytes_moved / seconds / 1e9:.0f} GB/s (read+write)")

    # bf16 matmul throughput.
    size = 8192
    left = torch.randn(size, size, device=device_index, dtype=torch.bfloat16)
    right = torch.randn(size, size, device=device_index, dtype=torch.bfloat16)
    for _ in range(3):
        left @ right
    torch.cuda.synchronize()
    started = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        left @ right
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / repeats
    print(f"  bf16 matmul       {2 * size**3 / seconds / 1e12:.0f} TFLOP/s")
    del source, target, left, right
    torch.cuda.empty_cache()


def main() -> None:
    print(f"host {socket.gethostname()}  pid {os.getpid()}  rank {os.environ.get('RANK', '?')}")
    print(f"visible devices {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    for index in range(torch.cuda.device_count()):
        probe(index)
    # Reference points: A100-80G ≈ 1550-2000 GB/s and 200-300 TFLOP/s bf16 dense;
    # A6000 ≈ 700-750 GB/s and 120-150 TFLOP/s.
    print("\nreference: A100-80G ~1550-2000 GB/s, ~200-300 TFLOP/s bf16 | A6000 ~700 GB/s, ~120 TFLOP/s")


if __name__ == "__main__":
    main()
