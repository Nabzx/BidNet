from __future__ import annotations

import torch


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Selected device backend: CUDA")
            return device

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            device = torch.device("mps")
            print("Selected device backend: MPS")
            return device

        device = torch.device("cpu")
        print("Selected device backend: CPU")
        return device

    device = torch.device(device_name)
    print(f"Selected device backend: {device.type.upper()}")
    return device
