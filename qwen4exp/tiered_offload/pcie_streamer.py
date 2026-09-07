"""
Asynchronous PCIe 3.0 x16 Layer Streamer.
Overlaps expert weights DMA transfer with GPU execution.
"""

import torch

class AsyncPCIeStreamer:
    def __init__(self, device_id: int):
        self.device = torch.device(f"cuda:{device_id}")
        self.stream = torch.cuda.Stream(device=self.device)

    def async_copy_layer(self, host_tensor: torch.Tensor, gpu_dest: torch.Tensor):
        with torch.cuda.stream(self.stream):
            gpu_dest.copy_(host_tensor, non_blocking=True)
        return self.stream
