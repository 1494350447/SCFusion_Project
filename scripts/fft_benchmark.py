import time
import torch
from models.basic_layers import fft_amp_phase


def bench_fft(batch=4, ch=8, h=256, w=256, iters=10):
    x = torch.randn(batch, ch, h, w)
    # warmup
    for _ in range(3):
        _ = fft_amp_phase(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(iters):
        _ = fft_amp_phase(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    avg_ms = (t1 - t0) / iters * 1000
    print(f"FFT forward avg: {avg_ms:.2f} ms (batch={batch}, ch={ch}, {h}x{w})")


if __name__ == '__main__':
    bench_fft()
