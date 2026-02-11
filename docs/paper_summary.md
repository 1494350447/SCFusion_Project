# SCFusion — Paper Summary and Implementation Notes

This document summarizes the architecture and key implementation details of the paper "Spatial-Channel Cross-Frequency Guided Fusion Network for Infrared-Visible Image Object Detection" and lists high-priority differences between the paper and the current repository implementation.

## High-level architecture
- Dual-branch encoder for infrared (IR) and visible (VIS) inputs.
- Each encoder block combines local convolutional processing with a frequency-selective global branch (FREFormer-like).
- Cross-frequency guided interaction modules (CFGIM) fuse amplitude and phase information across modalities.
- Decoder with upsampling and frequency reconstruction guidance (FRGM) recovers high-frequency details.
- Optional deep supervision on intermediate decoder outputs.

## Key components from paper
- Frequency decomposition: compute amplitude and phase via 2D FFT per-channel.
- Amplitude fusion: per-channel amplitude normalization + learned frequency-selective weighting.
- Phase fusion: represent phase as sin/cos components, learn mixing in that domain to avoid phase wrap issues, then reconstruct phase and inverse-FFT to spatial domain.
- FrequencySelectiveKernel: small MLP-like module producing channel-wise (B,C,1,1) gates from spectral descriptors.
- Learnable high-pass: depthwise residual high-pass filter to emphasize edges.

## Important hyperparameters and design choices (paper)
- FFT normalization: orthonormal FFT used (paper uses norm='ortho').
- Amplitude normalization: per-channel mean/std normalization before mix.
- Phase representation: sin/cos concatenation, then 1x1 conv mixing.
- Residual connections: fusion outputs added/residualized with original features.
- Normalization: paper uses LayerNorm or BN at specific locations; exact positions affect stability.

## High-priority mismatches (code vs paper)
1. Input channels: paper often uses single-channel IR and 3-channel VIS. Code now supports configurable channels but default used to be 3/3; updated to explicit `in_ch_ir`/`in_ch_vis`.
2. Normalization layers: paper indicates LayerNorm on transformer-like branches; implementation uses BatchNorm in several blocks. Consider switching FREFormer blocks to GroupNorm/LayerNorm for stability across batch sizes.
3. Spectral pooling: paper aggregates frequency descriptors; implementation added `spectral_pool` but some modules still use spatial adaptive pooling. We'll standardize to spectral pooling for FrequencySelectiveKernel.
4. Phase mixing dimension: ensure convs in `HSFB` output correct channel counts for sin/cos split and stable atan2 reconstruction (current code implements sin/cos pipeline but requires careful clipping/normalization).
5. Deep supervision outputs and loss weighting: repository supports deep supervision but training script must match paper's loss weights (to be configured in training configs).

## Next actions (priority order)
1. Standardize normalization (switch to LayerNorm/GroupNorm where paper recommends).
2. Tighten phase fusion numeric stability (ensure sin/cos normalization and `atan2` inputs are properly scaled/clamped).
3. Add unit tests for FFT helpers, FrequencySelectiveKernel, HSFB forward shape and numerical sanity.
4. Add micro-benchmarks: forward latency, param counts, and FFT cost profiling.
5. Implement any remaining architectural changes from paper after reviewer confirmation.

---
Notes: this summary is based on the attached paper and the repository code. The next step will implement items 1–3 and add tests/benchmarks.
