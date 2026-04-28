# Spectral Episode Temporal Mixer

## Title And Core Novelty Claim

**Title:** Spectral Episode Temporal Mixer: Prototype-Routed Fast Temporal Experts for Biomedical Time-Series Classification

**Core novelty claim:** SETM is a lightweight classifier that does not modify the input waveform. Instead, it extracts a coarse spectral summary, softly assigns each sample to a small library of learned spectral episodes, and uses that episode mixture to route a stack of temporal expert blocks. The paper's intended claim is narrow: prototype-mediated spectral context can improve the quality/efficiency tradeoff of a compact temporal mixer relative to uniform, static, or direct routing under matched training budgets.

## Abstract

Biomedical time-series classification often depends on both oscillatory context and time-local morphology, yet many fast waveform models must learn this interaction implicitly. We propose Spectral Episode Temporal Mixer (SETM), a compact classifier that first computes a low-dimensional spectral summary, then uses learned spectral prototypes to generate a block-wise routing schedule over lightweight temporal experts. Unlike front-end filtering methods, SETM does not alter the input signal; it conditions temporal computation on coarse spectral state. This yields a small, interpretable controller layered on top of an otherwise efficient temporal mixer. Our goal is not universal state of the art. The intended contribution is a reproducible demonstration that prototype-routed temporal mixing can improve class-sensitive metrics at near-baseline training cost on biomedical ECG and EEG tasks, with clear ablations against uniform routing, dataset-level static routing, and direct non-prototype routing.

## Introduction And Motivation

Physiological signals mix at least two useful forms of evidence:

- local waveform structure such as seizure transients and ECG morphology,
- and global spectral context such as rhythms, line-noise bands, and broad state changes.

The original idea in this repository used a spectral gate on the input. That was efficient, but too close to front-end filtering and too dependent on an external backbone story. SETM shifts the contribution upward. The model no longer filters the waveform at all. Instead, it asks a more interesting question: can a small learned notion of spectral state decide which temporal receptive fields should be emphasized for each sample?

That produces a cleaner novelty claim:

1. The spectral pathway is **summarization only**, not signal manipulation.
2. The controller is **prototype-based**, not just a direct MLP gate.
3. The backbone is a **routed temporal mixer**, not an inherited multi-branch CNN.

## Method

### Model Architecture

Input `x in R^(B x C x T)` goes through:

1. Real FFT summary: compute `S in R^(B x K)` from log band energies.
2. Prototype assignment: softly assign each sample to `P` learned spectral episodes.
3. Episode decoding: decode that mixture into block-wise expert weights and channel gains.
4. Temporal mixer backbone: pass the raw waveform through a stack of routed temporal blocks.
5. Head: global average pooling and linear classification.

Each temporal block uses:

- several depthwise temporal experts with shared channel width,
- sample-specific mixing weights over those experts,
- a pointwise mixing layer,
- and a channel MLP residual path.

The implementation stays lightweight because the expensive path remains 1D convolution, while routing is produced by a tiny controller.

### Main Equation

Let `s(x)` be the spectral summary, `a(x)` the prototype assignment, and `R_d(x)` the routing weights for block `d`. Then:

```text
a(x) = softmax(-||W s(x) - p_j||^2)_j
R_d(x) = softmax(sum_j a_j(x) E_{j,d})
```

where `p_j` are learned spectral prototypes and `E_{j,d}` are prototype-specific routing codes for block `d`.

The temporal block output is:

```text
z_{d+1} = z_d + Mix_d(sum_m R_{d,m}(x) Expert_{d,m}(Norm(z_d))) + MLP_d(Norm(.))
```

The classifier therefore conditions temporal computation on a compact spectral state rather than on direct spectral filtering.

### Objective

For dataset `D = {(x_i, y_i)}`, classifier `f_theta`, task loss `ell`, prototype assignment `a`, and routes `R`, optimize:

```text
min_theta  1/N sum_i ell(f_theta(x_i), y_i)
         + lambda_balance L_balance(a)
         + lambda_temporal_smooth L_smooth(R).
```

`L_balance` discourages all samples from collapsing to one prototype. `L_smooth` discourages abrupt block-to-block route changes.

## Hypotheses

- H1: Prototype-routed temporal mixing improves Macro-F1 or balanced accuracy on at least one major ECG or EEG benchmark relative to the same backbone with uniform routing.
- H2: The prototype library matters: `prototype` routing is preferable to `direct` routing under matched parameter and runtime budgets.
- H3: Learned prototype usage is non-degenerate and stable enough across seeds to support an interpretable spectral-state story.
- H4: The FFT summary overhead stays small enough that five-seed SETM runs remain near the training cost of a compact convolutional baseline.

## Experimental Design

### Datasets

Primary:

- **PTB-XL:** multilabel ECG superclass classification.
- **CHB-MIT:** subject-aware EEG seizure detection.

Recommended third benchmark:

- **Sleep-EDF Expanded:** sleep-stage classification, useful because spectral state is clinically meaningful and distinct from seizure detection.

### Baselines

Required controlled baselines:

- SETM with `uniform` routing.
- SETM with `static` routing.
- SETM with `direct` routing.
- SETM with full `prototype` routing.

Reviewer-facing external baselines:

- MiniRocket or MultiRocket.
- Hydra.
- At least one compact upstream neural baseline from a maintained implementation.

### Metrics

- Primary: Macro-F1 and balanced accuracy.
- Secondary: AUROC, AUPRC, calibration if time allows.
- Efficiency: parameter count, forward FLOPs, peak VRAM, epoch time, total wall-clock.
- Statistics: five matched seeds, mean +/- std, paired significance for `prototype` vs `uniform`.

### Ablations

- `uniform` vs `static` vs `direct` vs `prototype`.
- Number of prototypes `P in {4, 8, 12}`.
- Number of bands `K in {8, 16, 32}`.
- Route smoothness penalty on/off.
- Low-label PTB-XL subsets `{10%, 25%, 100%}` if time permits.

## Why This Could Be Main-Track Worthy

The strongest version of the paper is not "we built another fast biosignal CNN." It is:

- a distinct architecture rather than a wrapper around an existing one,
- a clean causal ablation story,
- a mechanism that is interpretable at the level of spectral state and temporal expert choice,
- and a compute story that remains realistic on one workstation-class accelerator.

That still does not guarantee NeurIPS acceptance. The idea is plausible, not proven. To be competitive, the evidence package has to show:

- gains that are directionally consistent across seeds,
- runtime close to the controlled baseline,
- and a mechanism figure that demonstrates non-trivial prototype usage.

## Failure Modes And Claim Boundaries

This project should avoid overstating what the method does.

- It does **not** discover causal physiological states.
- It does **not** prove that the learned prototypes correspond to clinical categories.
- It does **not** guarantee universal improvement across all time-series domains.

If the learned assignments collapse, if the gains disappear against `direct` routing, or if runtime drifts far beyond baseline, the paper should narrow its claims or pivot again.
