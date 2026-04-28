# Spectral Episode Temporal Mixer for Fast Biomedical Time-Series Classification

**Anonymous Authors**

## Abstract

Fast waveform classifiers for biomedical time series must capture both local morphology and broad spectral state without incurring the training cost of heavy attention models. We present **Spectral Episode Temporal Mixer (SETM)**, a compact architecture that computes a low-dimensional FFT-based summary, softly assigns each sample to a small library of learned spectral prototypes, and uses that episode mixture to route a stack of lightweight temporal expert blocks. SETM does not filter or reconstruct the input signal. Instead, it conditions temporal computation on coarse spectral context. This yields a cleaner architectural claim than front-end spectral filtering while preserving a small-compute training profile. The intended empirical claim is deliberately modest: under matched training budgets, prototype-routed temporal mixing can improve class-sensitive metrics relative to the same backbone with uniform, static, or direct non-prototype routing. The paper should be written to support that bounded claim rather than universal superiority.

## 1. Introduction

Physiological time-series classification often depends on an interaction between two types of evidence. The first is local morphology: waveform shapes, seizure transients, and short-lived patterns that are naturally modeled in the time domain. The second is coarse spectral state: rhythms, drift, broadband noise, and oscillatory context that alter which temporal features are likely to matter. Many fast classifiers can capture both signals implicitly, but they do so without an explicit mechanism that connects spectral context to temporal computation.

This paper studies that interaction directly. We ask whether a model can stay fast while using a small spectral controller to decide which temporal receptive fields should be emphasized for each sample. Our answer is **Spectral Episode Temporal Mixer (SETM)**, a model that summarizes the waveform into a small set of frequency-band statistics, maps that summary to a learned mixture over spectral prototypes, and decodes that mixture into a block-wise routing schedule for temporal experts.

The paper should make three points clearly. First, SETM is not an input-filtering method. The raw waveform stays intact; only the computation path changes. Second, SETM is not defined by a borrowed multi-branch CNN backbone. Its core block is a routed temporal mixer built from depthwise experts, pointwise mixing, and channel MLPs. Third, the intended claim is not that spectral prototypes are clinically identified states. They are learned architectural latents whose value is judged by predictive utility, efficiency, and interpretability.

## 2. Method

Given a window `x in R^{C x T}`, SETM computes:

1. A real-FFT summary `s(x)` from log band energies.
2. A soft assignment `a(x)` over `P` learned spectral prototypes.
3. A block-wise route schedule `R_d(x)` and channel gains decoded from that assignment.
4. A sequence of residual temporal mixer blocks applied to the raw waveform.

Each temporal block contains multiple depthwise temporal experts with different dilations. Their outputs are mixed according to `R_d(x)`, then passed through pointwise and channel-mixing layers. The summary path is therefore tiny, while the temporal path remains dominated by efficient 1D convolutions.

The central design choice is prototype mediation. A direct MLP could map spectral summaries to routes, but the prototype library gives the model a reusable notion of spectral episodes. That yields a stronger mechanism story and a more interpretable ablation package.

## 3. Main Experimental Story

The controlled comparison set should be:

- `uniform`: same backbone, no adaptive routing.
- `static`: one learned route schedule shared by all samples.
- `direct`: spectral summary routed directly without prototypes.
- `prototype`: full SETM.

The main datasets should be PTB-XL and CHB-MIT, with Sleep-EDF as the preferred third benchmark if time allows. The paper should emphasize Macro-F1, balanced accuracy, and efficiency metrics. Reviewer-facing external baselines such as MiniRocket and Hydra remain important, but the core claim depends most directly on the controlled in-family ablations above.

## 4. Claim Boundaries

The paper should avoid drastic claims. It should not claim clinical state discovery, universal superiority, or broad domain generalization without explicit evidence. A stronger but still defensible claim is that prototype-routed temporal mixing is a promising low-cost inductive bias for some biomedical waveform tasks and that its benefit appears through computation allocation rather than waveform filtering.

## 5. What Would Make This Submission Credible

A credible main-track submission would need:

- improvements over `uniform` that are directionally consistent across seeds,
- a meaningful gap over `static` and ideally `direct`,
- a measured runtime close to the controlled baseline,
- and mechanism figures showing non-degenerate prototype usage.

If those pieces do not appear, the project is still useful, but the paper should narrow accordingly.
