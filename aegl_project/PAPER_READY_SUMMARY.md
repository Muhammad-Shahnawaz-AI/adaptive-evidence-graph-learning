# AEGL Paper-Ready Summary

This repository contains a compact PyTorch implementation of the AEGL proposal: a variational routing framework over a memory repository, a graph transformer reasoning layer, ELBO-style regularization, and causal verification hooks.

## What is implemented
- Variational routing using Gumbel-Softmax relaxation
- Graph Transformer over retrieved subgraphs
- ELBO-based uncertainty calibration
- Causal-counterfactual verification module
- Lightweight data loader and evaluation harness

## Validation
- `pytest tests/` passes (1 test: verifies the synthetic data loader's tensor/label contract).
- All five `train.py --ablation {baseline,static_topology,no_elbo,no_causal,full}` configs
  run to completion and produce Top-1/5, ECE, and macro-F1 numbers on the synthetic
  in-distribution/OOD splits.
- **Not yet validated:** the real benchmark suite from Table 5.1 (ImageNet-A/R,
  iNaturalist 2024, MIMIC-IV/CMU-MOSEI, PEMS). All numbers so far come from the
  synthetic proxy dataset described in `aegl/data.py`, not the target datasets, so
  they should not be cited as evidence for the proposal's empirical claims.
