# AEGL Paper-Ready Summary

This repository contains a compact PyTorch implementation of the AEGL proposal: a variational routing framework over a memory repository, a graph transformer reasoning layer, ELBO-style regularization, and causal verification hooks.

## What is implemented
- Variational routing using Gumbel-Softmax relaxation
- Graph Transformer over retrieved subgraphs
- ELBO-based uncertainty calibration
- Causal-counterfactual verification module
- Lightweight data loader and evaluation harness

## Validation
The latest regression test was executed with the workspace virtual environment and passed.
