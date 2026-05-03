# UFI-Former: Unified Feature Interaction Transformer for Large-Scale CVR Prediction

## Overview

UFI-Former is a graduate-level machine learning project focused on **conversion rate prediction (CVR)** for large-scale recommendation systems using the **TAAC2026 academic benchmark dataset**.

The objective of this project is to build a unified recommender architecture capable of jointly learning:

- static user features,
- static item features,
- dense aligned numerical signals,
- multi-value sparse categorical fields,
- and multi-domain sequential user behavior

inside a single end-to-end deep learning framework.

Rather than modeling feature interaction and sequence behavior separately, UFI-Former converts all heterogeneous feature groups into a unified token sequence processed by a shared Transformer-style backbone.

---

## Problem Statement

Given a user-item interaction record with associated historical behavior signals, predict:

> **Will this user produce a positive conversion event for this candidate item?**

This is formulated as a **binary classification / recommendation ranking problem**, where:

- `label_type = 1` → positive conversion
- `label_type = 0` → negative or non-conversion

Primary evaluation metric:

- **ROC-AUC**

Secondary practical consideration:

- **Inference latency under large sparse feature spaces**

---

## Current Project Status

This repository currently contains:

### ✅ Phase 1 — Baseline Pipeline Validation

A complete PyTorch baseline using a **MultiFieldMLP** recommender:

- Hugging Face TAAC2026 sample dataset loading
- categorical feature preprocessing
- embedding-based sparse feature encoding
- binary CVR prediction with `BCEWithLogitsLoss`
- train/validation loop
- ROC-AUC evaluation
- baseline checkpointing

This baseline serves as the infrastructure sanity check before scaling to the final Transformer architecture.

---

### 🚧 Phase 2 — UFI-Former Implementation (In Progress)

Upcoming implementation includes:

- unified token builder for all feature domains
- field embeddings
- feature-type embeddings
- domain embeddings
- position embeddings for sequential behavior
- shared Transformer encoder blocks
- pooled MLP prediction head
- large-scale academic track submission experiments

---

## Proposed Model Architecture

```text
[CLS]
 + static user sparse tokens
 + static item sparse tokens
 + multi-value sparse tokens
 + dense aligned projected tokens
 + domain A/B/C/D behavior sequence tokens
        ↓
Unified token embedding space
(value + field + type + domain + position)
        ↓
Shared Transformer Encoder Blocks
        ↓
CLS / attention pooling / mean pooling
        ↓
Compact MLP Head
        ↓
Raw Logit
        ↓
Sigmoid Probability
