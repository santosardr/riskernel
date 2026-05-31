# RIS-Kernel: A Model-Agnostic Architecture for Long-Context LLM Inference via Sparse Attention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20476759.svg)](https://doi.org/10.5281/zenodo.20476759)
[![Code Ocean](https://img.shields.io/badge/Code%20Ocean-Capsule-blue)](https://doi.org/10.24433/CO.0351350.v1)

This repository contains the official implementation of **RIS-Kernel**, a systems-level sparse attention inference engine that runs massive context windows (64k+ tokens) on commodity, unaccelerated CPU hardware.

---

## 📖 Abstract

> Full self-attention in large language models scales as $O(N^2)$, limiting long-context document analysis to 65,536 tokens and requiring costly GPU clusters. The Reduced Interaction Sampling (RIS) inference engine addresses this constraint as a model-agnostic architecture. Without modifying weights, RIS reduces self-attention complexity to $O(N \log N)$ using sparse stochastic geometry that fits within commodity memory limits. We validate RIS on Qwen2-1.5B-Instruct across two regimes. In controlled evaluations at 32,768 tokens (where native dense attention serves as the upper bound), RIS-Stochastic at 1% density and 70 ensemble seeds achieves 75.00% accuracy, outperforming the native dense baseline (71.88%), while RIS-Stochastic at 5% density and 10 seeds matches it (71.88%). This demonstrates that sparse attention acts as a regularizer: low density (1%) over multiple seeds filters out sequence-level noise, whereas higher density (5%) reintroduces distractor noise. Under the tightest budget, RIS-Structural reaches 68.75% accuracy at 1% density with just 10 seeds, recovering 75% of the contextual gap relative to the zero-context floor (59.38%). At 65,536 tokens, where dense attention triggers out-of-memory faults, RIS yields retrieval gains of up to 14.06 percentage points over the zero-context floor (51.56%). All evaluations run on commodity, unaccelerated CPU servers (16–128 GB of RAM), demonstrating that long-context LLM inference is feasible on standard academic hardware without GPU acceleration.

---

## 🔬 Scientific Context & PoC

RIS-Kernel acts as a model-agnostic layer that intercepts attention calls at runtime. By implementing Reduced Interaction Sampling (RIS), it bypasses the $O(N^2)$ memory and compute bottleneck of standard Transformers.

We utilize **Qwen2-1.5B** as a Proof of Concept (PoC). Demonstrating that RIS can stabilize and guide retrieval in a compact model proves that the architecture maintains contextual coherence even under severe parameter constraints, scaling naturally to larger architectures.

---

## ⚠️ Hardware Disclaimer & Performance

This implementation is optimized for **CPU-only execution** to enable long-context experiments on commodity academic machines (like standard workstations or departmental servers).

- **RAM Requirements**: ~100GB+ RAM is required for stable 65,536 token inference sessions.
- **CPU Performance**: 
    - **Prefill**: ~50 minutes for 65k tokens (one-time cost, cached thereafter).
    - **Generation**: ~5 seconds per token.
- **GPU Note**: CUDA support is experimental. Running on GPU will drastically reduce prefill/generation times but requires high VRAM.

---

## 🛠️ Folder Structure & Components

The repository is structured to run both locally and as a reproducible Code Ocean capsule:

*   `code/`: All execution scripts, entry points, and visualization modules.
    - [code/scripts/ris_attention.py](file:///home/anderson/repos/riskernel/code/scripts/ris_attention.py): Core implementation of the Reduced Interaction Sampling sparse geometry.
    - [code/scripts/inference_ris_v3.py](file:///home/anderson/repos/riskernel/code/scripts/inference_ris_v3.py): High-performance CPU-bound inference engine utilizing dual-hash caching and PFUS.
    - `code/scripts/benchmark/`: Execution scripts for running sweeps across context windows and densities.
    - `code/article/fig/`: Visualization scripts for generating plots.
*   `data/`: Mounted/local directory for context documents (`genppi.txt`, `aom.txt`, etc.).
*   `results/`: Directory where benchmarks and generated figures are outputted.

---

## 🚀 Getting Started

### 1. Installation (CPU-only)
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r code/scripts/requirements-cpu.txt
```

### 2. Prepare Context
The context articles are already pre-loaded under the `data/` folder. If you wish to use your own PDFs, you can use the `extract_pdf.py` utility from the manuscript repository to preprocess them into clean text blocks.

### 3. Run Inference
Launch the inference engine using python:
```bash
PYTHONPATH=code/scripts python code/scripts/inference_ris_v3.py \
  --model_class qwen2 \
  --window 65536 \
  --context_files data/genppi.txt \
  --density 0.05 \
  --n_seeds 1
```

#### Key Arguments:
- `--window`: Context window size in tokens.
- `--density`: Active attention density fraction (e.g., `0.01` for 1%, `0.05` for 5%).
- `--n_seeds`: Number of stochastically projected masks to ensemble.
- `--save_graph`: Exports the attention topology to a `.dot` file.

---

## 📊 Visualization
You can export the sparse attention topology with the `--save_graph` flag. Open the resulting `.dot` file in Graphviz or Gephi to inspect the attention retrieval maps.

---

## 📄 License & Citation

The code is available for scientific transparency and reproducibility under the MIT License. If you use this work, please cite the preprint:

```bibtex
@misc{santos2026riskernel,
  author    = {Santos, Anderson R.},
  title     = {RIS-Kernel: A Model-Agnostic Architecture for Long-Context LLM Inference via Sparse Attention},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20476759},
  url       = {https://doi.org/10.5281/zenodo.20476759}
}
```
