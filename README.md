# RIS-Kernel: A Model-Agnostic Architecture for Long-Context LLM Inference via Sparse Attention

[![Theoretical Paper DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--026--59160--z-green)](https://doi.org/10.1038/s41598-026-59160-z)
[![RIS-Kernel DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20476759.svg)](https://doi.org/10.5281/zenodo.20476759)
[![Code Ocean](https://img.shields.io/badge/Code%20Ocean-Capsule-blue)](https://doi.org/10.24433/CO.0351350.v1)

> [!NOTE]
> **RIS-Kernel** is the concrete systems-level implementation and continuation of the original **RIS (Reduced Interaction Sampling)** framework. While the theoretical foundations, mathematical proofs, and initial simulations are established in the [RIS Repository](https://github.com/santosardr/ris) (theory) and detailed in the scientific article ([DOI: 10.1038/s41598-026-59160-z](https://doi.org/10.1038/s41598-026-59160-z)), this repository delivers the practical implementation, kernel execution patterns, and CPU-bound inference wrapper (practice).

RIS-Kernel is a model-agnostic runtime attention patching layer that enables running massive context windows (64k+ tokens) on commodity, unaccelerated CPU hardware. Rather than being a compiled standalone binary engine, it is implemented as a dynamic wrapper that intercepts standard Transformer self-attention calls at runtime, utilizing sparse stochastic geometry to bypass the quadratic memory bottleneck.

---

## 📖 Abstract

> Full self-attention in large language models scales as $O(N^2)$, limiting long-context document analysis to 65,536 tokens and requiring costly GPU clusters. The Reduced Interaction Sampling (RIS) inference engine addresses this constraint as a model-agnostic architecture. Without modifying weights, RIS reduces self-attention complexity to $O(N \log N)$ using sparse stochastic geometry that fits within commodity memory limits. We validate RIS on Qwen2-1.5B-Instruct across two regimes. In controlled evaluations at 32,768 tokens (where native dense attention serves as the upper bound), RIS-Stochastic at 1% density and 70 ensemble seeds achieves 75.00% accuracy, outperforming the native dense baseline (71.88%), while RIS-Stochastic at 5% density and 10 seeds matches it (71.88%). This demonstrates that sparse attention acts as a regularizer: low density (1%) over multiple seeds filters out sequence-level noise, whereas higher density (5%) reintroduces distractor noise. Under the tightest budget, RIS-Structural reaches 68.75% accuracy at 1% density with just 10 seeds, recovering 75% of the contextual gap relative to the zero-context floor (59.38%). At 65,536 tokens, where dense attention triggers out-of-memory faults, RIS yields retrieval gains of up to 14.06 percentage points over the zero-context floor (51.56%). All evaluations run on commodity, unaccelerated CPU servers (16–128 GB of RAM), demonstrating that long-context LLM inference is feasible on standard academic hardware without GPU acceleration.

---

## 🔬 Scientific Context & PoC

RIS-Kernel acts as a model-agnostic layer that intercepts attention calls at runtime. By implementing Reduced Interaction Sampling (RIS), it bypasses the $O(N^2)$ memory and compute bottleneck of standard Transformers.

We utilize **Qwen2-1.5B** as a Proof of Concept (PoC). Demonstrating that RIS can stabilize and guide retrieval in a compact model proves that the architecture maintains contextual coherence even under severe parameter constraints, scaling naturally to larger architectures.

---

## 🛠️ Folder Structure & Components

The repository is structured to run both locally and as a reproducible Code Ocean capsule:

*   `code/`: All execution scripts, entry points, and visualization modules.
    - [ris_attention.py](file:///home/anderson/repos/riskernel-github/riskernel/code/scripts/ris_attention.py): Core implementation of the Reduced Interaction Sampling sparse geometry.
    - [inference_ris_v3.py](file:///home/anderson/repos/riskernel-github/riskernel/code/scripts/inference_ris_v3.py): High-performance CPU-bound inference engine utilizing dual-hash caching and PFUS.
    - [benchmark](file:///home/anderson/repos/riskernel-github/riskernel/code/scripts/benchmark/): Execution scripts for running sweeps across context windows and densities.
    - [fig](file:///home/anderson/repos/riskernel-github/riskernel/code/article/fig/): Visualization scripts for generating plots.
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
The inference driver [inference_ris_v3.py](file:///home/anderson/repos/riskernel-github/riskernel/code/scripts/inference_ris_v3.py) supports two modes: interactive chat and non-interactive prompt queries.

> [!TIP]
> **Optimized Defaults**: RIS-Kernel is pre-configured with the optimal hyperparameters to maximize coherence and retrieval accuracy:
> *   **RIS Attention Density (`--density`)**: `0.03` (3% sparsity)
> *   **Ensemble Projections (`--n_seeds`)**: `30` independent seeds
> *   **Temperature (`--temp`)**: `0.2`
> *   **Repetition Penalty (`--repetition_penalty`)**: `1.1`
> 
> Under normal operation, **you do not need to modify these parameters** as their defaults represent the calibrated winning combination.

#### Non-Interactive Prompt Mode (Recommended for batch queries)
To run a single query against long documents and exit immediately:

```bash
PYTHONPATH=code/scripts python code/scripts/inference_ris_v3.py \
  --model_class qwen2 \
  --window 32768 \
  --context_files data/genppi.txt \
  --prompt "What is the role of Random Forest in GenPPi?"
```

You can also read the prompt query from a text file:
```bash
PYTHONPATH=code/scripts python code/scripts/inference_ris_v3.py \
  --model_class qwen2 \
  --window 32768 \
  --context_files data/genppi.txt \
  --prompt_file data/prompt.txt
```

#### Interactive Chat Mode
To start a multi-turn chat session inside your terminal:

```bash
PYTHONPATH=code/scripts python code/scripts/inference_ris_v3.py \
  --model_class qwen2 \
  --window 32768 \
  --context_files data/genppi.txt
```

---

## ⚙️ Technical Architecture & CPU Benchmarks

To help researchers and engineers understand the performance characteristics and resource footprint of this prototype:

### Performance & Hardware Footprint (Qwen2-1.5B-Instruct in float32)
*   **RAM Requirements**: ~100GB+ RAM is required for stable 65,536 token inference sessions.
*   **Prefill**: ~50 minutes for 65k tokens (one-time cost, cached thereafter).
*   **Generation**: ~5 seconds per token.
*   **GPU Note**: CUDA support is experimental. Running on GPU will drastically reduce prefill/generation times but requires high VRAM.

### Under the Hood: Why the High RAM & Prefill Overhead?
1. **PyTorch CPU Sparse Limitations (Prefill Overhead)**:
   The current implementation is an algorithmic Proof-of-Concept written in high-level Python/PyTorch. During the **prefill** phase, PyTorch's native CPU backend (`scaled_dot_product_attention`) does not support sparse tensor layouts. Even though RIS defines a highly sparse attention geometry (e.g., 1%-5% active connections), PyTorch still materializes the dense boolean attention mask of shape `(batch, heads, seq_len, seq_len)` in RAM.
   
   For a 65,536-token context and 12 attention heads in `float32`, this single intermediate matrix requires:
   $$12 \times 65,536 \times 65,536 \times 4 \text{ bytes} \approx 206 \text{ GB of RAM}$$
   This triggers massive virtual memory swapping to disk on standard workstations (16GB–128GB RAM), leading to the 50-minute prefill bottleneck.

2. **Persistent Caching**:
   To bypass this one-time PyTorch prefill cost, RIS-Kernel automatically serializes the prefilled KV-cache to disk. For subsequent queries in the same context, the prefill is completely skipped, loading the cached state in ~90 seconds.

3. **Generation Phase Advantage**:
   During the **generation (decoding)** phase, the $O(N)$ attention memory scaling is completely bypassed. Instead of performing attention over all $N = 65,536$ past tokens, the wrapper uses `torch.index_select` to slice the KV cache. It attends only to the union of a local sliding window (e.g., 1024 tokens) and the stochastic samples (e.g., 1% density = ~655 tokens), reducing active computations to just ~1,679 tokens. This keeps CPU latency stable and prevents out-of-memory crashes during chat turns.

4. **Production Path**:
   Because this is an algorithmic prototype, the speed numbers reflect PyTorch CPU overhead. Porting the RIS sparse geometry to low-level C++ (e.g., as a custom `llama.cpp` block-sparse kernel) or a Triton GPU kernel would avoid materializing the dense attention matrix, resulting in instantaneous prefill and native decoding speeds.

---

## 📊 Visualization

You can export the sparse attention topology with the `--save_graph` flag. Open the resulting `.dot` file in Graphviz or Gephi to inspect the attention retrieval maps.

---

## 📄 License & Citation

The code is available for scientific transparency and reproducibility under the MIT License. If you use this work, please cite both the theoretical paper and the repository implementation:

```bibtex
@article{santos2026ris,
  author    = {Santos, Anderson R.},
  title     = {Towards million-token context windows: a topology-preserving framework for adaptive transformer sparsification},
  journal   = {Scientific Reports},
  year      = {2026},
  doi       = {10.1038/s41598-026-59160-z},
  url       = {https://doi.org/10.1038/s41598-026-59160-z}
}

@misc{santos2026riskernel,
  author    = {Santos, Anderson},
  title     = {RIS-Kernel: A Model-Agnostic Architecture for Long-Context LLM Inference via Sparse Attention},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20814085},
  url       = {https://doi.org/10.5281/zenodo.20814085}
}
```
