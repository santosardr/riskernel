# RIS-Kernel Code Ocean Capsule

> [!NOTE]
> This capsule provides the concrete systems-level implementation and practical execution environment for **RIS-Kernel**, which is a continuation of the theoretical foundations established in the original [RIS Repository](https://github.com/santosardr/ris).

This directory replicates the structured environment required for running the RIS-Kernel benchmark suite on a Code Ocean capsule.

## Capsule Folder Structure

*   `code/`: Contains all execution scripts, requirements, entry points, and visualization modules.
    *   `run`: The bash entrypoint executed by Code Ocean.
    *   `scripts/`: Contains the core inference and sparse attention scripts.
        *   `scripts/ris_attention.py`: Core RIS sparse attention logic.
        *   `scripts/inference_ris_v3.py`: LLM loading, configuration, and inference wrapper.
        *   `scripts/benchmark/run_benchmark.py`: Grid sweep script for context windows and densities.
        *   `scripts/benchmark/results_64.csv` & `results_structural_sweet_spot_64.csv`: Pre-cached historical benchmarks.
        *   `scripts/benchmark32/results_baseline_32k.csv`: Pre-cached 32k context benchmarks.
    *   `article/fig/`: Contains the python plotting scripts for data visualization.
        *   `gen_fig_accuracy_window.py`: Plots accuracy vs window sizes (up to 64k).
        *   `gen_fig_heatmap_dual.py`: Plots density vs seeds heatmaps.
*   `data/`: Mounted input folder for read-only datasets and articles.
    *   Contains context articles: `genppi.txt`, `aom.txt`, `ajinshanensis.txt`, `meta.txt`.
*   `results/`: Mounted output directory. Contains the generated `results_64.csv` and primary figures:
    *   `fig_accuracy_window.pdf`
    *   `fig_heatmap_seeds_density.pdf`

---

## Configuring the Code Ocean Environment

To run this capsule successfully on Code Ocean:

1.  **Environment Settings**:
    *   Set the base environment to **Ubuntu 20.04/22.04** or **Python 3.10**.
    *   Install the dependencies listed in `code/scripts/requirements-cpu.txt` using the Code Ocean package manager (preferred) or let the `run` script install them.

2.  **Dataset Mounting**:
    *   Mount the context articles inside the `/data` folder of the capsule.
    *   **Pre-cached Models**: It is highly recommended to upload a tar.gz archive of the pre-cached model weights (`huggingface_hub_cache.tar.gz`) as a Data Asset mounted directly under `/data/models/`. The entrypoint script (`run`) will automatically detect the archive, verify if the files are already present (skipping extraction if so), and selectively extract only the required Qwen2 and TinyLlama model weights to `$HOME/.cache/huggingface` to avoid timeouts or disk space exhaustion.

---

## Resource & Execution Tuning

Code Ocean capsules run on standard instances with default limits on **RAM (e.g., 16GB)** and **execution time (e.g., 1-10 hours)**. 

To prevent out-of-memory errors and execution timeouts:

1.  **Baseline Bypassing**:
    *   Running the baseline (`w == 0`) with dense attention inside the RIS module can cause massive CPU indexing memory spikes (>20GB) when sequence lengths are high. The benchmark automatically uses `--disable_ris` for the baseline run to execute native dense attention and fit within the 16GB limit.
2.  **Window Size Bounds**:
    *   The `run` entrypoint is configured to test window sizes up to **16k** (`--windows "0,4096,8192,16384"`).
    *   If your capsule has a GPU or a higher-memory CPU instance (e.g., 32GB+ RAM), you can extend this to run full-context sweeps up to **64k** by modifying the `--windows` parameter in the `run` script:
        ```bash
        --windows "0,4096,8192,16384,32768,65536"
        ```
