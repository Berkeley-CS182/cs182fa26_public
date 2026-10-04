# HW06 notebooks

These notebooks accompany the coding questions in Homework 6. Start with the written homework for the required experiments and submissions. The notebooks contain intentional student TODOs; finish each TODO before running cells that depend on it.

| Question | Notebook | Open in Colab |
| --- | --- | --- |
| 1: CNN inductive bias | [edge_detection.ipynb](code/edge_detection.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/edge_detection.ipynb) |
| 4: GPU memory | [GPUMemory.ipynb](code/GPUMemory.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/GPUMemory.ipynb) |
| 7(a): TensorBoard | [tensorboard.ipynb](code/tensorboard.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/tensorboard.ipynb) |
| 7(b): Weights & Biases | [wandb.ipynb](code/wandb.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/wandb.ipynb) |
| 8: Zachary's karate club | [q_zkc.ipynb](code/q_zkc.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/q_zkc.ipynb) |
| 9: Muon | [q_coding_muon.ipynb](code/q_coding_muon.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/q_coding_muon.ipynb) |

## Setup

Use a Python 3 runtime and run cells in order. Colab already provides PyTorch and torchvision; retain that compatible pair. The notebooks install missing logging or graph packages where indicated. For local execution, install a compatible [PyTorch and torchvision pair](https://pytorch.org/get-started/locally/) first, and install the other packages listed in each notebook. CNN inductive bias also uses NumPy, SciPy, Matplotlib, Pillow, and tqdm.

For TensorBoard and W&B, keep [architectures.py](code/architectures.py) next to the notebook. A standalone Colab session automatically downloads that helper from this repository if it is missing. The CIFAR-10 notebooks download their dataset on first use. The CNN inductive-bias notebook generates its own images, and the karate-club notebook uses NetworkX's built-in graph.

The graph notebook runs on CPU. A GPU is recommended for the CNN experiments and required for meaningful GPU-memory measurements in Question 4. A CPU run of that notebook records unavailable CUDA memory measurements as missing values; it cannot answer the GPU-memory questions. Run times and feasible batch sizes depend on the hardware.

## Execution checks and full experiments

- **GPU memory:** set `SMOKE_TEST = True` in the setup cell for a small execution check. Set it back to `False` for the requested measurements. Check the printed active settings. Smoke logs and experiment logs are saved in different folders. Out-of-memory configurations are recorded explicitly; do not report stale results from an earlier run. Memory is reported in decimal MB unless marked otherwise.
- **TensorBoard and W&B:** each notebook first runs a short two-batch example. Complete `run()` and the configuration list, then set `RUN_EXPERIMENTS = True`. These examples do not replace the required five TensorBoard and ten W&B configurations. Keep the epoch budget fixed, use the fixed validation split to compare settings, and reserve the official test set for a final evaluation after tuning.
- **W&B:** the example defaults to offline logging, which needs no account. To inspect its hosted dashboard, follow the notebook's interactive login and online-run or offline-sync instructions. Do not save an API key in a notebook or its outputs.
- **Muon:** complete both required implementation TODOs, use `SMOKE_TEST = True` to check execution, and use `False` for the reported comparison. The optional hyperparameter sweep is disabled by default and can require substantially more compute.

Use fresh run names and keep the random seed, dataset size, epoch budget, optimizer settings, and hardware with your results. An execution check shows that code and logging work; its metrics are not a completed experiment.

Contributors: Matteo Guarrera, Mert Cemri, Sizhe Chen.
