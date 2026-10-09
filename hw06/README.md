# HW06 notebooks

These notebooks accompany the coding questions in Homework 6. Start with the written homework for the required experiments and submissions. The notebooks contain intentional student TODOs; finish each TODO before running cells that depend on it.

| Question | Notebook | Open in Colab |
| --- | --- | --- |
| 1: CNN inductive bias | [edge_detection.ipynb](code/edge_detection.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/edge_detection.ipynb) |
| 4: GPU memory | [GPUMemory.ipynb](code/GPUMemory.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/GPUMemory.ipynb) |
| 5: Muon | [q_coding_muon.ipynb](code/q_coding_muon.ipynb) | [Open](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw06/code/q_coding_muon.ipynb) |

The TensorBoard, Weights & Biases and karate-club notebooks in `code/` are not part of Homework 6. You do not need to complete them for this assignment.

## Setup

Use a Python 3 runtime and run cells in order. Colab already provides PyTorch and torchvision; retain that compatible pair. For local execution, install a compatible [PyTorch and torchvision pair](https://pytorch.org/get-started/locally/) first, and install the other packages listed in each notebook. CNN inductive bias also uses NumPy, SciPy, Matplotlib, Pillow, and tqdm.

The GPU-memory and Muon notebooks download CIFAR-10 on first use. The CNN inductive-bias notebook generates its own images.

A GPU is recommended for the CNN experiments and required for meaningful GPU-memory measurements in Question 4. A CPU run of that notebook records unavailable CUDA memory measurements as missing values; it cannot answer the GPU-memory questions. Run times and feasible batch sizes depend on the hardware.

## Execution checks and full experiments

- **GPU memory:** set `SMOKE_TEST = True` in the setup cell for a small execution check. Set it back to `False` for the requested measurements. Check the printed active settings. Smoke logs and experiment logs are saved in different folders. Out-of-memory configurations are recorded explicitly; do not report stale results from an earlier run. Memory is reported in decimal MB unless marked otherwise.
- **Muon:** complete both required implementation TODOs, use `SMOKE_TEST = True` to check execution, and use `False` for the reported comparison. The optional hyperparameter sweep is disabled by default and can require substantially more compute.

Keep the random seed, dataset size, epoch budget, optimizer settings, and hardware with your results. An execution check shows that code and logging work; its metrics are not a completed experiment.

Contributors: Matteo Guarrera, Mert Cemri, Sizhe Chen.
