# CS 182/282A Fall 2026 — HW02 optimization and initialization experiments

Run `hw2_optimizer_init.ipynb` from top to bottom and discuss the requested plots
and measurements in the Q3 portion of your written PDF. All optimizer,
initialization, gradient-logging, and training implementations are supplied.
The default experiments require no code changes.

## Google Colab

Open the [Fall 2026 notebook](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw02/code/hw2_optimizer_init.ipynb)
and run the setup cells. Setup first moves to `/content`, reconnects a stale
Drive mount if necessary, and verifies that `MyDrive` is accessible.

The persistent workspace is `MyDrive/cs182hw2_fa26`. Setup prints the exact
experiment directory. If an existing `cs182fa26_public` checkout predates the
supplied experiments, setup uses a separate `cs182fa26_public_q3_experiments_v1`
checkout; older files and edits are preserved.

Setup uses working installed NumPy, Matplotlib, and imageio packages immediately.
Only missing packages trigger pip. Installation output is visible, network
operations have a 20-second timeout and one retry, and the complete installer
has a 180-second limit. If setup reports a failure, check the connection,
restart the session, rerun setup, and complete any Drive authorization prompt.
A failed import of an installed package reports a fresh-runtime recovery step.
Automatic module reloading is optional; if it is unavailable and you edit Python
files while exploring, restart the session and rerun the notebook.

## Local setup

Use Python 3.11 or 3.12. From this directory, create and activate a virtual
environment, then install the local environment and open JupyterLab:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open this notebook next to `assignment_utils.py` and `deeplearning/`.
The Colab-only Drive steps skip automatically on a local machine.

## Experiments

The notebook compares SGD and momentum, larger minibatches, adaptive optimizers,
and zero/random/He initialization, then trains a deeper network with the supplied
reference settings. Keep the supplied seeds and settings for the first run.
To include the supplied optional RMSProp experiment, change
`RUN_OPTIONAL_RMSPROP = False` to `True` in the imports cell.
Optional exploration may change settings after the supplied run; use validation
data for comparisons and evaluate the test subset only after choosing a model.

Histories and measurements are written to `experiment_logs/` for your own
inspection. Use the notebook's plots and measured values in your written
PDF discussion.

## CIFAR-10 data

The helper downloads the original 170 MB CIFAR-10 Python archive from the
[BrainChip mirror](https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz),
with the [official Toronto server](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
as fallback. Both sources must pass MD5 `c58f30108f718f92721af3b95e74349a`
before safe extraction into `deeplearning/datasets/`. Progress is printed and
each source uses a 60-second connection/read timeout. A verified archive saved
manually to `deeplearning/datasets/cifar-10-python.tar.gz` is also accepted.
The experiments use several GB of working memory; runtime varies by machine.
