# HW09: Coding SSM Forward

Complete both starter notebooks and submit their code and written answers as
directed by the handout. The CPU notebook covers recurrence and dense causal
convolution. The GPU notebook repeats that comparison and adds diagonal
recurrence and depthwise convolution.

- [CPU notebook in Colab](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw09/code/q_coding_ssm_forward_cpu.ipynb)
- [GPU notebook in Colab](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw09/code/q_coding_ssm_forward_gpu.ipynb)

Use Python 3.10 or later, PyTorch 2.0 or later, Matplotlib, and a Jupyter notebook
interface. Colab includes these packages. For a local environment, install the
PyTorch build appropriate for your hardware using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install the remaining dependencies:

```text
python -m pip install matplotlib jupyterlab
python -m jupyter lab
```

Windows and Linux can use a supported CUDA installation, or run the CPU notebook
with the CPU PyTorch build. Apple Silicon Macs can use the MPS backend in a
supported PyTorch/macOS installation. The GPU notebook chooses CUDA, then MPS,
then CPU. A CPU fallback lets you verify correctness; it does not supply a GPU
benchmark. A Colab GPU can be used for this SSM problem if your machine lacks a
supported GPU. For the separate hardware-check problem, follow that problem's
requirement to measure your own machine.

Fill the TODO functions before running the sanity checks. The model has no
nonlinearity; the bias enters at every time step. The stored output at index
`t` is the hidden state after input `x[:, t, :]`. All tensors and intermediate
results must preserve the input's device and dtype.

The timing cells use small default grids, warmup, inference mode, and CUDA/MPS
synchronization. They time input projection, kernel construction, and the
forward evaluation together. Each method receives the same input. Start with
these grids; larger cases are optional. Dense kernels use `T * H * H` entries,
so a large hidden dimension and sequence length can consume substantial
memory. Report dimensions, device/backend, software version, CPU threads,
actual timings, and any changes to the grid. Compare CPU and GPU with the
same `N`, `D`, `H`, and `T` values. Hardware differences and library algorithms
can change which method is fastest.

These notebooks benchmark a direct `conv1d` implementation. They do not claim
FFT convolution, a universal convolution speedup, or an absence of CPU
parallelism. All released starter outputs are cleared and no reference
implementations are included in the student code folder.

## Problem 4: Hardware Check

Use [hardware_check.py](hardware_check.py) and the
[hardware setup guide](README-hardware.md) for the Qwen3.5 size ladder on your
own machine. The runner performs the prescribed inference, training,
quantization, and offloading checks. Submit only its `hardware-results.csv`;
no written hardware analysis or modified benchmark code is required.
