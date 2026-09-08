# HW02 Q4 — Fall 2026

Open `q_linearized_features.ipynb`. The notebook generates its own regression data and contains its helpers; no earlier homework, dataset download, or repository clone is required.

Use Python 3.9–3.12. In Google Colab, run the notebook's `%pip` dependency cell. For a local environment, install these packages into the same Python environment used by Jupyter:

```text
python -m pip install "numpy>=1.22,<2" "matplotlib>=3.5,<4" "torch>=2.0,<3" "ipywidgets>=8,<9" "notebook>=6,<8"
python -m notebook
```

Run the cells in order. Complete the marked TODOs in parts (a), (b), and (c) as you reach them. Part (a) differentiates the scalar network output; part (b) uses the training inputs to fit one SVD, then evaluates its principal features on `X_test`; part (c) extends the same procedure to two hidden layers. The written handout gives the submission requirements.

Keep the supplied widths `[10, 20, 40]`, SGD learning rate `0.02`, and `150000` updates per network. Training time depends on the device. The notebook chooses CUDA when available and CPU otherwise. The fixed seed makes repeat runs reproducible on the same device and software setup. Keep initial and trained network objects separate when comparing features.

The NumPy upper bound supports older installed PyTorch versions that cannot exchange arrays with NumPy 2. The code was verified with Python 3.9.12, NumPy 1.22.4, PyTorch 2.0.1, Matplotlib 3.5.3, and ipywidgets 8.1.9. Do not interpret shorter instructor verification runs as replacement homework settings.
