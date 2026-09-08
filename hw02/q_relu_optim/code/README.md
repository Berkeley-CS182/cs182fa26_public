# EECS 182 Fall 2026: ReLU optimizers

Open `q_relu_optim.ipynb` in Colab using the homework link, or run it locally from this folder. For a fresh local environment, install `requirements.txt` using the same Python environment as Jupyter. Keep `helpers.py` and the entire `ckpts` folder beside the notebook.

Run all active cells. The default workflow loads the 270 supplied training histories (three widths, three optimizers, 30 seeds) and displays training/test curves and final ReLU elbows. The gradient-accumulation bug is intentionally shown as non-executing Markdown for students to diagnose.

The optional training block is not required. If uncommented, it sets `USE_PRETRAINED=False` so subsequent plots use the newly trained results. Historical checkpoint training losses are minibatch MSE; test losses use the full test set. Error bars show the 25th to 75th percentiles around the median.

This homework copy derives from the Fall 2025 `dis02/code` experiment and carries the reviewed Fall 2026 fixes. `nets_by_size.pth` is retained as a legacy asset but is not required: current code constructs models and loads each history's final state directly.
