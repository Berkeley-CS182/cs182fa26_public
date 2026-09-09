# Homework 02 student notebooks - Fall 2026

Run the notebooks with a Python 3 Colab CPU runtime. They include all data and
code needed for the assigned exercises; no training-data download is needed.

- [SGD and interpolation](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw01/code/sgd_interpolation.ipynb): for Problem 1, run the top import/helper cells and Part (4). You may use 100,000 iterations as stated in the handout.
- [Watching Momentum Work](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw02/code/q_sgd_momentum_analysis.ipynb): run all cells for Problem 2(h)/(i) and Problem 5. Submit the two written answers once.
- [Optional eigenvalue visualization](https://colab.research.google.com/github/Berkeley-CS182/cs182fa26_public/blob/main/hw02/code/q_sgd_momentum_analysis_visualization.ipynb): run all cells to see eigenvalues move as the learning rate changes.
- [Optional companion video](https://github.com/Berkeley-CS182/cs182fa26_public/blob/main/hw02/code/q_sgd_momentum_analysis_visualization.mp4).

The momentum example uses singular values 2 and 1, beta=3/4, GD learning rate
1/5, and momentum learning rate 1/3. Both parameters and momentum start at zero.
The notebooks require NumPy and Matplotlib; the animation also uses IPython,
which Colab includes. A standard Google account is needed to run code in Colab;
viewing the public notebook does not require GitHub repository access.

If Colab is unavailable, download the notebook from this public repository and
run it in a local Jupyter environment with NumPy and Matplotlib installed.

Contributors: Matteo Guarrera, Mert Cemri, Sizhe Chen, Suhong Moon, Gabriel Goh,
Anant Sahai, Peter Wang, Yuxi Liu.
