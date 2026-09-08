import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
from pathlib import Path
from ipywidgets import fixed, interactive, widgets


def to_torch(x, like=None):
    return torch.as_tensor(x, dtype=torch.float32 if like is None else like.dtype,
                           device=None if like is None else like.device)


def to_numpy(x):
    return x.detach().cpu().numpy()


figsize = (10, 7)


def plot_data(X, y, X_test, y_test, ax=None):
    ax_none = False
    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)
    clip_bound = 2.5
    ax.set_xlim(0, 1)
    ax.set_ylim(-clip_bound, clip_bound)
    ax.scatter(X[:, 0], y, c="darkorange", s=40.0, label="training data points")
    ax.plot(
        X_test, y_test, "--", color="royalblue", linewidth=2.0, label="Ground truth"
    )
    if ax_none:
        plt.show()


def plot_relu(bias, slope, ax=None):
    ax_none = False
    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)

    plot_x = np.array([0., 1.])
    if slope != 0:
        x_break = -bias / slope
        if 0 <= x_break <= 1:
            plot_x = np.array([0., x_break, 1.])
            ax.scatter([x_break], [0.], c="darkgrey", s=40.0)
    ax.plot(plot_x, np.maximum(0, slope * plot_x + bias), ":")

    if ax_none:
        plt.show()


def plot_relus(params, ax=None):
    slopes = to_numpy(params[0]).ravel()
    biases = to_numpy(params[1])
    for relu in range(biases.size):
        plot_relu(biases[relu], slopes[relu], ax=ax)


def plot_function(X_test, net, ax=None):
    ax_none = False
    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)
    with torch.no_grad():
        y_pred = net(to_torch(X_test, like=next(net.parameters())))
    ax.plot(X_test, to_numpy(y_pred), "-", color="forestgreen", label="prediction")
    if ax_none:
        plt.show()


def plot_update(X, y, X_test, y_test, net, state=None, optim="sgd", ax=None):
    ax_none = False
    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)

    if state is not None:
        net = copy.deepcopy(net)
        net.load_state_dict(state)
    plot_relus(list(net.parameters()), ax=ax)
    plot_function(X_test, net, ax=ax)
    plot_data(X, y, X_test, y_test, ax=ax)
    ax.legend()
    ax.set_title(optim)
    if ax_none:
        plt.show()


def train_network(
    X,
    y,
    X_test,
    y_test,
    net: torch.nn.Module,
    optim,
    n_steps,
    save_every,
    initial_weights=None,
    verbose=False,
    optimizer="sgd",
    seed=0,
    ckpt_dir=None,
):
    if n_steps < 1 or save_every < 1 or y.size < 1:
        raise ValueError("Use positive n_steps/save_every and nonempty training data.")
    loss = torch.nn.MSELoss()
    parameter = next(net.parameters())
    y_torch = to_torch(y.reshape(-1, 1), like=parameter)
    X_torch = to_torch(X, like=parameter)
    test_inputs = to_torch(X_test, like=parameter)
    test_targets = to_torch(y_test.reshape(-1, 1), like=parameter)
    history = {}
    rng = np.random.RandomState(seed)

    if initial_weights is not None:
        net.load_state_dict(initial_weights)
    net.train()

    for s in range(n_steps):
        subsample = rng.choice(y.size, max(1, y.size // 5))
        step_loss = loss(y_torch[subsample], net(X_torch[subsample, :]))
        optim.zero_grad()
        step_loss.backward()
        optim.step()
        if (s + 1) % save_every == 0 or s == 0 or s == n_steps - 1:
            # plot_update(X, y, X_test, y_test, net)
            history[s + 1] = {}
            history[s + 1]["state"] = copy.deepcopy(net.state_dict())
            with torch.no_grad():
                test_loss = loss(test_targets, net(test_inputs))
            history[s + 1]["train_error"] = to_numpy(step_loss).item()
            history[s + 1]["test_error"] = to_numpy(test_loss).item()
            if verbose:
                print(f"{optimizer} Iteration %d" % (s + 1))
                print("\tTrain Loss: %.3f" % to_numpy(step_loss).item())
                print("\tTest Loss: %.3f" % to_numpy(test_loss).item())

    if ckpt_dir is not None:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save(history, f"{ckpt_dir}/ckpt_and_history.pt")

    return history


def plot_test_train_errors(
    history, optim="", plot_test=False, plot_train=False, ax=None
):
    ax_none = False
    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)
    sample_points = np.array(list(history.keys()))
    etrain = [history[s]["train_error"] for s in history]
    etest = [history[s]["test_error"] for s in history]
    if plot_train:
        ax.plot(sample_points / 1e3, etrain, label=f"Train Error {optim}")
    if plot_test:
        ax.plot(sample_points / 1e3, etest, label=f"Test Error {optim}")
    ax.set_xlabel("Iterations (1000's)")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    if ax_none:
        plt.show()


TRANS = [0, 1, -1]


def plot_with_error_bar(
    list_of_history,
    optim="",
    plot_test=False,
    plot_train=False,
    idx=0,
    ax=None,
):
    ax_none = False
    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)
    if not list_of_history:
        raise ValueError("At least one training history is required.")
    sample_points = np.array(sorted(list_of_history[0]))
    if any(sorted(hist) != list(sample_points) for hist in list_of_history):
        raise ValueError("All runs must use the same recorded training steps.")
    etrain = [[hist[s]["train_error"] for s in sample_points] for hist in list_of_history]
    etest = [[hist[s]["test_error"] for s in sample_points] for hist in list_of_history]
    etrain, etest = np.array(etrain), np.array(etest)
    med_etrain, lower_train, upper_train = (
        np.median(etrain, axis=0),
        np.percentile(etrain, q=25, axis=0),
        np.percentile(etrain, q=75, axis=0),
    )
    med_etest, lower_test, upper_test = (
        np.median(etest, axis=0),
        np.percentile(etest, q=25, axis=0),
        np.percentile(etest, q=75, axis=0),
    )

    # Matplotlib expects distances from the median, not percentile endpoints.
    quantile_train = np.vstack((med_etrain - lower_train, upper_train - med_etrain))
    quantile_test = np.vstack((med_etest - lower_test, upper_test - med_etest))

    if plot_train:
        ax.errorbar(
            sample_points / 1e3 + 0.2 * TRANS[idx],
            med_etrain,
            yerr=quantile_train,
            fmt="-o",
            label=f"Train Error {optim}",
            capsize=5,
        )
    if plot_test:
        ax.errorbar(
            sample_points / 1e3 + 0.2 * TRANS[idx],
            med_etest,
            yerr=quantile_test,
            fmt="-o",
            label=f"Test Error {optim}",
            capsize=5,
        )
    ax.set_xlabel("Iterations (1000's)")
    ax.set_ylabel("Minibatch training MSE" if plot_train else "Test MSE")
    ax.set_yscale("log")
    ax.legend()
    if ax_none:
        plt.show()


def make_iter_slider(iters):
    iters = list(iters)
    if not iters:
        raise ValueError("The training history is empty.")
    return widgets.SelectionSlider(
        options=iters, value=iters[0], description="Training step: ", disabled=False
    )


def history_interactive(history, idx, X, y, X_test, y_test, net):
    plot_update(X, y, X_test, y_test, net, state=history[idx]["state"])
    plt.show()
    print("Train Error: %.3f" % history[idx]["train_error"])
    print("Test Error: %.3f" % history[idx]["test_error"])


def make_history_interactive(history, X, y, X_test, y_test, net):
    sample_points = list(history.keys())
    return interactive(
        history_interactive,
        history=fixed(history),
        idx=make_iter_slider(sample_points),
        X=fixed(X),
        y=fixed(y),
        X_test=fixed(X_test),
        y_test=fixed(y_test),
        net=fixed(net),
    )
