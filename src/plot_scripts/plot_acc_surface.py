import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_acc_surface():
    from matplotlib import cm

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 15))

    # Make data.
    X = torch.arange(1, 4, 1)
    Y = torch.arange(3, 11, 1)
    X, Y = torch.meshgrid(X, Y, indexing="xy")
    Z = torch.rand(8, 3)

    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_formatter("{x:.02f}")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["a", "b", "c"])

    ax.set_xlabel("width")
    ax.set_ylabel("depth")

    plt.show()


plot_acc_surface()
