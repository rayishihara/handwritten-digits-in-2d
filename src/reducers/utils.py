import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).parents[2] / "data"

def load_mnist(root=DATA_DIR, train=True):
    # Load the MNIST split as a torch dataset of (1, 28, 28) tensors in [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return datasets.MNIST(root, train=train, download=True, transform=transform)

def to_matrix(dataset):
    # Flatten a dataset into an (n_samples, 28*28) array and its labels
    X = np.stack([x.numpy().ravel() for x, _ in dataset])
    return X, dataset.targets.numpy()

def plot_embedding(Z, y, ax=None, title=None):
    # Scatter a 2-D embedding coloured by digit, with a label at each class median
    ax = ax or plt.gca()
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        pts = Z[y == d]
        ax.scatter(*pts.T, s=2, alpha=0.6, color=cmap(d), label=str(d))
        ax.annotate(str(d), np.median(pts, axis=0), ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="circle", fc="white", ec=cmap(d), alpha=0.8))
    ax.set_title(title)
    ax.set_axis_off()
    return ax
