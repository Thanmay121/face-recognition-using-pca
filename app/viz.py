"""Matplotlib visualizations for the eigenfaces pipeline."""

import math

import matplotlib.pyplot as plt
import numpy as np

from data.loader import get_data
from pca.engine import fit, project


IMAGE_HEIGHT = 112
IMAGE_WIDTH = 92


def _as_image(face):
    """Return a flattened face as an image with shape ``(112, 92)``"""
    return np.asarray(face).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)


def plot_mean_face(mean_face):
    """Return a figure showing one mean face vector"""
    figure, axis = plt.subplots()
    axis.imshow(_as_image(mean_face), cmap="gray")
    axis.set_title("Mean face")
    axis.axis("off")
    figure.tight_layout()
    return figure


def plot_eigenfaces(eigenfaces, count=12):
    """Return a figure showing the first ``count`` eigenface columns"""
    eigenfaces = np.asarray(eigenfaces)
    count = min(count, eigenfaces.shape[1])

    columns = min(4, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < count:
            axis.imshow(_as_image(eigenfaces[:, index]), cmap="gray")
            axis.set_title(f"Eigenface {index + 1}")
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_reconstructions(mean_face, eigenfaces, face, projections):
    """Return a figure comparing a face with reconstructions at several ``k`` values."""
    mean_face = np.asarray(mean_face)
    eigenfaces = np.asarray(eigenfaces)
    face = np.asarray(face)
    projections = list(projections)

    figure, axes = plt.subplots(1, len(projections) + 1, squeeze=False)
    axes = axes[0]
    axes[0].imshow(_as_image(face), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    for axis, projection in zip(axes[1:], projections):
        projection = np.asarray(projection)
        component_count = min(projection.size, eigenfaces.shape[1])
        reconstruction = mean_face + eigenfaces[:, :component_count] @ projection[:component_count]
        axis.imshow(_as_image(reconstruction), cmap="gray")
        axis.set_title(f"k={component_count}")
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_accuracy(k_values, accuracies):
    """Return a figure plotting accuracy against the number of components."""
    figure, axis = plt.subplots()
    axis.plot(k_values, accuracies, marker="o")
    axis.set_xlabel("Number of components (k)")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy by number of components")
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(matrix, labels=None):
    """Return a figure showing a confusion matrix."""
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion matrix")
    if labels is not None:
        axis.set_xticks(range(len(labels)), labels=labels)
        axis.set_yticks(range(len(labels)), labels=labels)
    figure.tight_layout()
    return figure


def main():
    """Display a small visual summary of the fitted eigenface model."""
    X_train, _, _, _ = get_data()
    model = fit(X_train, k=25)
    projection = project(model, X_train[0])

    plot_mean_face(model["mean_face"])
    plot_eigenfaces(model["eigenfaces"], count=12)
    plot_reconstructions(
        model["mean_face"],
        model["eigenfaces"],
        X_train[0],
        [projection[:k] for k in (5, 15, 25)],
    )
    plt.show()


if __name__ == "__main__":
    main()
