import numpy as np


def predict(model, train_proj, train_y, face_proj):
    """Return the nearest training label and Euclidean distance.

    ``train_proj`` contains projected training faces with shape ``(n, k)``;
    ``face_proj`` is one already-projected face with shape ``(k,)``.
    ``model`` is accepted to preserve the project interface, but prediction
    only needs the projected arrays.
    """
    del model

    train_proj = np.asarray(train_proj, dtype=float)
    train_y = np.asarray(train_y)
    face_proj = np.asarray(face_proj, dtype=float)

    assert train_proj.ndim == 2
    assert train_y.ndim == 1 and train_y.shape[0] == train_proj.shape[0]
    assert face_proj.ndim == 1 and face_proj.shape[0] == train_proj.shape[1]
    assert train_proj.shape[0] > 0

    distances = np.linalg.norm(train_proj - face_proj, axis=1)
    nearest_index = np.argmin(distances)
    return train_y[nearest_index], distances[nearest_index]