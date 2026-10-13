import numpy as np

def fit(X, k=50):
    """Fit PCA on faces ``X`` and return a model with ``k`` eigenfaces."""
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2
    assert 0 < k <= X.shape[0]

    mean_face = X.mean(axis=0)
    X_centered = X - mean_face      #centers the faces to have mean 0

    # Use the number-of-faces Gram matrix instead of the huge pixel matrix.
    gram = X_centered @ X_centered.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:k]        
    vectors = vectors[:, order]

    eigenfaces = X_centered.T @ vectors         #matrix multiplication of X_centered.T with the selected eigenvectors to get eigenfaces
    eigenfaces /= np.linalg.norm(eigenfaces, axis=0)    # normalize

    return {                                    #return a dictionary containing the mean face, eigenfaces, and eigenvalues
        "mean_face": mean_face,
        "eigenfaces": eigenfaces,
        "eigenvalues": eigenvalues[order],
    }

def project(model, face_vec):
    """Project one face vector onto the model's eigenfaces, then get its weight"""
    face_vec = np.asarray(face_vec, dtype=np.float64)         #pick up one face
    assert face_vec.ndim == 1
    assert face_vec.shape[0] == model["mean_face"].shape[0]
    return (face_vec - model["mean_face"]) @ model["eigenfaces"]
