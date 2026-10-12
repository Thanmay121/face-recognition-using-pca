import numpy as np

from pca.engine import fit, project


def make_faces(seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(8, 5))


def test_fit_returns_normalized_sorted_eigenfaces():
    X = make_faces()
    model = fit(X, k=3)

    assert model["mean_face"].shape == (5,)
    assert model["eigenfaces"].shape == (5, 3)
    assert model["eigenvalues"].shape == (3,)
    np.testing.assert_allclose(
        np.linalg.norm(model["eigenfaces"], axis=0),
        np.ones(3),
    )
    assert np.all(np.diff(model["eigenvalues"]) <= 0)


def test_project_uses_the_training_mean():
    X = make_faces()
    model = fit(X, k=3)
    face = X[0]

    projection = project(model, face)
    expected = (face - X.mean(axis=0)) @ model["eigenfaces"]

    assert projection.shape == (3,)
    np.testing.assert_allclose(projection, expected)


def test_fit_does_not_use_data_outside_training_input():
    X = make_faces()
    model = fit(X, k=2)
    altered_test_face = np.full(5, 1000.0)

    np.testing.assert_allclose(model["mean_face"], X.mean(axis=0))
    np.testing.assert_allclose(
        project(model, altered_test_face),
        (altered_test_face - X.mean(axis=0)) @ model["eigenfaces"],
    )
