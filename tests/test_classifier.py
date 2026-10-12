import numpy as np

from recognition.classifier import predict


def test_predict_returns_closest_label_and_distance():
    train_proj = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 10.0]])
    train_y = np.array([7, 8, 9])
    face_proj = np.array([2.0, 3.0])

    label, distance = predict(None, train_proj, train_y, face_proj)

    assert label == 8
    assert np.isclose(distance, np.sqrt(2.0))
