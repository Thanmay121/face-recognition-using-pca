import numpy as np

from app.pipeline import evaluate
from data.loader import get_data


def test_processed_data_has_expected_shapes_and_types():
    X_train, X_test, y_train, y_test = get_data()

    assert X_train.shape == (320, 10304)
    assert X_test.shape == (80, 10304)
    assert y_train.shape == (320,)
    assert y_test.shape == (80,)
    assert X_train.dtype == np.float64
    assert X_test.dtype == np.float64
    assert np.issubdtype(y_train.dtype, np.integer)
    assert len(np.unique(y_train)) == 40


def test_evaluate_supports_required_component_counts():
    for k in (10, 25, 50, 100, 200):
        accuracy = evaluate(k)
        assert 0.0 <= accuracy <= 1.0
