import numpy as np

from data.loader import stratified_split


def test_stratified_split_is_reproducible_and_keeps_each_label():
    X = np.arange(40 * 3, dtype=np.float64).reshape(40, 3)
    y = np.repeat(np.arange(1, 5), 10)

    first = stratified_split(X, y, test_size=0.2, seed=42)
    second = stratified_split(X, y, test_size=0.2, seed=42)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)

    X_train, X_test, y_train, y_test = first
    assert X_train.shape == (32, 3)
    assert X_test.shape == (8, 3)
    assert set(y_train) == set(y_test) == {1, 2, 3, 4}
