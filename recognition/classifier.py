import numpy as np
from data_prep import get_data
from eigenfaces import PCA1  # Replace with Surya's actual filename


def evaluate_classifier(k=50):
    """Run nearest-neighbor classification and return the test accuracy."""
    X_train, X_test, y_train, y_test = get_data()

    mean = X_train.mean(axis=0)
    components = PCA1(X_train, k)

    # Center both sets using the training mean, then project.
    train_vectors = (X_train - mean) @ components
    test_vectors = (X_test - mean) @ components

    predictions = []

    for test_vector in test_vectors:
        distances = np.linalg.norm(train_vectors - test_vector, axis=1)
        nearest_index = np.argmin(distances)
        predictions.append(y_train[nearest_index])

    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_test)

    print(f"Accuracy with {k} components: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    evaluate_classifier(k=50)