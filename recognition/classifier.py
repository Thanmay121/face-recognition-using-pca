import numpy as np
from data_prep import get_data
from eigenfaces import PCA1  # Replace with Surya's actual filename

X_train, X_test, y_train, y_test = get_data()

k = 50
mean = X_train.mean(axis=0)

# PCA1 currently centers internally and returns the components.
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