import numpy as np

from data.loader import get_data
from pca.engine import fit, project
from recognition.classifier import predict


def project_faces(model, faces):
	"""Project a face table with shape ``(n, d)`` to ``(n, k)``"""
	return np.vstack([project(model, face) for face in faces])


def evaluate(k):
	"""Fit on training faces and return test accuracy for ``k`` components"""
	X_train, X_test, y_train, y_test = get_data()
	# Learn the PCA model from training faces only to keep the test evaluation fair.
	model = fit(X_train, k=k)
	train_proj = project_faces(model, X_train)
	test_proj = project_faces(model, X_test)

	# Classify each test face by finding its nearest projected training face.
	predictions = [
		predict(model, train_proj, y_train, face_proj)[0]
		for face_proj in test_proj
	]
	# The mean of True/False values gives the fraction classified correctly.
	return np.mean(np.asarray(predictions) == y_test)


def main():
	# Compare recognition accuracy when different numbers of eigenfaces are kept.
	k_values = (10, 25, 50, 100, 200)
	accuracies = [evaluate(k) for k in k_values]
	for k, accuracy in zip(k_values, accuracies):
		print(f"k={k}: accuracy={accuracy:.2%}")

	from app.viz import plot_accuracy
	import matplotlib.pyplot as plt

	plot_accuracy(k_values, accuracies)
	plt.show()


if __name__ == "__main__":
	main()
