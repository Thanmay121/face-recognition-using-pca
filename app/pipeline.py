import numpy as np

from data.loader import get_data
from pca.engine import fit, project
from recognition.classifier import predict


def project_faces(model, faces):
	"""Project a face table with shape ``(n, d)`` to ``(n, k)``."""
	return np.vstack([project(model, face) for face in faces])


def evaluate(k):
	"""Fit on training faces and return test accuracy for ``k`` components."""
	X_train, X_test, y_train, y_test = get_data()
	model = fit(X_train, k=k)
	train_proj = project_faces(model, X_train)
	test_proj = project_faces(model, X_test)

	predictions = [
		predict(model, train_proj, y_train, face_proj)[0]
		for face_proj in test_proj
	]
	return np.mean(np.asarray(predictions) == y_test)


def main():
	for k in (10, 25, 50, 100, 200):
		print(f"k={k}: accuracy={evaluate(k):.2%}")


if __name__ == "__main__":
	main()
