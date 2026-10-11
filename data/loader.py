import os
import cv2
import numpy as np

W, H = 92, 112

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

NAMES = ("X_train", "X_test", "y_train", "y_test")


def load_faces():
    people = sorted(
        [d for d in os.listdir(RAW_DIR)
         if d.startswith("s") and os.path.isdir(os.path.join(RAW_DIR, d))],
        key=lambda x: int(x[1:]),
    )
    if not people:
        raise FileNotFoundError(f"No s1..s40 folders found in {RAW_DIR}")

    X, y = [], []
    for person in people:
        folder = os.path.join(RAW_DIR, person)
        for fname in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (W, H))
            X.append(img.flatten())
            y.append(int(person[1:]))
    return np.array(X, dtype=np.uint8), np.array(y)


def stratified_split(X, y, test_size=0.2, seed=42):
    """Per-person split so every person appears in both train and test."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def build_processed(test_size=0.2, seed=42):
    X, y = load_faces()
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size, seed)
    for name, arr in zip(NAMES, (X_train, X_test, y_train, y_test)):
        np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), arr)
    print("Saved:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def get_data():
    """Surya calls this. Loads from data/processed."""
    return tuple(np.load(os.path.join(PROCESSED_DIR, f"{n}.npy")) for n in NAMES)


if __name__ == "__main__":
    build_processed()
