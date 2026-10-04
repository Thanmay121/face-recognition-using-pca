import os
import cv2
import numpy as np
path = r"C:\Users\sujay\.cache\kagglehub\datasets\kasikrit\att-database-of-faces\versions\2"
print(os.listdir(path))
path = r"C:\Users\sujay\.cache\kagglehub\datasets\kasikrit\att-database-of-faces\versions\2"

W, H = 92, 112  # cv2.resize takes (width, height)

# folders s1..s40, sorted numerically
people = sorted(
    [d for d in os.listdir(path) if d.startswith("s") and os.path.isdir(os.path.join(path, d))],
    key=lambda x: int(x[1:])
)

X, y = [], []
for person in people:
    folder = os.path.join(path, person)
    for fname in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)  # load + grayscale
        if img is None:
            continue
        img = cv2.resize(img, (W, H))
        X.append(img.flatten())          # 112*92 = 10304
        y.append(int(person[1:]))        # identity label 1..40

X = np.array(X, dtype=np.uint8)
y = np.array(y)

print(X.shape, y.shape)  # expect (400, 10304) (400,)

# save
np.save("faces_X.npy", X)
np.save("faces_y.npy", y)

# optional CSV: label in first column, then 10304 pixel columns
np.savetxt("faces.csv", np.column_stack([y, X]), delimiter=",", fmt="%d")