# face-recognition-using-pca

Face recognition with eigenfaces (PCA), written in NumPy. Implemented the PCA step ourselves instead of using sklearn.

## How it works

Every face image is flattened into a vector of 10,304 pixel values (92 x 112). PCA finds the directions along which the faces vary the most, and each face is then described by a few coefficients along those directions instead of all the pixels. To recognize a new face, we project it the same way and pick the closest known face.

Steps:

1. Load the images, convert to grayscale, resize to 92 x 112, flatten
2. Split per person into 8 train and 2 test photos (fixed seed)
3. Subtract the mean face from the training faces
4. Find the top k eigenvectors (eigenfaces) of the training data
5. Project train and test faces onto them
6. Classify each test face by its nearest training face (Euclidean distance)

The eigenvectors are computed from the small 320 x 320 matrix `Xc @ Xc.T` and mapped back to pixel space with `Xc.T @ u`, which gives the eigenvectors of the full covariance matrix without ever building the 10304 x 10304 one.

## Setup

```
git clone https://github.com/Thanmay121/face-recognition-using-pca.git
cd face-recognition-using-pca
pip install -r requirements.txt
python -m app.pipeline
```

Run everything from the repo root.

## Results

Test accuracy on the 80 held-out faces:

| k | accuracy |
|---|----------|
| 10 | |
| 25 | |
| 50 | |
| 100 | |
| 200 | |

(add plots here: eigenfaces, reconstruction, accuracy vs k)

## Project structure

```
app/            pipeline and plots
data/           loading, preprocessing, train/test split
pca/            fit and project
recognition/    nearest neighbour classifier
```

## Dataset

AT&T (ORL) Database of Faces: 40 people, 10 grayscale images each. Details and terms are in `data/raw/README`.

## Limitations

Eigenfaces only works well on aligned, evenly lit photos like this dataset. It falls apart with pose and lighting changes, which is why modern systems use deep learning.

## License

MIT
