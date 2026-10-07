may require
numpy
opencv-python
matplotlib
streamlit
install using pip install -r requirements.txt

data/loader - data to be imported by sujay and converted in a set of (x,y) pairs for training and testing the model - sujay's part
pca/engine - PCA engine which reduces the dimensionality of the data and extracts the most important features for classification - surya's part
recognition/classifier.py - takes the PCA features and trains a classifier to recognize the patterns in the data - sourabh's part
app/pipline - me

TLDR
load -> split train/test -> fit (train only) -> project -> predict -> accuracy

btw decide with yourselves about what function signatures you are using so others can use ur functions