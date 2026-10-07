import numpy as np
import joblib
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA


#Step-1 load the data

X=np.load("faces_X.npy")   # Loading the numpy arrays to perform operations onto them
y=np.load("faces_y.npy")

#Step-2 split the data into train and test

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# random_state=42  fixes the random split so we get the same result every time 
#...we can choose 10,100,25 anything 

#Step-3 Centering the data by subtrating the values by mean to know how the data varies around its mean.

mean=np.mean(X_train,axis=0)  # axis=0 → mean of each pixel across all faces
X_centered=X_train-mean       # subtracts each value by mean

#Step-4 Calculate 100 pca componenets and 
pca=PCA(n_components=100)   #Keep the 100 most important principal components.
                            #At this point, PCA hasn't actually calculated the components yet. 
                            #We have just created the PCA object and specified what we want.
pca.fit(X_centered)         #PCA examines the centered data and finds directions in which the data varies the most

X_train_pca=pca.transform(X_centered) # Convert training data into PCA features
                                        #We will Take each face and express it using the new PCA features.
X_test_centered=X_test-mean
X_test_pca=pca.transform(X_test_centered)

#step-5 saving the files
np.save("X_train_pca.npy", X_train_pca)
np.save("y_train.npy", y_train)

np.save("X_test_pca.npy", X_test_pca)
np.save("y_test.npy", y_test)

joblib.dump(pca, "pca_model.pkl") # Saving the new PCA features 






