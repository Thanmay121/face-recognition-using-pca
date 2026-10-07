import numpy as np
from sklearn.decomposition import PCA

def pca(X_train):
    pca=PCA(n_components=100)   #Keep the 100 most important principal components.
                                #At this point, PCA hasn't actually calculated the components yet. 
                                #We have just created the PCA object and specified what we want.
    pca.fit(X_train)            #Automatically centers the data and variates with the PCA value
    return pca


def transform(X,pca):
    return pca.transform(X) 



    






