import numpy as np
from sklearn.decomposition import PCA

def pca(X_train):
    pca=PCA(n_components=100)   #Keep the 100 most important principal components.Thes 100 pca values/features are combination of all the pixels
                                #At this point, PCA hasn't actually calculated the components yet. 
                                #We have just created the PCA object and specified what we want.
    
    
    pca.fit(X_train)            #Automatically centers the data and finds the direction where the faces have the most variation
                                #fit() does not give you the reduced data.It just gives the 100 components to the pca object
    return pca


def transform(X,pca):
    return pca.transform(X)     #transform() reduces the dimensionality to the number of components you specified.



    






