import numpy as np

def PCA1(X_train,k):
    mean = np.mean(X_train,axis=0)  #axis=0 finds the mean for each column
    X_centered = X_train-mean      
    n = X_centered.shape[0]   #gives Number of rows
    covariance = (X_centered.T @ X_centered)/(n-1)    #X_centered.T is nothing but transpose
    eigenvalues,eigenvectors = np.linalg.eig(covariance)  
    indices = np.argsort(eigenvalues)[::-1]   #contains indices that correspond to eigen values in descending order
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    components = eigenvectors[:, :k]  #Now we take the top 100 eigenvectors...these are nothing but our 100 most imp directions(pca values)
    return components

def PCA2(X_test,components):
    return X_test @ components



    






