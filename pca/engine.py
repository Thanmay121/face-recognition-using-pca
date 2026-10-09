import numpy as np


def fit(X,k):
    """Learn PCA components using the smaller X @ X.T matrix."""
    X=np.asarray(X,dtype=float)    # Converts the data into numpy arrays and each and every value to floating point number

    mean=np.mean(X,axis=0)          #Mean of each column 
    X_centered=X-mean               #Centers X

    gram=X_centered @ X_centered.T  

    eigenvalues,U=np.linalg.eigh(X_centered)    #np.linalg.eigh() returns eigenvalues in ascending order.
                                                #Its eigenvectors are stored as columns of U.

    index=np.argsort(eigenvalues)[::-1]  
    eigenvalues=eigenvalues[index]
    U=U[:,index][:,:k]          #We reorder the columns of U in the same way,
                                #then keep only the first 'k' columns(eigen vector corresponding to largest eigen value)

    components=(X_centered.T @ U)/np.sqrt(eigenvalues)

    return {"mean":mean,"components":components,"eigenvalues":eigenvalues}


def project(model,face_vec):
    face_vec=np.asarray(face_vec,dtype=float)
    centered_face=face_vec-model["mean"]
    return centered_face @ model["components"]

    




    






