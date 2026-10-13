***PCA finds directions that capture the greatest variance, meaning the greatest spread of the data. It then represents the data along selected directions.***



**Step 1: Calculate the mean face**



Why do we calculate the mean?

Different faces share common structure. Subtracting the mean lets PCA focus on how individual images differ from the average, rather than their common baseline.



**Step 2: Center the training images**



Why is Centering necessary? PCA should identify directions of variation around the average. Without centering, the common mean can influence the covariance structure and the resulting principal directions.



**Step 3: Calculate the Gram matrix**



***GRAM MATRIX:***

Each entry in ***GRAM MATRIX*** measures the dot product between two centered training faces. These dot products describe how the training samples relate to each other in the original pixel space.



This measures how their centered pixel-value patterns relate to each other in the original feature space.

\- A large positive dot product generally indicates similar directions of variation.

\- A negative dot product indicates opposing directions of variation.

\- A dot product near zero indicates little linear alignment.



Why use the Gram matrix? It allows the project to calculate the principal directions through a much smaller matrix than the full (10304,10304) pixel-space matrix.



**Step 4: Find eigenvalues and eigenvectors**



In PCA, the larger eigenvalues correspond to directions that capture more variance in the centered training data.

The eigenvector describes a direction in sample space, where each coordinate corresponds to a training face.



**Step 5: Select the top-k directions**



**order = np.argsort(eigenvalues)\[::-1]\[:k]**

**vectors = vectors\[:, order]**



np.argsort(eigenvalues) returns indices that would sort the eigenvalues in ascending order.

\[::-1] reverses that order, putting the largest eigenvalues first.

\[:k] selects the first k indices.

vectors\[:, order] selects the corresponding eigenvectors, keeping all rows and only the selected columns.



Why select the largest eigenvalues? They correspond to the directions capturing the greatest variance, which is the objective of PCA.



**step-6 Construct and normalize the eigenfaces**



**eigenfaces = X\_centered.T @ vectors**

**eigenfaces /= np.linalg.norm(**

&#x20;   **eigenfaces, axis=0**

**)**



The eigenvectors in vectors have 320 entries, corresponding to the 320 training samples. They describe directions in sample space, not directly in the 10,304-dimensional pixel space.

Multiplying by X\_centered.T maps those directions into pixel space.

Each column of eigenfaces now contains 10,304 values, so it can be reshaped into a \\(92\\times112\\) image for visualization.

Eigenfaces represent the most significant ways in which the training face images differ from their mean face.



normalization ensures that every eigenface has unit length. This gives a consistent scale when projecting a face onto the eigenfaces.



**step-7 Project a FACE onto EIGEN FACES**

**return (face\_vec - model\["mean\_face"]) @ model\["eigenfaces"]**



&#x09;**Step A** — Subtract the mean face ...(face\_vec - model\["mean\_face"])=x\_c

&#x09;(face\_vec - model\["mean\_face"]

&#x09;

&#x09;**Step B** — Multiply by the eigenfaces x\_c @ V

&#x09;Here, V is the matrix of selected eigenfaces.

&#x09;The result contains the projection coefficients: the values that describe the face along the selected eigenface directions.

&#x09;

&#x09;

