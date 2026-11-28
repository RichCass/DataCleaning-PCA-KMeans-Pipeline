## DataCleaning-PCA-KMeans-Pipeline

The pipeline consists of 3 classes, each with an associated ReadMe file
* DataCleaning [Python Class](./DataCleaning.py) | [Documentation](./DataCleaning_README.md)
* PCA [Python Class](./PCA.py) | [Documentation](./PCA_README.md)
* KMEANS [Python Class](./KMEANS.py) | [Documentation](./KMEANS_README.md)

## Highlights

* A cleaning class that ensures all values are numeric (with no empty or NaN values), elimination of unwanted duplicate records and different options for the treatment of outliers.
* A PCA class that performs PCA using linear algebra and Singular Value Decomposition (SVD). It provides an elbow plot, a variance explained plot, 2D and 3D component plots plus a 2D biplot to show how the original features are represented by the key components.
* A K-means class that enables a user to select any number of clusters   
