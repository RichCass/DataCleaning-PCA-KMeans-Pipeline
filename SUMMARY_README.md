## DataCleaning-PCA-KMeans-Pipeline

The pipeline consists of 3 classes, each with an associated ReadMe file
* DataCleaning [Python Class](./DataCleaning.py) | [Documentation](./DataCleaning_README.md)
* PCA [Python Class](./PCA.py) | [Documentation](./PCA_README.md)
* KMEANS [Python Class](./KMEANS.py) | [Documentation](./KMEANS_README.md)

## Highlights

* Readying a dataset for further analysis such as PCA and K-Means. Ensuring all values are numeric (with no empty or NaN values), the elimination of unwanted duplicate records and providing different options for the treatment of outliers.
* Building up PCA through using linear algebra. Centering and scaling the dataset, performing Singular Value Decomposition (SVD) and transforming the centred and scaled dataset to give the PCA matrix.
* Providing an elbow plot and a variance explained plot to aid the decision over how many principal components to use. Also 2D and 3D component vs component plots and a 2D biplot. 
* Performing K-means cluster analysis on as many columns / principal components as desired. Opportunity to view a 2D or 3D representation of those clusters.  
