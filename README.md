## DataCleaning-PCA-KMeans-Pipeline

The pipeline consists of 3 classes, each with an associated ReadMe file:
* DataCleaning [Python Class](./DataCleaning.py) | [Documentation](./DataCleaning_README.md)
* PCA [Python Class](./PCA.py) | [Documentation](./PCA_README.md)
* KMEANS [Python Class](./KMEANS.py) | [Documentation](./KMEANS_README.md)

## Highlights

* Readies a dataset for further analysis such as PCA and K-Means. Ensuring all values are numeric (with no empty or `NaN` values), the elimination of unwanted duplicate records and providing different options for the treatment of outliers.
* Builds up PCA using linear algebra: centring and scaling the dataset, performing Singular Value Decomposition (SVD) and transforming the centred and scaled dataset to give the PCA matrix.
* Provides an elbow plot and a variance explained plot to aid the decision over how many principal components to use. Also, 2D & 3D component vs component plots and a 2D biplot are supplied to aid the identification of any groups in the data and understand how the original columns are represented by the components. 
* Performs K-means cluster analysis on as many columns / principal components as desired. Opportunity to view a 2D or 3D representation of those clusters.  
