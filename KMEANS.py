from scipy.cluster.vq import kmeans2
import numpy as np
import matplotlib.pyplot as plt


class KMEANS():
    def __init__(self, X: np.ndarray, clusters: int, dimensions_to_plot: int = 2, random_state: int = None):
        """Initialises K-means for n clusters"""

        X_arr = np.asarray(X, dtype=float)

        # Input validation
        if X_arr.ndim != 2:
            raise ValueError("Dataset must be a 2D array with rows and more than 1 column")
        if clusters < 2:
            raise ValueError("Clusters must be at least 2")
        if dimensions_to_plot not in (2, 3):
            raise ValueError("Dimensions to plot must be 2 or 3")
        if np.shape(X_arr)[1] < dimensions_to_plot:
            raise ValueError("There must be sufficient columns to plot")

        self.random_state = random_state  # ensuring centres chosen for k-means are reproducible
        self.clusters = clusters
        self.dimensions_to_plot = dimensions_to_plot
        self.X = X_arr.copy()  # .copy() to prevent SettingWithCopy warnings
        self.centres = None
        self.labels = None

    def fit(self):
        """Perform K-means using kmeans2 from scipy.cluster.vq"""
        # minit="++" for carefully selected starting points and seed=self.random_state for reproducibility
        self.centres, self.labels = kmeans2(self.X, self.clusters, minit="++", seed=self.random_state)
        return self

    def plot(self):
        """2D or 3D plot of clusters from kmeans2 algorithm"""

        # Ensure fit() run before plot()
        if self.centres is None:
            raise RuntimeError("Run fit() method before plot()")

        fig = plt.figure(figsize=(10, 7))

        if self.dimensions_to_plot == 2:
            ax = fig.add_subplot()

            for i in range(self.clusters):
                mask = self.labels == i
                ax.scatter(self.X[mask, 0], self.X[mask, 1], marker='^', label=f"Group {i + 1}")

            # Plot the 2D cluster centres
            ax.scatter(self.centres[:, 0], self.centres[:, 1], color='black', marker='x', label='Centres',
                       s=250)

            ax.set_title(f"K-Means Cluster Analysis, Clusters = {self.clusters}", fontsize=14)
            ax.set_xlabel("Dimension 1", fontsize=12, fontweight='bold')
            ax.set_ylabel("Dimension 2", fontsize=12, fontweight='bold')


        elif self.dimensions_to_plot == 3:
            ax = fig.add_subplot(projection='3d')

            for i in range(self.clusters):
                mask = self.labels == i
                ax.scatter(self.X[mask, 0], self.X[mask, 1], self.X[mask, 2], marker='^', label=f"Group {i + 1}")

            # Plot the 3D cluster centres
            ax.scatter(self.centres[:, 0], self.centres[:, 1], self.centres[:, 2], color='black',
                       marker='x', label='Centres', s=250, depthshade=False)

            ax.set_title(f"K-Means Cluster Analysis, Clusters = {self.clusters}", fontsize=14)
            ax.set_xlabel("Dimension 1", fontsize=12, fontweight='bold')
            ax.set_ylabel("Dimension 2", fontsize=12, fontweight='bold')
            ax.set_zlabel("Dimension 3", fontsize=12, fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True)
        ax.legend(title='Group', bbox_to_anchor=(1.3, 1), loc='upper right')

        plt.show()

        return self

    def print_centres_labels(self):
        print(f"centre coordinates: {self.centres} \ngroup labels: {self.labels}")