import pandas as pd
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class PCA:
    def __init__(self, data: pd.DataFrame):  # Use a Pandas DataFrame as the input
        """Initialises PCA, with a copy of the cleaned DataFrame to prevent SettingWithCopy
        warnings. Sets up matrices for linear algebra operations"""
        self.data = data.copy()
        self.X_bar = None
        self.U = None
        self.s_squared = None
        self.V = None
        self.Z = None

    def centre_scale_dataset(self):
        """Scales and centres the cleaned DataFrame"""
        self.X_bar = ((self.data - self.data.mean()) /
                      self.data.std())
        return self

    def SVD(self):
        """Decomposes the scaled, centred DataFrame using SVD"""
        self.U, s, VT = svd(self.X_bar, full_matrices=False)  # full_matrices argument to save memory
        self.s_squared = s ** 2
        self.V = VT.T  # Un-transpose matrix V
        return self

    def transform(self):
        """Obtains the principal components matrix"""
        # Rotation of the centred data matrix so that the new basis is in the directions
        # of maximum variance
        self.Z = self.X_bar @ self.V
        return self

    # Add a master method that performs all 3 methods above
    def fit_transform(self):
        """Executes the full PCA pipeline sequentially"""
        return self.centre_scale_dataset().SVD().transform()

    def elbow_var_explained_plots(self):
        """Elbow and variance explained plots to aid selection of the number of
        principal components"""

        # Producing elbow and variance explained plots for the principal components
        fig = plt.figure(figsize=(14, 7))

        # Add a GridSpec to control the horizontal spacing between the 2 subplots
        gs = fig.add_gridspec(1, 2, wspace=0.4)

        # Creating x-tick values for both plots
        x = np.arange(1, len(self.s_squared) + 1)

        # Creating variance explained results
        variance = self.s_squared / (len(self.data) - 1)  # Note: divide by DOF
        y = np.cumsum(variance) / sum(variance)

        # Left block: elbow plot
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(x, self.s_squared, marker='+', color='blue')
        ax1.set_title("Elbow Plot", fontsize=14)
        ax1.set_xlabel("Principal Component", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Singular Value Squared", fontsize=12, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=11)

        # Right block: variance explained plot
        ax2 = fig.add_subplot(gs[1])
        ax2.scatter(x, y, marker='o', color='purple')
        ax2.set_title("Variance Explained by each Principal Component", fontsize=14)
        ax2.set_xlabel("Principal Component", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Variance Explained", fontsize=12, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=12)

        # Add a vertical divider between the 2 plots
        divider = Line2D([0.51, 0.51], [0.08, 0.92], transform=fig.transFigure,
                         color='black', linewidth=2, zorder=5)  # preserve IDE formatting
        fig.add_artist(divider)

        plt.show()

    def components_plot(self):
        """Scatter plots of the first two and first three principal components"""
        z_matrix = self.Z.to_numpy()  # Strip away unwanted column headers

        # Initial defensive block for just two components if only two features
        if len(self.data.columns) < 3:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(z_matrix[:, 0], z_matrix[:, 1], marker='o', color='grey')
            ax.set_title("Principal Component 1 vs 2 (Low Dimensional Domain)", fontsize=14)
            ax.set_xlabel("Principal Component 1", fontsize=12, fontweight='bold')
            ax.set_ylabel("Principal Component 2", fontsize=12, fontweight='bold')
            ax.grid(True)
            ax.set_aspect('equal')
            plt.show()
            return

        # Plotting the first 2 and first 3 components separately
        fig = plt.figure(figsize=(14, 7))

        # Add a GridSpec to control the horizontal spacing between the 2 subplots
        gs = fig.add_gridspec(1, 2, wspace=0.4)

        # Left block: 2D view of first 2 components
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(z_matrix[:, 0], z_matrix[:, 1], marker='o', color='grey')
        ax1.set_title("Principal Component 1 vs 2", fontsize=14)
        ax1.set_xlabel("Principal Component 1", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Principal Component 2", fontsize=12, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.set_aspect('equal')  # Equal aspect ratio
        ax1.grid(True)

        # Right block: 3D view of first 3 components
        ax2 = fig.add_subplot(gs[1], projection='3d')
        ax2.scatter(z_matrix[:, 0], z_matrix[:, 1], z_matrix[:, 2], marker='o', color='grey')
        ax2.set_title("Principal Component 1 vs 2 vs 3", fontsize=14)
        ax2.set_xlabel("Principal Component 1", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Principal Component 2", fontsize=12, fontweight='bold')
        ax2.set_zlabel("Principal Component 3", fontsize=12, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_box_aspect((1, 1, 1))  # Equal aspect ratio
        ax2.grid(True)

        # Add a vertical divider between the 2 plots
        divider = Line2D([0.51, 0.51], [0.08, 0.92], transform=fig.transFigure,
                         color='black', linewidth=2, zorder=5)  # preserve formatting for IDEs
        fig.add_artist(divider)

        plt.show()

    def biplot(self):
        """2D biplot to demonstrate how the initial columns are represented by the
        first two components. Includes up to 8 most impactful original features"""

        z_matrix = self.Z.to_numpy()  # Strip away unwanted column headers
        arrow_vectors = self.V  # From the eigenvectors of the covariance matrix
        scale_factor = 4.0  # to make the arrows readable

        # Indexing the arrow sizes
        arrow_magnitudes = np.sqrt(self.V[:, 0] ** 2 + self.V[:, 1] ** 2)
        top_8_indices = np.argsort(arrow_magnitudes)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(z_matrix[:, 0], z_matrix[:, 1], marker='o', color='lightgrey')

        # Identifying the weights for each column from the first 2 columns of V
        for i, col_name in enumerate(self.data.columns):
            if i in top_8_indices[-8:]:  # Take the largest 8 arrows to reduce visual clutter
                x_value = arrow_vectors[i, 0] * scale_factor
                y_value = arrow_vectors[i, 1] * scale_factor

                # Plotting the arrows and text
                ax.arrow(0, 0, x_value, y_value, color='black', head_width=0.12, head_length=0.15,
                         length_includes_head=True)
                ax.text(x_value * 1.2, y_value * 1.2, col_name, color='black', ha='right',
                        va='center')

        ax.set_title("2D Biplot: PC1 vs PC2 Loadings (Top 8 Dimensions)", fontsize=14)
        ax.set_xlabel("Principal Component 1", fontsize=12, fontweight='bold')
        ax.set_ylabel("Principal Component 2", fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_aspect('equal')  # Equal aspect ratio
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.show()