import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataCleaning:
    def __init__(self, data: pd.DataFrame):  # Use a Pandas DataFrame as the input
        """Initialises the pipeline with a copy of the DataFrame to prevent
        SettingWithCopy warnings and converts empty whitespace strings to NaN"""
        self.data = data.copy()
        self.data = self.data.replace(r'^\s*$', np.nan, regex=True)

    def instructions(self):
        """Documents the key steps in the pipeline"""
        print("1. Delete columns with all NaN values")
        print("2. Check column datatypes")
        print("3. Perform EDA")
        print("4. Convert any columns that aren't numeric, but should be numeric, to numeric")
        print("5. Delete any non-numeric columns")
        print("6. Find NaN counts for each column")
        print("7. Drop rows or columns that fall below a minimum of non-NaN counts (OPTIONAL)")
        print("8. Fill remaining NaN cells using either the mean, median or mode of a column")
        print("9. Delete any exact duplicate rows (OPTIONAL)")
        print("10. Obtain boxplots for all remaining numeric columns")
        print("11. Remove outliers using Tukey's method (1.5 * IQR) (OPTIONAL)")
        print("12. Get the cleaned data")

    # 1.
    def del_all_null_cols(self):
        """Delete columns with all NaN values"""
        self.data = self.data.dropna(axis=1, how='all')  # Delete columns where all are NaN

    # 2.
    def print_datatypes(self):
        """Prints DataFrame shape and column types"""
        print(self.data.info())

    # 3.
    def EDA(self):
        """Prints Descriptive Statistics"""
        print(self.data.describe())

    # 4.
    def convert_col_to_numbers(self, column):
        """Convert columns that should be numeric to numeric"""
        self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

    # 5.
    def del_non_numeric_cols(self):
        """Vectorised removal of all non-numeric columns"""
        self.data = self.data.select_dtypes(include='number')

    # 6.
    def find_na(self):
        """Count of NaN cells by column"""
        na = self.data.isna().sum()
        return na[na != 0]

    # 7. OPTIONAL
    def drop(self, threshold, axis_direction):
        """Drop rows / columns with threshold <= minimum number of non-NaN values (OPTIONAL)"""
        axis_map = {'rows': 0, 'columns': 1}

        # Check axis parameter = rows or columns
        if axis_direction not in axis_map:
            raise ValueError("axis must equal rows or columns")

        else:
            self.data = self.data.dropna(thresh=threshold, axis=axis_map[axis_direction])

    # 8.
    def fill(self, column, method):
        """Fill NaN cells using column mean, median or mode"""
        # Check method parameter = mean, median or mode
        acceptable_methods = ['mean', 'median', 'mode']
        if method not in acceptable_methods:
            raise ValueError("acceptable methods are mean, median and mode")

        else:
            if method == 'mean':
                self.data[column] = self.data[column].fillna(self.data[column].mean())
            elif method == 'median':
                self.data[column] = self.data[column].fillna(self.data[column].median())
            else:
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])

    # 9. OPTIONAL
    def del_duplicates(self):
        """Keep only first record (row) of exact duplicates (OPTIONAL)"""
        self.data = self.data.drop_duplicates(keep='first')

    # 10.
    def get_boxplots(self):
        """Obtain a boxplot for each column to view the distributions and outliers"""
        # Set up key varables for plotting and for loop
        columns = self.data.columns
        num_cols = len(columns)

        # Validating there are remaining columns in the DataFrame
        if num_cols == 0:
            raise ValueError("No columns provided to plot")

        fig, axs = plt.subplots(num_cols, 1, figsize=(10, 2.5 * num_cols))

        # Ensure axs is always a flat array, even with one column
        if num_cols == 1:
            axs = [axs]

        # Creating the subplots for each column
        for i, col_name in enumerate(columns):
            axs[i].boxplot(self.data[col_name], vert=False)
            axs[i].set_title(col_name, fontsize=14)

        fig.suptitle("Boxplots of Explanatory Variables for PCA / K-Means", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.975])  # Leaves a gap above the subplots for the suptitle
        plt.show()

    # 11. OPTIONAL
    def remove_outliers(self, approach='cap'):
        """Cap or drop approach to outliers using bounds of LQ - 1.5 * IQR and UQ + 1.5 * IQR (OPTIONAL)"""
        # Approaches:
        # 'cap' - clips outlier values to the whiskers. Preserves 100% of rows
        # 'drop' - drops rows if they contain an outlier

        if approach not in ['cap', 'drop']:
            raise ValueError("Approach must be cap or drop")

        # Setting up a columns variable for the for loop underneath
        columns = self.data.columns

        # 'cap' approach
        if approach == 'cap':
            for col in columns:
                UQ = self.data[col].quantile(0.75)  # Upper Quartile
                LQ = self.data[col].quantile(0.25)  # Lower Quartile
                IQR = UQ - LQ  # IQR
                lower_bound = LQ - 1.5 * IQR  # Lower bound for permitted data
                upper_bound = UQ + 1.5 * IQR  # Upper bound for permitted data

                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)

        # 'drop approach'
        else:
            # Start with a True baseline mask
            global_mask = pd.Series(True, index=self.data.index)

            for col in columns:
                UQ = self.data[col].quantile(0.75)  # Upper Quartile
                LQ = self.data[col].quantile(0.25)  # Lower Quartile
                IQR = UQ - LQ  # IQR
                lower_bound = LQ - 1.5 * IQR  # Lower bound for permitted data
                upper_bound = UQ + 1.5 * IQR  # Upper bound for permitted data

                column_mask = self.data[col].between(lower_bound, upper_bound)
                global_mask = global_mask & column_mask

                # Remove rows with outliers once at the end
            self.data = self.data[global_mask]

    # 12.
    def get_clean_data(self):
        """Get the cleaned DataFrame"""
        self.data = self.data.reset_index(drop=True)  # Required for array masking in PCA
        return self.data
