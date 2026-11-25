## Foreword:
The `DataCleaning` class is general and has been robustly built to handle any messy dataset. It creates a Pandas DataFrame that consists of all numeric columns, without unwanted duplicates and zero `NaN` values.

## Purpose:
For dimensionality reduction methods like PCA, and clustering methods like K-means to be most effective, input data needs to be in the best condition.

### What is required of the dataset:
* All numeric columns
* No empty or `NaN` (‘not a number’) values that would spoil subsequent matrix calculations
* Removal of duplicates 
* Possible treatment of any outliers

## Input:
The input dataset should be in the form of a Pandas DataFrame and created as an instance of the `DataCleaning` class.

As part of the initialisation method, a copy of the DataFrame is taken to prevent `SettingWithCopy` warnings and all empty cells are converted to `NaN`.

## Methods:
The `DataCleaning` Class has 12 steps / methods which should be performed sequentially, although note that some steps are optional (made clear below). These are summarised below:

1. `del_all_null_cols` - delete columns with all `NaN` values.
2. `print_datatypes` - check to see which columns are numeric, object or string.
3. `EDA` - summary statistics for each column to describe the data.
4. `convert_col_to_numbers` - convert any columns that aren't numeric, but should be numeric, to numeric. Uses the Pandas method `pd.to_numeric()`, with the argument `errors='coerce'`, forcing values that aren’t recognised as numbers to `NaN`. The desired column is to be entered as the argument of the method.
5. `del_non_numeric_cols` - delete any remaining non-numeric columns. Uses the Pandas library vectorisation method `data.select_dtypes(include='number')` to remove columns.
6. `find_na` - provides a summary of counts of `NaN` for each feature.
7. `drop` - (optional) an option to delete rows or columns that have a perceived unacceptable count of `NaN` values (note - remaining `NaN` values will be filled in step 8). Note that the threshold argument is the minimum number of non-`NaN` values accepted in a row or column. The axis direction argument must be either ‘rows’ or ‘columns’.
8. `fill` - for remaining `NaN` values in each column, impute either the mean, median or mode of the column. The column is the first argument, and the mean / median / mode is the second. For modal imputation, the first mode of a feature is used if the distribution is multi-modal.
9. `del_duplicates` - (optional) – the option to remove all exact replica rows, keeping only the first instance.
10. `get_boxplots` - a chance to see the distribution for each feature, particularly the amount of and scale of outliers.
11. `remove_outliers` - (optional) remove outliers using Tukey's method for each feature. Option to either ‘cap’ outlier values to the whiskers, or completely ‘drop’ rows containing the outliers.
12. `get_clean_data` – as a Pandas DataFrame.

## Library functions required:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Useful extensions
* The inclusion of categorical columns