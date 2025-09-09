# Sleek-Price
A Machine Learning-Based Laptop Pricer
Notebook Change History
This section provides a summary of the significant changes made to this Colab notebook, along with explanations and justifications for each step.

Initial Setup and Data Loading:

Changes: Imported necessary libraries (pandas, numpy, matplotlib.pyplot, seaborn, sklearn). Mounted Google Drive to access the dataset. Loaded the laptop_price.csv file into a pandas DataFrame. Displayed the head of the DataFrame.
Explanation and Justification: These are standard initial steps in any data analysis project. Importing libraries provides the tools for manipulation, visualization, and modeling. Mounting Google Drive allows access to the dataset stored there. Loading the data into a DataFrame is essential for performing operations on it, and displaying the head helps in quickly understanding the data structure and column names.
Data Cleaning - Missing Values:

Changes: Checked for missing values using df.isnull().sum() and df.isnull().values.any().
Explanation and Justification: Identifying missing data is a crucial part of data cleaning. Missing values can cause errors or biases in analysis and modeling. In this case, no missing values were found, which simplifies the subsequent steps.
Data Cleaning - Outliers:

Changes: Visualized numerical features (Inches, Weight, Price_euros) using box plots. Defined and applied functions (remove_outliers_iqr, remove_outliers_zscore) to remove outliers based on IQR and Z-score methods. Converted 'Weight' to numeric before outlier removal. Printed the shape of the DataFrame before and after outlier removal.
Explanation and Justification: Outliers can significantly impact statistical analysis and machine learning models. Box plots help in visualizing the distribution and identifying potential outliers. Using both IQR (robust to skewed data) and Z-score (suitable for data closer to normal) methods allows for a tailored approach to outlier removal based on the characteristics of each numerical feature, aiming to improve model robustness.
