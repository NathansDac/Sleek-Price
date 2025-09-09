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
Data Cleaning - Data Type Conversion:

Changes: Converted the Ram column from string (e.g., '8GB') to integer by removing 'GB'. Converted the Weight column to float after removing 'kg'.
Explanation and Justification: Machine learning models typically require numerical input. The 'Ram' and 'Weight' columns were initially strings containing units ('GB', 'kg'). Converting them to numerical types (integer for RAM, float for Weight) is necessary for numerical operations and model compatibility.
Feature Engineering - Categorical Features (One-Hot Encoding):

Changes: Performed one-hot encoding on Company, TypeName, Cpu_Brand, Cpu_Type, Gpu_Brand, and OpSys_Type columns using pd.get_dummies(). drop_first=True was used.
Explanation and Justification: Categorical features (like Company, TypeName, etc.) need to be converted into a numerical format for most machine learning algorithms. One-hot encoding creates new binary columns for each category. drop_first=True is used to avoid multicollinearity, which can be an issue in linear models.
Feature Engineering - Screen Resolution:

Changes: Extracted width and height from the ScreenResolution string using regex. Calculated PPI (Pixels Per Inch). Created a binary Touchscreen column based on the presence of 'Touchscreen' in resolution types. Extracted the Screen_Resolution_Type. Dropped the original ScreenResolution column.
Explanation and Justification: The original ScreenResolution column contained multiple pieces of information in a single string. Extracting width, height, and calculating PPI provides numerical features that are likely better predictors of price than the original string. Creating a binary Touchscreen feature simplifies this information for the model. Dropping the original column removes redundant information.
Feature Engineering - CPU:

Changes: Created functions to extract Cpu_Brand, Cpu_Type, and Cpu_Clock_Speed from the Cpu string using keyword matching and regex. Applied these functions to create new columns. Dropped the original Cpu column.
Explanation and Justification: Similar to screen resolution, the Cpu column contained multiple attributes. Separating the brand, type, and clock speed allows the model to learn the individual impact of these factors on price. Dropping the original column removes redundant information.
Feature Engineering - Memory:

Changes: Created a function to parse the Memory string and extract the sizes for SSD, HDD, and Flash Storage. Applied the function to create new columns (SSD_GB, HDD_GB, Flash_Storage_GB). Dropped the original Memory column.
Explanation and Justification: The Memory column combined storage type and size. Separating these into distinct numerical features (size in GB for each type) provides the model with more structured information about the storage configuration. Dropping the original column removes redundant information.
Dropping Redundant/Unnecessary Columns:

Changes: Dropped the laptop_ID and Product columns.
Explanation and Justification: laptop_ID is just an identifier and has no predictive power. The Product column had too many unique values and was too granular, making it unsuitable as a feature for this model.
Exploratory Data Analysis (EDA):

Changes: Generated a correlation matrix heatmap of numerical features. Created scatter plots of Ram vs. Price_euros, Weight vs. Price_euros, and Screen_Resolution_Width vs. Price_euros.
Explanation and Justification: EDA is crucial for understanding the data and the relationships between features. The correlation matrix helps identify which features are strongly correlated with the target variable (Price_euros). Scatter plots provide visual confirmation of these relationships and help in understanding the nature of the correlations (e.g., linear, non-linear).
