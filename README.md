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
Model Building and Evaluation:

Changes: Split the data into training and testing sets using train_test_split. Initialized and trained a LinearRegression model. Made predictions on the test set. Evaluated the model using MAE, MSE, RMSE, and R-squared. Printed the evaluation metrics and an interpretation of R-squared.
Explanation and Justification: This is the core machine learning part of the project. Splitting the data ensures that the model is evaluated on unseen data, providing a realistic measure of its performance. Linear Regression is a suitable model for predicting a continuous target variable like price. Evaluating the model with standard regression metrics quantifies how well the model performs and allows for comparison with other models if necessary.
Model Usage for Prediction:

Changes: To demonstrate how to use the trained model for prediction on new data, a hypothetical new laptop was created as a pandas DataFrame. This new DataFrame was carefully constructed to have the exact same columns as the training data (X_train) and in the same order, with appropriate values for the laptop's specifications and 0s for the features that do not apply. The trained model was then used to predict the price of this single new data point using model.predict(new_laptop_df). The resulting predicted price was printed, formatted to two decimal places.
Explanation and Justification: The ultimate goal of building a predictive model is to use it to make predictions on new data. This step illustrates the practical application of the trained model. It is crucial that the format and features of the new data match those of the training data, including the order of columns and the encoding of categorical variables. This ensures that the model receives input in the expected format and can make accurate predictions.
Streamlit Web Application:

Changes: To create an interactive interface for the model, the streamlit library was installed using !pip install -q streamlit. To make the locally running Streamlit app accessible over the internet from the Colab environment, the localtunnel tool was installed using !npm install -g localtunnel. The code for the Streamlit web application was written into a Python file named app.py using the %%writefile magic command. This script included the necessary imports, data loading and preprocessing steps (replicated from the notebook to make the app self-contained), model training, and the Streamlit UI elements for user input and displaying predictions. Finally, the Streamlit application was run in the background using !streamlit run app.py &>/content/logs.txt &, and a public URL was generated using !npx localtunnel --port 8501.
Explanation and Justification: Deploying the model as a web application makes it user-friendly and accessible to a wider audience who may not be familiar with Python or the Colab environment. Streamlit is a straightforward library for building interactive web applications with Python. Localtunnel provides a convenient way to expose a local server (like the one running the Streamlit app in Colab) to the internet, allowing others to access it via a public URL. Running the app in the background allows the notebook execution to continue. Writing the app code to a file is necessary for Streamlit to run it.
IP Address Check:

Changes: A shell command !curl ipv4.icanhazip.com was included in a code cell.
Explanation and Justification: This command retrieves the public IPv4 address of the Colab instance. While not directly part of the core data analysis or modeling, it can be a useful utility for debugging network connectivity issues when attempting to access services running within the Colab environment, such as the Streamlit application exposed via localtunnel.
Note on Deployment: In this notebook, localtunnel was used to temporarily expose the Streamlit application for testing and demonstration purposes directly from the Colab environment. For a more stable and permanent deployment, the application was deployed through GitHub to Streamlit Cloud. The instructions for deploying via GitHub are provided in the report section.

