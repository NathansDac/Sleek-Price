#!/usr/bin/env python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import os

# --- 1. Set up the Streamlit page with custom styling ---
# Set the title and description of the Streamlit application.
st.set_page_config(layout="wide")
st.title("Sleek Price 💻")
st.markdown("""
This web application predicts the price of a laptop based on its specifications.
To function correctly, the 'laptop_price.csv' must be in the same GitHub repository directory as this script.
""")

# Custom CSS for the navy blue theme and improved UI elements
st.markdown("""
<style>
body {
    background-color: #000080; /* Navy blue background */
    color: #F0F2F6; /* Light text color for contrast */
    font-family: sans-serif;
}
.stApp {
    background-color: #000080; /* Ensure the main app container has navy blue background */
    color: #F0F2F6;
}
.st-cy, .st-d1, .st-ch { /* Unify styling for input labels, number input, and selectbox */
    color: #ADD8E6; /* Light blue for labels and text */
}
.st-d1, .st-ch {
    background-color: #4682B4; /* Steel blue background for input fields */
    border-radius: 8px;
    padding: 10px;
    color: #F0F2F6; /* Light text for input fields */
}
.st-h1, .st-h2, .st-h3, .st-h4, .st-h5, .st-h6 { /* Headers */
    color: #B0C4DE; /* Light steel blue for headers */
}
.stButton > button {
    background-color: #0000CD; /* Medium blue button background */
    color: white;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}
.stButton > button:hover {
    background-color: #1E90FF; /* Dodger blue on hover */
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- 2. Data Loading and Model Training Function ---
@st.cache_data
def load_data_and_train_model():
    """
    Loads the dataset and performs all necessary preprocessing and model training.
    This function is cached to avoid re-running on every user interaction.
    """
    try:
        # Use a direct relative path for Streamlit Cloud deployment.
        DATASET_PATH = 'laptop_price.csv'
        df = pd.read_csv(DATASET_PATH, encoding='latin-1')

        # --- FIX: Robust data type conversion with error handling ---
        df['Weight'] = pd.to_numeric(df['Weight'].str.replace('kg', '', regex=False), errors='coerce')
        df['Ram'] = pd.to_numeric(df['Ram'].str.replace('GB', '', regex=False), errors='coerce')
        df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')

        # Fill any NaN values that resulted from the above conversion to prevent errors
        df.fillna(0, inplace=True)

        # --- Preprocessing and Feature Engineering (simplified for robustness) ---
        def get_cpu_brand(cpu_string):
            if 'Intel' in str(cpu_string): return 'Intel'
            elif 'AMD' in str(cpu_string): return 'AMD'
            return 'Other'
        df['Cpu_Brand'] = df['Cpu'].apply(get_cpu_brand)

        def get_cpu_type(cpu_string):
            if 'Core i' in str(cpu_string): return 'Core i'
            elif 'Ryzen' in str(cpu_string): return 'Ryzen'
            elif 'Celeron' in str(cpu_string): return 'Celeron'
            elif 'Pentium' in str(cpu_string): return 'Pentium'
            return 'Other'
        df['Cpu_Type'] = df['Cpu'].apply(get_cpu_type)

        def get_gpu_brand(gpu_string):
            if 'Nvidia' in str(gpu_string): return 'Nvidia'
            elif 'AMD' in str(gpu_string): return 'AMD'
            elif 'Intel' in str(gpu_string): return 'Intel'
            return 'Other'
        df['Gpu_Brand'] = df['Gpu'].apply(get_gpu_brand)

        def get_os_type(os_string):
            if 'Windows' in str(os_string): return 'Windows'
            elif 'Mac' in str(os_string): return 'Mac'
            elif 'Linux' in str(os_string): return 'Linux'
            return 'Other'
        df['OpSys_Type'] = df['OpSys'].apply(get_os_type)

        def parse_storage(storage_string):
            ssd_gb, hdd_gb = 0, 0
            storage_types = str(storage_string).split('+')
            for item in storage_types:
                match = re.search(r'(\d+)\s*(GB|TB)', item)
                if match:
                    size = int(match.group(1))
                    unit = match.group(2)
                    size_gb = size * 1024 if unit == 'TB' else size
                    if 'SSD' in item: ssd_gb = size_gb
                    elif 'HDD' in item: hdd_gb = size_gb
            return pd.Series([ssd_gb, hdd_gb])
        df[['SSD_GB', 'HDD_GB']] = df['Memory'].apply(parse_storage)

        # Handle PPI and Touchscreen
        df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False, na=False).astype(int)
        df['Screen_Resolution_Width'] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')[0].fillna(0).astype(int)
        df['Screen_Resolution_Height'] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')[1].fillna(0).astype(int)
        df['PPI'] = ((df['Screen_Resolution_Width']**2 + df['Screen_Resolution_Height']**2)**0.5 / df['Inches']).fillna(0)

        # Drop original columns and prepare for one-hot encoding
        df_processed = df.drop(columns=['ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys', 'Product', 'laptop_ID'], errors='ignore')
        
        # Perform one-hot encoding on categorical columns
        categorical_cols = ['Company', 'TypeName', 'Cpu_Brand', 'Cpu_Type', 'Gpu_Brand', 'OpSys_Type']
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, dtype=int)

        # Get the final list of columns for the trained model
        trained_columns = df_processed.drop('Price_euros', axis=1).columns.tolist()

        # Get unique values for dropdowns after preprocessing
        original_values = {}
        for col in ['Company', 'TypeName', 'Cpu_Brand', 'Cpu_Type', 'Gpu_Brand', 'OpSys_Type']:
            original_values[col] = sorted(df[col].unique())
        
        # Separate features (X) and target variable (y)
        X = df_processed.drop('Price_euros', axis=1)
        y = df_processed['Price_euros']

        # Train the model
        model = LinearRegression()
        model.fit(X, y)

        return model, trained_columns, original_values
    except FileNotFoundError:
        st.error("Error: Dataset not found. Please ensure 'laptop_price.csv' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()

# Load data and train the model. This is done only once due to caching.
model, trained_columns, original_values = load_data_and_train_model()

# --- 3. Streamlit UI for User Input ---
st.header("Enter Laptop Specifications")


[Image of a person typing on a laptop]


# Input fields for numerical features
cols1 = st.columns(3)
with cols1[0]:
    inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
with cols1[1]:
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
with cols1[2]:
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

# Storage Input
cols2 = st.columns(2)
with cols2[0]:
    storage_type = st.selectbox("Storage Type", ['SSD', 'HDD'])
with cols2[1]:
    storage_amount = st.selectbox("Storage Amount (GB)", [128, 256, 512, 1024, 2048])

# Touchscreen
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

# Categorical features
cols3 = st.columns(2)
with cols3[0]:
    company = st.selectbox("Company", original_values['Company'])
with cols3[1]:
    typename = st.selectbox("Type Name", original_values['TypeName'])

cols4 = st.columns(2)
with cols4[0]:
    cpu_brand = st.selectbox("CPU Brand", original_values['Cpu_Brand'])
with cols4[1]:
    cpu_type = st.selectbox("CPU Type", original_values['Cpu_Type'])

cols5 = st.columns(2)
with cols5[0]:
    gpu_brand = st.selectbox("GPU Brand", original_values['Gpu_Brand'])
with cols5[1]:
    opsys_type = st.selectbox("Operating System Type", original_values['OpSys_Type'])

# Predict button
if st.button("Predict Price"):
    # Create the input dictionary for the model, initializing with zeros
    new_laptop_data = dict.fromkeys(trained_columns, 0)
    
    # Fill in numerical features
    new_laptop_data['Inches'] = float(inches)
    new_laptop_data['Ram'] = int(ram)
    new_laptop_data['Weight'] = float(weight)
    
    # Storage
    if storage_type == 'SSD':
        new_laptop_data['SSD_GB'] = int(storage_amount)
    elif storage_type == 'HDD':
        new_laptop_data['HDD_GB'] = int(storage_amount)
    
    # Touchscreen
    new_laptop_data['Touchscreen'] = 1 if touchscreen == "Yes" else 0
    
    # Assume a standard resolution for PPI calculation to avoid errors from user text input
    screen_width, screen_height = 1920, 1080
    if inches > 0:
        new_laptop_data['PPI'] = ((screen_width**2 + screen_height**2)**0.5 / inches)
    
    # Categorical features (One-Hot Encoded)
    # The columns for one-hot encoding are generated dynamically
    categorical_inputs = {
        'Company': company,
        'TypeName': typename,
        'Cpu_Brand': cpu_brand,
        'Cpu_Type': cpu_type,
        'Gpu_Brand': gpu_brand,
        'OpSys_Type': opsys_type
    }
    
    for prefix, value in categorical_inputs.items():
        # Check if the generated column name exists in the trained columns list
        col_name = f'{prefix}_{value}'
        if col_name in trained_columns:
            new_laptop_data[col_name] = 1

    # Create DataFrame and predict
    new_laptop_df = pd.DataFrame([new_laptop_data])
    predicted_price = model.predict(new_laptop_df)

    # Display the result
    st.subheader("Predicted Price")
    st.success(f"The estimated price of this laptop is: **{predicted_price[0]:.2f} €**")

st.markdown("""
---
**Disclaimer:** This prediction is based on a linear regression model trained on the provided dataset. The actual price may vary due to various factors not included in this model, such as market fluctuations, specific configurations, and retailer pricing.
""")
       
