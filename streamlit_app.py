#!/usr/bin/env python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import os

# Set the title and description of the Streamlit application.
st.title("Sleek Price 💻")
st.markdown("""
This web application predicts the price of a laptop based on its specifications.
Input the details of the laptop below to get an estimated price in Euros.
""")

def load_data():
    """
    Loads the dataset and performs all necessary preprocessing and model training.
    This function is cached to avoid re-running on every user interaction.
    """
    try:
        # Use a relative path to the dataset, assuming it's in the same directory.
        DATASET_PATH = os.path.join(os.path.dirname(__file__), 'laptop_price.csv')
        df = pd.read_csv(DATASET_PATH, encoding='latin-1')

        # Store original categorical column values for Streamlit selectboxes.
        original_companies = sorted(df['Company'].unique().tolist())
        original_typenames = sorted(df['TypeName'].unique().tolist())

        # Feature Engineering - Preprocessing
        df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)
        df['Inches'] = df['Inches'].astype(float)
        
        # CPU
        def get_cpu_brand(cpu_string):
            if 'Intel' in cpu_string: return 'Intel'
            elif 'AMD' in cpu_string: return 'AMD'
            return 'Other'
        df['Cpu_Brand'] = df['Cpu'].apply(get_cpu_brand)
        
        def get_cpu_type(cpu_string):
            if 'Core i' in cpu_string: return 'Core i'
            elif 'Ryzen' in cpu_string: return 'Ryzen'
            elif 'Celeron' in cpu_string: return 'Celeron'
            elif 'Pentium' in cpu_string: return 'Pentium'
            return 'Other'
        df['Cpu_Type'] = df['Cpu'].apply(get_cpu_type)
        
        # GPU
        def get_gpu_brand(gpu_string):
            if 'Nvidia' in gpu_string: return 'Nvidia'
            elif 'AMD' in gpu_string: return 'AMD'
            elif 'Intel' in gpu_string: return 'Intel'
            return 'Other'
        df['Gpu_Brand'] = df['Gpu'].apply(get_gpu_brand)
        
        # OS
        def get_os_type(os_string):
            if 'Windows' in os_string: return 'Windows'
            elif 'Mac' in os_string: return 'Mac'
            elif 'Linux' in os_string: return 'Linux'
            return 'Other'
        df['OpSys_Type'] = df['OpSys'].apply(get_os_type)
        
        # Screen Resolution
        df['Screen_Resolution_Type'] = df['ScreenResolution'].apply(lambda x: ' '.join(x.split()[:-1]) if 'x' in x else x)
        df[['Screen_Resolution_Width', 'Screen_Resolution_Height']] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)').astype(int)
        df['PPI'] = ((df['Screen_Resolution_Width']**2 + df['Screen_Resolution_Height']**2)**0.5 / df['Inches']).astype(float)
        df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False, na=False).astype(int)

        # Storage
        def parse_storage(storage_string):
            ssd_gb, hdd_gb = 0, 0
            storage_types = storage_string.split('+')
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

        # Drop original columns
        df = df.drop(columns=['ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys', 'Product', 'laptop_ID', 'Screen_Resolution_Width', 'Screen_Resolution_Height'], errors='ignore')
        
        # One-hot encoding
        categorical_cols = ['Company', 'TypeName', 'Screen_Resolution_Type', 'Cpu_Brand', 'Cpu_Type', 'Gpu_Brand', 'OpSys_Type']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

        # Handle 'Flash_Storage_GB' if it exists. Note: Your original code had 'Flash_Storage_GB', but the parser didn't set it. I've simplified it here.
        if 'Flash_Storage_GB' in df.columns:
            df = df.drop(columns=['Flash_Storage_GB'])

        # Separate features (X) and target variable (y)
        X = df.drop('Price_euros', axis=1)
        y = df['Price_euros']

        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get unique values for dropdowns after preprocessing
        original_cpu_brands = sorted(df['Cpu_Brand_Intel'].unique().tolist() + df['Cpu_Brand_Other'].unique().tolist() + ['AMD']) # Example of getting back original values
        original_cpu_types = sorted(df['Cpu_Type_Core i'].unique().tolist() + df['Cpu_Type_Other'].unique().tolist() + ['Ryzen', 'Celeron', 'Pentium'])

        return model, X.columns.tolist(), {
            'Company': sorted(df['Company_Dell'].unique().tolist() + ['Apple', 'Dell', 'HP', 'MSI', 'Acer']),
            'TypeName': sorted(df['TypeName_Gaming'].unique().tolist() + ['Gaming', 'Ultrabook', 'Notebook', 'Workstation']),
            'ScreenResolutionType': sorted(df.filter(like='Screen_Resolution_Type_').columns.str.replace('Screen_Resolution_Type_', '').tolist()),
            'CpuBrand': sorted(df.filter(like='Cpu_Brand_').columns.str.replace('Cpu_Brand_', '').tolist() + ['Intel', 'AMD']),
            'CpuType': sorted(df.filter(like='Cpu_Type_').columns.str.replace('Cpu_Type_', '').tolist() + ['Core i', 'Ryzen']),
            'GpuBrand': sorted(df.filter(like='Gpu_Brand_').columns.str.replace('Gpu_Brand_', '').tolist() + ['Nvidia', 'AMD', 'Intel']),
            'OpSysType': sorted(df.filter(like='OpSys_Type_').columns.str.replace('OpSys_Type_', '').tolist() + ['Windows', 'Mac', 'Linux'])
        }

    except FileNotFoundError:
        st.error(f"Error: Dataset not found. Please ensure 'laptop_price.csv' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()
        
# Load data once and cache the result
st.cache_data
model, trained_columns, original_values = load_data()

# --- Streamlit UI for User Input ---
st.header("Enter Laptop Specifications")

# Input fields for numerical features
inches = st.number_input("Screen Size (Inches)", min_value=5.0, max_value=30.0, value=15.6, step=0.1)
ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (kg)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

# Storage Input
storage_amount = st.selectbox("Storage Amount (GB)", [128, 256, 512, 1024, 2048])
storage_type = st.selectbox("Storage Type", ['SSD', 'HDD'])

# Screen Resolution
screen_resolution = st.text_input("Screen Resolution (e.g., 1920x1080)", value="1920x1080")
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

# Categorical features
company = st.selectbox("Company", sorted(original_values['Company']))
typename = st.selectbox("Type Name", sorted(original_values['TypeName']))
cpu_brand = st.selectbox("CPU Brand", sorted(original_values['CpuBrand']))
cpu_type = st.selectbox("CPU Type", sorted(original_values['CpuType']))
gpu_brand = st.selectbox("GPU Brand", sorted(original_values['GpuBrand']))
opsys_type = st.selectbox("Operating System Type", sorted(original_values['OpSysType']))


# Predict button
if st.button("Predict Price"):
    # Create the input dictionary for the model
    new_laptop_data = dict.fromkeys(trained_columns, 0)
    
    # Fill in numerical features
    new_laptop_data['Inches'] = inches
    new_laptop_data['Ram'] = int(ram)
    new_laptop_data['Weight'] = weight
    
    # Storage
    if storage_type == 'SSD':
        new_laptop_data['SSD_GB'] = int(storage_amount)
    elif storage_type == 'HDD':
        new_laptop_data['HDD_GB'] = int(storage_amount)
    
    # Screen Resolution and Touchscreen
    match = re.search(r'(\d+)x(\d+)', screen_resolution)
    if match:
        screen_width = int(match.group(1))
        screen_height = int(match.group(2))
        if inches > 0:
            ppi = ((screen_width**2 + screen_height**2)**0.5 / inches)
            new_laptop_data['PPI'] = ppi
    
    new_laptop_data['Touchscreen'] = 1 if touchscreen == "Yes" else 0
    
    # Categorical features (One-Hot Encoded)
    def set_one_hot_column(data, prefix, value):
        col_name = f'{prefix}_{value}'
        if col_name in data:
            data[col_name] = 1

    set_one_hot_column(new_laptop_data, 'Company', company)
    set_one_hot_column(new_laptop_data, 'TypeName', typename)
    set_one_hot_column(new_laptop_data, 'Cpu_Brand', cpu_brand)
    set_one_hot_column(new_laptop_data, 'Cpu_Type', cpu_type)
    set_one_hot_column(new_laptop_data, 'Gpu_Brand', gpu_brand)
    set_one_hot_column(new_laptop_data, 'OpSys_Type', opsys_type)
    
    # Create DataFrame and predict
    new_laptop_df = pd.DataFrame([new_laptop_data], columns=trained_columns)
    predicted_price = model.predict(new_laptop_df)

    # Display the result
    st.subheader("Predicted Price")
    st.write(f"The estimated price of this laptop is: **{predicted_price[0]:.2f} €**")

st.markdown("""
---
**Disclaimer:** This prediction is based on a linear regression model trained on the provided dataset. The actual price may vary due to various factors not included in this model, such as market fluctuations, specific configurations, and retailer pricing.
""")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import os

# Set the title and description of the Streamlit application.
st.title("Sleek Price 💻")
st.markdown("""
This web application predicts the price of a laptop based on its specifications.
Input the details of the laptop below to get an estimated price in Euros.
""")

def load_data():
    """
    Loads the dataset and performs all necessary preprocessing and model training.
    This function is cached to avoid re-running on every user interaction.
    """
    try:
        # Use a relative path to the dataset, assuming it's in the same directory.
        DATASET_PATH = os.path.join(os.path.dirname(__file__), 'laptop_price.csv')
        df = pd.read_csv(DATASET_PATH, encoding='latin-1')

        # Store original categorical column values for Streamlit selectboxes.
        original_companies = sorted(df['Company'].unique().tolist())
        original_typenames = sorted(df['TypeName'].unique().tolist())

        # Feature Engineering - Preprocessing
        df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)
        df['Inches'] = df['Inches'].astype(float)
        
        # CPU
        def get_cpu_brand(cpu_string):
            if 'Intel' in cpu_string: return 'Intel'
            elif 'AMD' in cpu_string: return 'AMD'
            return 'Other'
        df['Cpu_Brand'] = df['Cpu'].apply(get_cpu_brand)
        
        def get_cpu_type(cpu_string):
            if 'Core i' in cpu_string: return 'Core i'
            elif 'Ryzen' in cpu_string: return 'Ryzen'
            elif 'Celeron' in cpu_string: return 'Celeron'
            elif 'Pentium' in cpu_string: return 'Pentium'
            return 'Other'
        df['Cpu_Type'] = df['Cpu'].apply(get_cpu_type)
        
        # GPU
        def get_gpu_brand(gpu_string):
            if 'Nvidia' in gpu_string: return 'Nvidia'
            elif 'AMD' in gpu_string: return 'AMD'
            elif 'Intel' in gpu_string: return 'Intel'
            return 'Other'
        df['Gpu_Brand'] = df['Gpu'].apply(get_gpu_brand)
        
        # OS
        def get_os_type(os_string):
            if 'Windows' in os_string: return 'Windows'
            elif 'Mac' in os_string: return 'Mac'
            elif 'Linux' in os_string: return 'Linux'
            return 'Other'
        df['OpSys_Type'] = df['OpSys'].apply(get_os_type)
        
        # Screen Resolution
        df['Screen_Resolution_Type'] = df['ScreenResolution'].apply(lambda x: ' '.join(x.split()[:-1]) if 'x' in x else x)
        df[['Screen_Resolution_Width', 'Screen_Resolution_Height']] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)').astype(int)
        df['PPI'] = ((df['Screen_Resolution_Width']**2 + df['Screen_Resolution_Height']**2)**0.5 / df['Inches']).astype(float)
        df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False, na=False).astype(int)

        # Storage
        def parse_storage(storage_string):
            ssd_gb, hdd_gb = 0, 0
            storage_types = storage_string.split('+')
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

        # Drop original columns
        df = df.drop(columns=['ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys', 'Product', 'laptop_ID', 'Screen_Resolution_Width', 'Screen_Resolution_Height'], errors='ignore')
        
        # One-hot encoding
        categorical_cols = ['Company', 'TypeName', 'Screen_Resolution_Type', 'Cpu_Brand', 'Cpu_Type', 'Gpu_Brand', 'OpSys_Type']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

        # Handle 'Flash_Storage_GB' if it exists. Note: Your original code had 'Flash_Storage_GB', but the parser didn't set it. I've simplified it here.
        if 'Flash_Storage_GB' in df.columns:
            df = df.drop(columns=['Flash_Storage_GB'])

        # Separate features (X) and target variable (y)
        X = df.drop('Price_euros', axis=1)
        y = df['Price_euros']

        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get unique values for dropdowns after preprocessing
        original_cpu_brands = sorted(df['Cpu_Brand_Intel'].unique().tolist() + df['Cpu_Brand_Other'].unique().tolist() + ['AMD']) # Example of getting back original values
        original_cpu_types = sorted(df['Cpu_Type_Core i'].unique().tolist() + df['Cpu_Type_Other'].unique().tolist() + ['Ryzen', 'Celeron', 'Pentium'])

        return model, X.columns.tolist(), {
            'Company': sorted(df['Company_Dell'].unique().tolist() + ['Apple', 'Dell', 'HP', 'MSI', 'Acer']),
            'TypeName': sorted(df['TypeName_Gaming'].unique().tolist() + ['Gaming', 'Ultrabook', 'Notebook', 'Workstation']),
            'ScreenResolutionType': sorted(df.filter(like='Screen_Resolution_Type_').columns.str.replace('Screen_Resolution_Type_', '').tolist()),
            'CpuBrand': sorted(df.filter(like='Cpu_Brand_').columns.str.replace('Cpu_Brand_', '').tolist() + ['Intel', 'AMD']),
            'CpuType': sorted(df.filter(like='Cpu_Type_').columns.str.replace('Cpu_Type_', '').tolist() + ['Core i', 'Ryzen']),
            'GpuBrand': sorted(df.filter(like='Gpu_Brand_').columns.str.replace('Gpu_Brand_', '').tolist() + ['Nvidia', 'AMD', 'Intel']),
            'OpSysType': sorted(df.filter(like='OpSys_Type_').columns.str.replace('OpSys_Type_', '').tolist() + ['Windows', 'Mac', 'Linux'])
        }

    except FileNotFoundError:
        st.error(f"Error: Dataset not found. Please ensure 'laptop_price.csv' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()
        
# Load data once and cache the result
st.cache_data
model, trained_columns, original_values = load_data()

# --- Streamlit UI for User Input ---
st.header("Enter Laptop Specifications")

# Input fields for numerical features
inches = st.number_input("Screen Size (Inches)", min_value=5.0, max_value=30.0, value=15.6, step=0.1)
ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (kg)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

# Storage Input
storage_amount = st.selectbox("Storage Amount (GB)", [128, 256, 512, 1024, 2048])
storage_type = st.selectbox("Storage Type", ['SSD', 'HDD'])

# Screen Resolution
screen_resolution = st.text_input("Screen Resolution (e.g., 1920x1080)", value="1920x1080")
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

# Categorical features
company = st.selectbox("Company", sorted(original_values['Company']))
typename = st.selectbox("Type Name", sorted(original_values['TypeName']))
cpu_brand = st.selectbox("CPU Brand", sorted(original_values['CpuBrand']))
cpu_type = st.selectbox("CPU Type", sorted(original_values['CpuType']))
gpu_brand = st.selectbox("GPU Brand", sorted(original_values['GpuBrand']))
opsys_type = st.selectbox("Operating System Type", sorted(original_values['OpSysType']))


# Predict button
if st.button("Predict Price"):
    # Create the input dictionary for the model
    new_laptop_data = dict.fromkeys(trained_columns, 0)
    
    # Fill in numerical features
    new_laptop_data['Inches'] = inches
    new_laptop_data['Ram'] = int(ram)
    new_laptop_data['Weight'] = weight
    
    # Storage
    if storage_type == 'SSD':
        new_laptop_data['SSD_GB'] = int(storage_amount)
    elif storage_type == 'HDD':
        new_laptop_data['HDD_GB'] = int(storage_amount)
    
    # Screen Resolution and Touchscreen
    match = re.search(r'(\d+)x(\d+)', screen_resolution)
    if match:
        screen_width = int(match.group(1))
        screen_height = int(match.group(2))
        if inches > 0:
            ppi = ((screen_width**2 + screen_height**2)**0.5 / inches)
            new_laptop_data['PPI'] = ppi
    
    new_laptop_data['Touchscreen'] = 1 if touchscreen == "Yes" else 0
    
    # Categorical features (One-Hot Encoded)
    def set_one_hot_column(data, prefix, value):
        col_name = f'{prefix}_{value}'
        if col_name in data:
            data[col_name] = 1

    set_one_hot_column(new_laptop_data, 'Company', company)
    set_one_hot_column(new_laptop_data, 'TypeName', typename)
    set_one_hot_column(new_laptop_data, 'Cpu_Brand', cpu_brand)
    set_one_hot_column(new_laptop_data, 'Cpu_Type', cpu_type)
    set_one_hot_column(new_laptop_data, 'Gpu_Brand', gpu_brand)
    set_one_hot_column(new_laptop_data, 'OpSys_Type', opsys_type)
    
    # Create DataFrame and predict
    new_laptop_df = pd.DataFrame([new_laptop_data], columns=trained_columns)
    predicted_price = model.predict(new_laptop_df)

    # Display the result
    st.subheader("Predicted Price")
    st.write(f"The estimated price of this laptop is: **{predicted_price[0]:.2f} €**")

st.markdown("""
---
**Disclaimer:** This prediction is based on a linear regression model trained on the provided dataset. The actual price may vary due to various factors not included in this model, such as market fluctuations, specific configurations, and retailer pricing.
""")
