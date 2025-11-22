# data_cleanup.py
# this file contains functions for cleaning up and preprocessing data

import pandas as pd
import numpy as np

def load_data(file_path) -> pd.DataFrame:
    """Load data from a XLSX file."""
    return pd.read_excel(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    # Row 0 contains headers, so set it as the header
    df.columns = df.iloc[0]
    # Drop the first row as it is now the header
    df = df.iloc[1:]
    # Drop rows with any missing values
    df = df.dropna()
    # Reset index
    df = df.reset_index(drop=True)
    # Drop irrelevant columns (The Region is always USA)
    df = df.drop(columns=['Region'])

    # Renaming columns for ease of use
    # First strip newline characters from column names, replace with spaces
    df.columns = df.columns.str.replace('\n', ' ')
    # Now rename columns to shorter names
    df = df.rename(columns={
        'Not Seasonally-Adjusted Purchase-Only Index  (1991Q1=100)': 'Index',
        'Seasonally-Adjusted Purchase-Only Index  (1991Q1=100)': 'Adjusted Index',
        'Not Seasonally-Adjusted Purchase-Only Index % Change Over  Previous Quarter': 'Index Change Over Previous Quarter',
        'Seasonally-Adjusted Purchase-Only Index % Change Over  Previous Quarter': 'Adjusted Index Change Over Previous Quarter',
        'Not Seasonally-Adjusted Purchase-Only Index % Change Over  Previous 4 Quarters': 'Index Change Over Previous Year',
        'Seasonally-Adjusted Purchase-Only Index % Change Over  Previous 4 Quarters': 'Adjusted Index Change Over Previous Year'
    })
    # Convert all columns to numeric type
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def main(): 
    # Load initial dataset
    df = load_data('data/raw/hpi_po_summary.xlsx')
    # Display column names of the initial dataset
    print("Initial dataset columns:")
    print(df.columns)
    # Display first few rows of the initial dataset
    print("\nFirst few rows of the initial dataset:")
    print(df.head())
    # Clean up initial dataset
    cleaned_df = clean_data(df)
    # Display cleaned dataset columns
    print("\nCleaned dataset columns:")
    print(cleaned_df.columns)
    # Display first few rows of the cleaned dataset
    print("\nFirst few rows of the cleaned dataset:")
    print(cleaned_df.head())
    # Check if value types are numeric where expected
    print("\nData types of cleaned dataset columns:")
    for col in cleaned_df.columns:
        print(f"{col}: {cleaned_df[col].dtype}")
    # Display size of cleaned dataset
    print(f"\nCleaned dataset size: {cleaned_df.shape}")
    # # Save cleaned dataset to a new file
    # cleaned_df.to_excel('data/clean/hpi_po_summary_cleaned.xlsx', index=False)

   

if __name__ == "__main__":
    main()