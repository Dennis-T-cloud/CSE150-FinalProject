# data_cleanup.py
# this file contains functions for cleaning up and preprocessing data

import pandas as pd
import numpy as np

def load_data(file_path) -> pd.DataFrame:
    """Load data from a XLSX file."""
    return pd.read_excel(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up the dataset by handling missing values and duplicates."""
    # Drop duplicate rows
    df = df.drop_duplicates()
    # Fill missing values with the mean of the column
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].mean(), inplace=True)
    # Fill missing values in categorical columns with the mode
    for column in df.select_dtypes(include=[object]).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def main(): 
    # Load dataset
    df = load_data('data/raw/hpi_po_summary.xlsx')
    # Display initial data info
    print("Initial Data Info:")
    print(df.info())
    # Display first few rows of the dataset
    print("\nFirst few rows of the dataset:")
    print(df.head())
    # Clean up dataset
    cleaned_df = clean_data(df)
    # Display cleaned data info
    print("\nCleaned Data Info:")
    print(cleaned_df.info())
    # Display first few rows of the cleaned dataset
    print("\nFirst few rows of the cleaned dataset:")
    print(cleaned_df.head())
    # # Save cleaned dataset to a new file
    # cleaned_df.to_excel('data/clean/hpi_po_summary_cleaned.xlsx', index=False)

   

if __name__ == "__main__":
    main()