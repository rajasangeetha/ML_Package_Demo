import os
import pandas as pd
import numpy as np

def read_file(directory_path, file_name, sheet_name=0):
    """
    Function to read a file (Excel or CSV) by combining directory path and file name.

    Parameters:
    directory_path (str): The path to the directory where the file is located.
    file_name (str): The name of the file (Excel or CSV).
    sheet_name (str or int, optional): The sheet name or sheet number to read (only used for Excel files). Defaults to the first sheet.

    Returns:
    DataFrame: Pandas DataFrame containing the data from the file.
    """
    # Combine the directory and file name to get the full path
    dataset_path = os.path.join(directory_path, file_name)
    
    # Get the file extension
    file_extension = os.path.splitext(file_name)[1].lower()

    try:
        # Check if the file is a CSV
        if file_extension == '.csv':
            df = pd.read_csv(dataset_path)
            print("CSV file loaded successfully.")
        
        # Check if the file is an Excel file
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(dataset_path, sheet_name=sheet_name)
            print("Excel file loaded successfully.")
        
        else:
            print(f"Unsupported file format: {file_extension}")
            return None

        return df

    except FileNotFoundError:
        print(f"The file at {dataset_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None