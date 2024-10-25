def drop_single_value_columns(df):
    """
    Function to drop columns that have the same single value for all rows in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with single-value columns removed.
    int: The number of single-value columns dropped.
    """
    # Get descriptive statistics of the DataFrame
    desc_stats = df.describe()
    
    # Get the list of columns from desc_stats
    desc_column_list = desc_stats.columns.tolist()
    
    # List to hold columns with a single value
    singleValueColumnList = []

    # Identify columns with a single value
    for col in desc_column_list:
        # Calculate min and max to check for single-value columns
        min_value = df[col].min()
        max_value = df[col].max()

        # If min and max are equal, it's a single-value column
        if min_value == max_value:
            singleValueColumnList.append(col)
    
    print("Number of single-value columns: ", len(singleValueColumnList))
    print(singleValueColumnList)

    # Drop the single-value columns
    df_cleaned = df.drop(columns=singleValueColumnList)
    single_value_columns_dropped = len(singleValueColumnList)
    
    print(f"Number of single-value columns dropped: {single_value_columns_dropped}")
    
    return df_cleaned, single_value_columns_dropped

import pandas as pd

def check_and_drop_all_null_columns(df):
    """
    Function to check if any column in a DataFrame has all null values and drop those columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to check and modify.

    Returns:
    df (pd.DataFrame): The modified DataFrame with columns having all null values dropped.
    null_columns (list): A list of columns that had all null values.
    """
    # Get the list of columns that have all null values
    null_columns = [col for col in df.columns if df[col].isnull().all()]

    if null_columns:
        print(f"These columns have all null values: {null_columns}")
        # Drop columns that have all null values
        df = df.drop(null_columns, axis=1)
        print(type(df))
        print(f"Dropped columns: {null_columns}")
    else:
        print("No columns have all null values.")

    return df, null_columns


import pandas as pd

def drop_ID_columns(df):
    # List to keep track of columns to drop
    columns_to_drop = []
    
    # Iterate through each column in the DataFrame
    for col in df.columns:
        # Check if the column is numeric and contains more than one unique value
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
            # Check if values in the column are incrementing
            is_incrementing = (df[col].diff().dropna() >= 0).all()
            
            # If incrementing, add to columns_to_drop list
            if is_incrementing:
                columns_to_drop.append(col)
    
    # Drop the incrementing columns
    df_dropped = df.drop(columns=columns_to_drop)
    
    # Return the updated DataFrame and the list of dropped columns
    return df_dropped, columns_to_drop

from sklearn.impute import SimpleImputer
import numpy as np
def performImputation(df, numeric_cols, nonnumeric_cols):
    for col in numeric_cols:
        if df[col].isnull().sum()>0:
            df[col] =  SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df[[col]])

    for col in nonnumeric_cols:
        if df[col].isnull().sum()>0:
            df[col] =  SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(df[[col]]).ravel()
    return df

# Label Encode dependent variable value
from sklearn.preprocessing import LabelEncoder
def lableEncoding(df, target):
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    return df

def performDummyEncoding(X):
    colsToBeDummied = X.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_delete=[]
    for col in colsToBeDummied:
        print(col)
        unique_vals = X[col].unique()
        if (len(unique_vals)==2):
            cols_delete.append(unique_vals[0])
        dummy_value=pd.get_dummies(X[col])
        X=pd.concat([X, dummy_value], axis=1)  
        X=X.drop(col, axis=1)

    for col in cols_delete:
        X=X.drop(col, axis=1)        
    return X

from sklearn.preprocessing import StandardScaler, normalize
def performScale_Normalize(X):
    scaler=StandardScaler()
    X = scaler.fit_transform(X)
    # Normalizing the Data 
    X = normalize(X) 
  
    # Converting the numpy array into a pandas DataFrame 
    X = pd.DataFrame(X) 
    
    return X