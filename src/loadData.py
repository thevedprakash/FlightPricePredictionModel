# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_path = "data/Train.xlsx"
test_path = 'data/Test.xlsx'

def load_data(path):
    '''
    This function loads the data from excel file into a dataframe.
    inputs:
        path: Path of data file
    returns:
        df : dataframe
    '''
    df = pd.read_excel(path)
    return df

if __name__ == "__main__":
    path = "data/Train.xlsx"
    # Read the data 
    df = load_data(path)
    print(df.columns)
    print(df['Airline'].unique())