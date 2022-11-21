import pandas as pd
import numpy as np
## Applying Label Encoder to Categorical columns as Hit and Trail.
from sklearn.preprocessing import LabelEncoder

# import load module from load.py
from loadData import load_data , path


# Duration converted to Minutes.
def to_minutes(x):
    if len(x.split(" ")) > 1:
        hour_value = int(x.split(" ")[0].replace("h",""))
        minute_value = int(x.split(" ")[1].replace("m",""))
    else:
        if x.endswith("h"):
            hour_value = int(x.replace("h",""))
            minute_value = 0
        else:
            hour_value = 0
            minute_value = int(x.replace("m",""))
    duration = hour_value*60 + minute_value
    return duration

# Total_Stops column converted to integer.
stop_map = {
    'non-stop': 0 ,
    '2 stops' : 2 ,
    '1 stop'  : 1 ,
    '3 stops' : 3 ,
    '4 stops' : 4
}

## Buildng a sanity_check function
def sanity_check(df,mode='train'):
    '''
      This function perform sanity and check create a dataframe.
      Input:
        df : Dataframe which require sanity-check
        mode : train or predict/inference
      return : None
    '''
    if mode == 'train':
        # Drop any duplicaties (check size before and after dropping duplicates.)
        df.drop_duplicates(inplace=True)

    # Date_of_Journey ,Arrival_Time and Dep_Time must be datetime object
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'])

    # Duration converted to Minutes.
    df['Duration'] = df['Duration'].apply(lambda x: to_minutes(x))

    # Total_Stops column converted to integer.
    df['Total_Stops'] = df['Total_Stops'].replace(stop_map)
    return df


## Building function to handle missing value
def handle_missing_value(df,mode='train'):
    '''
      This function handles missing value create a dataframe.
      Input:
        df : Dataframe which require missing value treatment
        mode : train or predict/Inference
      returns :
         Dataframe with all missing value handled.
    '''
    if mode == 'train':
        # Seems to be the same row and make sense  if you don't have route can't decide on stops.
        # It would be appropiate to drop them in this case.
        df.dropna(inplace=True)

    # Dropping Additional_Info columns as around 78% of values are not provided.
    df.drop('Additional_Info',axis=1,inplace=True)  
    return df

def frequency_encoder(df,col):
    """
    This function encodes a categorical column based on the frequency of their occurence.
    input:
        df : Input DataFrame in which encoding has to be created 
        col : Column name which has to be encoded
    return: 
          frequency encoded dictionary for columns
    """
    freq_value = df.groupby(col).size()/len(df)
    freq_dict = freq_value.to_dict()
    df["Freq_encoded_"+col] = df[col].replace(freq_dict)
    return freq_dict

def mean_encoder(df,col,target_col):
    """
    This function encodes a categorical column based on the frequency of their occurence.
    input:
        df : Input DataFrame in which encoding has to be created 
        col : Column name which has to be encoded
    return: 
          Mean encoded dict for column
    """
    mean_value = df.groupby(col)[target_col].mean()
    mean_dict = mean_value.to_dict()
    df["Mean_encoded_"+col] = df[col].replace(mean_dict)
    return mean_dict

## Label encoder for function and later usages:
def label_encoder(df,col):
    """
    This function encodes a categorical column based on the basis of their order label.
    input:
        df : Input DataFrame in which encoding has to be created 
        col : Column name which has to be encoded
    return: 
          label encoded dict for column
    """
    le = LabelEncoder()
    le.fit(df[col])
    label_dict = dict(zip((le.classes_),le.transform(le.classes_)))
    df["Label_encoded_"+col] = df[col].replace(label_dict)
    return label_dict


## Create a function to handle categorical value
def handle_categorical_values(df,target):
    '''
      This function handles categorical value and create a dataframe.
      Input:
        df : Dataframe which require categorical value treatment
      returns :
         Dataframe with all categorical value handled.
    '''
    encoded_dict = {}
    # Getting all object columns
    object_columns = df.select_dtypes(object).columns

    ## generate frequency encoded categorical values
    frequency_encoded_dict ={} 
    for col in object_columns:
        freq_dict = frequency_encoder(df,col)
        frequency_encoded_dict[col] = freq_dict

    ## generate target mean encoded categorical values
    mean_encoded_dict ={} 
    for col in object_columns:
        mean_dict = mean_encoder(df,col,target)
        mean_encoded_dict[col] = mean_dict

    
    ## generate label encoded categorical values
    label_encoded_dict ={} 
    for col in object_columns:
        label_dict = label_encoder(df,col)
        label_encoded_dict[col] = label_dict
    
    encoded_dict["Frequency"] = frequency_encoded_dict
    encoded_dict["Mean"] = mean_encoded_dict
    encoded_dict["Label"] = label_encoded_dict

    return df, encoded_dict


def generate_additional_features(df):
    '''
    This Function generates additional features.
    Input :
        df : DataFrame from which feature has to be genrated
    return None
    '''
    # Time based features can be genrated.
    # Day_of_week
    df['day_of_week'] = df['Date_of_Journey'].dt.day_of_week 
    # Day_of_month
    df['day_of_month'] = df['Date_of_Journey'].dt.day
    # Weekdays 
    df['weekday'] = np.where(df["day_of_week"].isin([5,6]),0,1)
    # Month of Travel
    df['month'] = df['Date_of_Journey'].dt.month
    # Hour of Departure etc.
    df['dep_hour'] = df['Dep_Time'].dt.hour

    return df

def filter_predictor_columns(df):
    '''
    This function filters predictor columns from the incoming Data
    '''
    predictor_columns = ['Duration', 'Total_Stops', 'Label_encoded_Airline',
                            'Label_encoded_Source', 'Label_encoded_Destination',
                            'Label_encoded_Route', 'Freq_encoded_Airline', 'Freq_encoded_Source',
                            'Freq_encoded_Destination', 'Freq_encoded_Route',
                            'Mean_encoded_Airline', 'Mean_encoded_Source',
                            'Mean_encoded_Destination', 'Mean_encoded_Route', 'day_of_week',
                            'day_of_month', 'weekday', 'month', 'dep_hour']
    return df[predictor_columns]

def pre_process(df,target):
    '''
      This function applies pre-processing on any incoming observations
    Input:
      df : DataFrame which require pre-processing
      target : dependent variable
    return clean_df : Cleaned Dataframe
    '''
    sanity_check(df)
    handle_missing_value(df)
    encoded_dict = handle_categorical_values(df,target)
    generate_additional_features(df)
    X = filter_predictor_columns(df)
    y = df[target]
    # robust_transformer = RobustScaler().fit(X)
    # robust_transformer.transform(X)
    return X,y,encoded_dict
    

if __name__ == "__main__": 

    df = load_data(path)
    # print(df.shape)
    # print(df.head())
    # print("-"*72)
    # print("Data Pre-Processing.")
    # print("-"*72)
    target = 'Price'
    # df = sanity_check(df)
    # df = handle_missing_value(df)
    # df, encoded_dict = handle_categorical_values(df,target)
    # df = generate_additional_features(df)
    # print(df.shape)
    # print(df.head())
    X,y,encoded_dictpre_process = (df,target)
    print(X.head())
    print(X.shape,y.shape)