from preProcessing import sanity_check, handle_missing_value, generate_additional_features, filter_predictor_columns
import pickle
from loadData import load_data , test_path
import joblib

import warnings
warnings.filterwarnings("ignore")



def encode_predict_input(df,encoded_dict):
      '''
      This function encodes categorical values with same values as training encoded values.
      Input:
          df : DataFrame
          encoded_dict : Category encoded dictionary
      returns :None
      '''
      encoded_cols = ['Airline', 'Source', 'Destination', 'Route']
      frequency_dict = encoded_dict['Frequency']
      mean_dict = encoded_dict['Mean']
      label_dict = encoded_dict['Label']
      for col in encoded_cols:
          df["Freq_encoded_"+col] = df[col].replace(frequency_dict[col])
          df["Mean_encoded_"+col] = df[col].replace(mean_dict[col])
          df["Label_encoded_"+col] = df[col].replace(label_dict[col])


def preprocess_and_predict(df,encoded_dict):
    '''
      This function takes in new dataframe or row of observation and generate all features
    Input :
        df : DataFrame or row of observation
        encoded_dict : Dictonary created while training for Categorical Encoded Value.
    '''
    sanity_check(df,mode='predict')
    handle_missing_value(df,mode='predict')
    
    encode_predict_input(df,encoded_dict)
    generate_additional_features(df)
    X = filter_predictor_columns(df)
    return X

if __name__ == "__main__":


    print("Loading the TeatData.")
    # Load data (deserialize)
    with open('models/encoded.pickle', 'rb') as handle:
        encoded_dict = pickle.load(handle)

    print(encoded_dict)
    
    model_path = "models/randomForestModel.sav"
    saved_model= joblib.load(model_path)

    test_df = load_data(test_path)
    test_input = preprocess_and_predict(test_df,encoded_dict)
    print(test_input.head())
    saved_model.predict(test_input.iloc[0:5,:])
    print(saved_model.predict(test_input.iloc[0:5,:]))