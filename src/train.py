import pandas as pd
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split

from loadData import load_data, train_path
from preProcessing import pre_process
from model import regression, decisiontree, knearestneighbour, randomForest

import warnings
warnings.filterwarnings("ignore")


def save_model(model,file_name):
    joblib.dump(model,file_name)

def save_pickle(model,file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(file_name):
    model = joblib.load(file_name)
    return model

def train(X,y,modelType):
    # Split your dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = modelType(X_train, X_test, y_train, y_test)
    return model

if __name__ == "__main__":
    
    target = "Price"
    print("Loading the Data.")
    df = load_data(train_path)

    print("Starting Pre-processing of Data")
    X,y,encoded_dict = pre_process(df,target)
    # Store data (serialize)
    with open('models/encoded.pickle', 'wb') as handle:
        pickle.dump(encoded_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Statrted Training the model.")
    regression_model = train(X,y,regression)
    print("Saving the model.")
    file_name = "models/linearRegressionModel.pickle" 
    save_pickle(regression_model,file_name)

    print("Statrted Training the decisiontree model.")
    decisiontree_model = train(X,y,decisiontree)
    print("Saving the model.")
    file_name = "models/decisiontreeModel.pickle" 
    save_pickle(decisiontree_model,file_name)

    print("Statrted Training the knearestneighbour model.")
    knearestneighbour_model = train(X,y,knearestneighbour)
    print("Saving the model.")
    file_name = "models/knearestneighbourModel.pickle" 
    save_pickle(knearestneighbour_model,file_name)

    print("Statrted Training the randomForest model.")
    randomForest_model = train(X,y,randomForest)
    print("Saving the model.")
    file_name = "models/randomForestModel.pickle" 
    save_pickle(randomForest_model,file_name)



