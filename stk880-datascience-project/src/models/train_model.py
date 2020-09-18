import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
import pickle
import sys
import logging

sys.path.append('src')
sys.path.append('src/visualization')
from visualization.visualize import *

def compile_model(n_features):
    model=Sequential()
    model.add(Dense(12,input_dim=8,activation="sigmoid"))
    model.add(Dropout(.2))
    model.add(Dense(8,  activation="relu"))
    model.add(Dropout(.1))
    model.add(Dense(1,  activation="sigmoid"))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=["accuracy"])

    return model

def fit_model(model,features,labels,n_epochs=10,n_batch=10,val_split=.1):
    history=model.fit(features,labels,epochs=n_epochs,batch_size=n_batch,validation_split=val_split)
    return history

def main(logging):
    logging.info("#### Compiling model")
    model=compile_model(8)
    logging.info("####### Load data")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    #Train models on data
    logging.info("####### Training models on data")
    history=fit_model(model,X_train,y_train,n_epochs=50,n_batch=30,val_split=.2)
    logging.info("##### plottin loss-plot")
    loss_plot(history)
    logging.info("####### saving model")
    model_path="models/stk-model-v1.h5"
    history.model.save(model_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,filename="stk-cookiecutter.log")
    main(logging)