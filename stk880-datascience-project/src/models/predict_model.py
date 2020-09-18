import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import sys
import logging

sys.path.append('src')
sys.path.append('src/visualization')
from visualization.visualize import *

def  main(logging):
    logging.info("####### Loading Data")
    X_test=pd.read_csv('data/processed/X_test.csv')
    y_test=pd.read_csv('data/processed/y_test.csv')
    logging.info("####### Loading Model")
    model=load_model('models/stk-model-v1.h5')
    logging.info("#### Evaluete Model")
    _,accuracy=model.evaluate(X_test,y_test,verbose=0)
    logging.info("######## Model accuracy {}".format(accuracy))
    y_pred=model.predict_classes(X_test)
    conf_mat=confusion_matrix(y_test,y_pred)
    logging.info("####### Confusion matrix {}".format(conf_mat))
    plot_confusion_matrix(conf_mat,target_names=['0','1'],normalize=True,filepath="reports/figures/confusion_matrix.png")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,filename="stk-cookiecutter-project.log")
    main(logging)