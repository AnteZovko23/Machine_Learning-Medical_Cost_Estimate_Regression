from keras_tuner.engine import hypermodel
from tensorflow import keras
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model

from Model import process_data

##### Overhead ####################





def getMAE(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)



def getMSE(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)



def plot_predictions(train_data=[], train_labels=[], test_data=[],test_labels=[], predictions=[]):
    plt.figure(figsize=(10,7))
    # plt.scatter(train_data, train_labels, c="b", label="train data")
    plt.scatter(np.arange(len(test_labels)), test_labels, c="g", label="Test Data")
    
    plt.figure(figsize=(10,8))
    
    plt.scatter(np.arange(len(test_labels)), test_labels, c="g", label="Test Data")

    plt.scatter(np.arange(len(predictions)), predictions, c ="r", label="Predictions")
    plt.legend()
    plt.show()

def plotModel(model):
    plot_model(model=model, show_shapes=True)



def load_model():
    return keras.models.load_model('my_model')

if __name__ == '__main__':

    hypermodel = load_model()

    dataframe = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
    
    X_train_normal, X_test_normal, y_train, y_test = process_data(dataframe)

    predictions = hypermodel.predict(X_test_normal)
    
    plot_predictions(test_labels=y_test, predictions=predictions)

    # model, history = create_model2()

    # pred_1 = hypermodel.predict(X_test_normal)
    # plot_predictions(test_labels=y_test, predictions=pred_1)
    # plt.show()
