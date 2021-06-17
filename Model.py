from pandas.core.reshape.reshape import get_dummies
from tensorflow import keras
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

#### Overhead ######

##

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

def plot_model(model):
    plot_model(model=model, show_shapes=True)

#####################################

# Get data


def load_data(dataframe): 
    
    
    ## One hot encode data
    dataframe_temp = pd.get_dummies(dataframe)

    ## Get X or features and y or labels
    X = dataframe_temp.drop("charges", axis=1)
    y = dataframe_temp["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def process_data(dataframe):

    ct = make_column_transformer(
        (MinMaxScaler(), ["age", "bmi", "children"]), # All columns between 0 and 1
        (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"]),
    )

    X = dataframe.drop("charges", axis=1)
    y = dataframe["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the transformer on the data
    ct.fit(X_train)

    X_train_normal = ct.fit_transform(X_train)
    X_test_normal = ct.transform(X_test)

    return X_train_normal, X_test_normal, y_train, y_test


def create_model():

    model = tf.keras.Sequential([
        

        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        # tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)

    ])

    model.compile(loss=tf.keras.losses.LogCosh(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mae'])

    return model

#####################################

dataframe = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")


X_train, X_test, y_train, y_test = load_data(dataframe)

# tf.random.set_seed(42)

# model_1 = create_model()

# history = model_1.fit(X_train, y_train, epochs=200) ## Uncomment to train

# pred_1 = model_1.predict(X_test)

# print(model_1.evaluate(X_test, y_test))

# pd.DataFrame(history.history).plot()
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.show()
# plot_predictions(X_train.index, y_train, X_test.index, y_test, pred_1)
## Loss: 2371


######### Preprocess data now ##############


X_train_normal, X_test_normal, y_train, y_test = process_data(dataframe)
print(len(y_test))
tf.random.set_seed(42)


model_2 = create_model()

history2 = model_2.fit(X_train_normal, y_train, epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)])

print(model_2.evaluate(X_test_normal, y_test))


pd.DataFrame(history2.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

l = np.arange(len(y_test))

pred_1 = model_2.predict(X_test_normal)
plot_predictions(test_labels=y_test, predictions=pred_1)
plt.show()
