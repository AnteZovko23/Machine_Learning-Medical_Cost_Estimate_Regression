from tensorflow import keras
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


import kerastuner as kt

#### Overhead ######

##



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


def create_model2():

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(32 ,activation='relu'))
    model.add(tf.keras.layers.Dense(992 ,activation='relu'))
    model.add(tf.keras.layers.Dense(1 ,activation='linear'))

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
    metrics=['mean_absolute_error'])


    return model

def create_model(hp):

    
    # Tune num or layers, neurons and the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001

    model = keras.Sequential()

    ## Optimal Hidden layers 2-20
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(tf.keras.layers.Dense(units= hp.Int('units_' + str(i), min_value=32, max_value=1000, step=32), activation='relu'))

    ## Output layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), 
    metrics=['mean_absolute_error'])

    return model


def save_model(model):
    model.save('my_model')


    #####################################

if __name__ == '__main__':

    dataframe = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")



    ######### Preprocess data now ##############

    
    X_train_normal, X_test_normal, y_train, y_test = process_data(dataframe)


    tf.random.set_seed(42)



    ## Stop early if loss doesn't go down
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    ####
    # tuner = kt.BayesianOptimization(create_model,
    #                     objective='val_mean_absolute_error',
    #                     max_trials=50,
    #                     directory='my_dir',
    #                     project_name='intro_to_kt',
    #                     overwrite=True)

    # print(tuner.search_space_summary())

    # tuner.search(X_train_normal, y_train, epochs=100, validation_split=0.2, callbacks=[callback])
    # best_hps=tuner.get_best_hyperparameters(num_trials=50)[0]

    # print(tuner.results_summary())

    # # print(f"""
    # # The hyperparameter search is complete. The optimal number of units in the first densely-connected
    # # layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    # # is {best_hps.get('learning_rate')}.
    # # """)



    # # Build the model with the optimal hyperparameters and train it on the data for 100 epochs
    # model = tuner.hypermodel.build(best_hps)
    # history = model.fit(X_train_normal, y_train, epochs=100, validation_split=0.2)

    # loss_per_epoch = history.history['val_mean_absolute_error']
    # best_epoch = loss_per_epoch.index(max(loss_per_epoch)) + 1
    # print('Best epoch: %d' % (best_epoch,))


    # hypermodel = tuner.hypermodel.build(best_hps)

    # # Retrain the model
    # hypermodel.fit(X_train_normal, y_train, epochs=best_epoch, validation_split=0.2)

    # save_model(hypermodel)
    ####

    model2 = create_model2()

    history = model2.fit(X_train_normal, y_train, epochs=100, callbacks=[callback])
    
    save_model(model2)


    # eval_result = hypermodel.evaluate(X_test_normal, y_test)
    # print("[test loss, test accuracy]:", eval_result)
    # history2 = model_2.fit(X_train_normal, y_train, epochs=100, callbacks=[callback])

    # print(model_2.evaluate(X_test_normal, y_test))


    pd.DataFrame(history.history).plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.show()
    # l = np.arange(len(y_test))

