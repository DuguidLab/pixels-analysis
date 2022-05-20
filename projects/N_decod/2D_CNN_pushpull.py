import os
from importlib_metadata import version
import numpy as np
from torch import seed
import tensorflow.config.experimental as mental
gpus = mental.list_physical_devices('GPU')
for gpu in gpus:
    mental.set_memory_growth(gpu, True)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GaussianNoise #This is for later
from sklearn.model_selection import StratifiedKFold


### Defines current path for programme
current_path=os.path.dirname(__file__)

### Load training, validation and hyperparameter scoring datasets
def load_data(current_path, seeds=None):
    new_folder = "npy_data"
    path = os.path.join(current_path, new_folder)       
    if not os.path.exists(path):
        print("There is no {} folder".format(path))
    list_of_files=[element.rsplit(("'")[0])[1] for element in os.listdir(path)] #Only 2 datasets are used, third is for true scoring (better be avoided during model optimization)
    label_train=list_of_files[0]
    X_train=np.load("{}\\M1_3D-['{}'].npy".format(path, label_train))
    Y_train=np.load("{}\\M1_L-['{}'].npy".format(path, label_train))
    label_valsc=list_of_files[1]
    SKFOLD_valsc=StratifiedKFold(n_splits=1, shuffle=True, random_state=seeds)
    X_valsc=np.load("{}\\M1_3D-['{}'].npy".format(path, label_valsc))
    Y_valsc=np.load("{}\\M1_L-['{}'].npy".format(path, label_valsc))
    for val_ix, score_ix in SKFOLD_valsc.split(X_valsc, Y_valsc):
            X_val, X_scoring = X_valsc[val_ix, :, :], X_valsc[score_ix, :, :]
            Y_val, Y_scoring = Y_valsc[val_ix], Y_valsc[score_ix]
    return X_train, Y_train, X_val, Y_val, X_scoring, Y_scoring, label_train, label_valsc


def define_model(dim3, model): #This function will be called to generate all models
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import InputLayer
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPooling1D
    from tensorflow.keras.layers import LSTM
    from keras.regularizers import L2

    l2=L2(0.01) #Weight regularization hyperparameter
    if model==1: #This model is sliding 1-D windows, no assumed proximal context
        modelname="Sliding 1-D CNN"
        version="1.0.0"
        model=Sequential([
        InputLayer(input_shape=(60000, dim3)), #batch_size missing
        BatchNormalization(),
        Conv1D(128, 31, kernel_regularizer=l2, bias_regularizer=l2), #Stride reduces feature map size. Kernel sizes 31, 17, 7, 5, 3?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(2, strides=2), #Not overlapping pooling
        #Layer
        BatchNormalization(),
        Conv1D(256, 5, strides=2, kernel_regularizer=l2, bias_regularizer=l2), 
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(2, strides=2),
        #Layer
        BatchNormalization(),
        Conv1D(384, 3, stride=2, kernel_regularizer=l2, bias_regularizer=l2), 
        Activation("relu"),#Inverting order with pooling
        #Layer
        BatchNormalization(),
        Conv1D(512, 3, kernel_regularizer=l2, bias_regularizer=l2), #Here I increased feature number and removed relu
        #Layer
        MaxPooling1D(2, strides=2),
        #Layer
        Flatten(),
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(2048, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
                #Layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
                #Layer
        BatchNormalization(), #Here I removed the dropout
        Dense(512, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        Dense(1, activation="sigmoid")
        ])
    elif model==2:
        modelname="LSTM_1"
        version="1.0.0"
        model=Sequential([
        InputLayer(input_shape=(60000, dim3)), #batch_size missing
        #Layer
        BatchNormalization(),
        LSTM(2048, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2, bias_regularizer=l2, recurrent_regularizer=l2), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(3), #Overlapping pooling
        #Layer
        BatchNormalization(),
        LSTM(2048, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2, bias_regularizer=l2, recurrent_regularizer=l2), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(3), #Overlapping pooling
        #Layer
        BatchNormalization(),
        LSTM(2048, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2, bias_regularizer=l2, recurrent_regularizer=l2), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling        
        #Layer
        Flatten(),
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(2048, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        #Layer
        BatchNormalization(), #Here I removed the dropout
        Dense(512, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        Dense(1, activation="sigmoid")
        ])
    elif model==3:
        modelname="CNN+LSTM_1"
        version="1.0.0"
        model=Sequential([
        InputLayer(input_shape=(60000, dim3)), #batch_size missing
        #Layer
        Conv1D(128, 31, kernel_regularizer=l2, bias_regularizer=l2), #Stride reduces feature map size. Kernel sizes 31, 17, 7, 5, 3?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(2, strides=2), #Not overlapping pooling
        #Layer
        BatchNormalization(),
        Conv1D(256, 5, strides=2, kernel_regularizer=l2, bias_regularizer=l2), 
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(2, strides=2),
        #Layer
        BatchNormalization(),
        Conv1D(384, 3, stride=2, kernel_regularizer=l2, bias_regularizer=l2), 
        Activation("relu"),#Inverting order with pooling
        #Layer
        BatchNormalization(),
        LSTM(1024, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2, bias_regularizer=l2, recurrent_regularizer=l2), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(3, strides=2), #Overlapping pooling
        #Layer
        BatchNormalization(),
        LSTM(1024, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2, bias_regularizer=l2, recurrent_regularizer=l2), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(3, strides=2), #Overlapping pooling
        #Layer
        BatchNormalization(),
        LSTM(1024, dropout=0.1, recurrent_dropout=0.1, kernel_regularizer=l2, bias_regularizer=l2, recurrent_regularizer=l2), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling        
        #Layer
        Flatten(),
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        #Layer
        BatchNormalization(), #Here I removed the dropout
        Dense(512, activation='relu', kernel_regularizer=l2, bias_regularizer=l2),
        Dense(1, activation="sigmoid")
        ])
    return model, modelname, version


def train_model_n(X_train, Y_train, X_val, Y_val, X_scoring, Y_scoring, batch_size, model_id, iter, label_train, label_valsc):

    from sklearn.metrics import fbeta_score
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorflow.keras.backend as K
    from tensorflow.keras.callbacks import EarlyStopping
    import gc
    #import pickle #to save the trained models
    plt.style.use('seaborn-whitegrid')
    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelweight='bold', labelsize='large',
    titleweight='bold', titlesize=18, titlepad=10)
    plt.rc('animation', html='html5')
    #from tensorflow.compat.v1 import reset_default_graph

    Optimizer=Adam(
    learning_rate=0.001,
    epsilon=1
    ) #Further tuning? Look into dissertation data
    early_stopping=EarlyStopping(
    patience=80,
    restore_best_weights=True
    )
    _, _, dim3=X_train.shape
    model, modelname, version=define_model(dim3, model_id)
    model.compile(
        optimizer=Optimizer, #return "adam" to backtrack
        loss="binary_crossentropy",
        metrics=["accuracy"] #defined in the function args
    )
    model.summary()
    print(">>>Training Model: {} on iteration {} for batch size {}".format(modelname, iter, batch_size))
    history=model.fit(
        X_train, Y_train, validation_data=(X_val, Y_val),
        batch_size=batch_size, epochs=1000, callbacks=[early_stopping],
        verbose=0
    )
    #Y_pred=(model.predict(X_test) > 0.5).astype("int32")
    #Y_pred=(model(X_test, training=False).numpy() > 0.5).astype("int32") #Only in eager execution
    Y_pred=model.predict_classes(X_scoring, batch_size=batch_size)
    score = fbeta_score(Y_scoring, Y_pred, beta=1.0)
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    plt.savefig('/Photos/{}_{}_model_{}_version_{}'.format(label_train, label_valsc, iter, modelname, version))
    plt.close()
    plt.clf()
    with open("results.txt", "a") as text_file:
        text_file.write("Model: {}, Model_version: {}, Score: {}, iteration: {}, t_dataset_ {}, valsc_dataset: {}, batch_size:{} \n".format(modelname, version, score, iter, label_train, label_valsc, batch_size))
        text_file.close()
    #All of this is to free as much memory as possible for each cycle of training
    #reset_default_graph() #Only works in tensorflow.v1, if not in use not useful
    K.clear_session()
    del model, history, history_df, Y_pred
    gc.collect()
    return score


def model_training(current_path, batch_size, models_to_train, seeds=None):
    from time import time
    X_train, Y_train, X_val, Y_val, X_scoring, Y_scoring, label_train, label_valsc=load_data(current_path, seeds)
    #Next two loops effectively perform a K-fold of K=2, with 3 iterations for each, by swapping the val/sc datasets
    All_scores=[]
    All_times=[]
    models_examined=[]
    for model_id in range(1, models_to_train+1):
        for iter in range(1,4):
            timestart=time()
            score=train_model_n(X_train, Y_train, X_val, Y_val, X_scoring, Y_scoring, batch_size, model_id, iter, label_train, label_valsc)
            timeend=time()-timestart
            All_scores.append(score)
            All_times.append(timeend)
        for iter in range(4,7):
            timestart=time()
            score=train_model_n(X_train, Y_train, X_scoring, Y_scoring, X_val, Y_val, batch_size, model_id, iter, label_train, label_valsc)
            timeend=time()-timestart
            All_scores.append(score)
            All_times.append(timeend)
        models_examined.append(model_id)
    with open("results.txt", "a") as text_file:
        text_file.write("Overall evaluation of models by ID:\n")
        for model_id in models_examined:
            sc_mean=np.mean(All_scores[0+6*(model_id-1):6+6*(model_id-1)])
            sc_sd=np.std(All_scores[0+6*(model_id-1):6+6*(model_id-1)])
            time_mean=np.mean(All_times[0+6*(model_id-1):6+6*(model_id-1)])
            time_sd=np.std(All_times[0+6*(model_id-1):6+6*(model_id-1)])
            text_file.write("Model_id: {}, Score_mean: {}, Score_sd:{}, time_mean:{}, time_sd:{}, batch_size:{} \n".format(model_id, sc_mean, sc_sd, time_mean, time_sd, batch_size))
        text_file.write("Start of new evaluation\n")
        text_file.close()
### Execution
seeds=None
models_to_train=3
batch_sizes=[16,24,32]
for batch_size in batch_sizes:
    model_training(current_path, batch_size, models_to_train, seeds)

#Other things to try:
# Change to overlapping pooling
#Change feature numbers
#Change kernel sizes
#Change Dense layer shape
