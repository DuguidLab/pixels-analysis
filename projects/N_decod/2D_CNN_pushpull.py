import os
import numpy as np
import tensorflow.config.experimental as mental
gpus = mental.list_physical_devices('GPU')
for gpu in gpus:
    mental.set_memory_growth(gpu, True)
from tensorflow.keras.optimizers import Adam
#Remove all strides and change poolings 

current_path=os.path.dirname(__file__)
new_folder = "npy_data"
path = os.path.join(current_path, new_folder)       
if not os.path.exists(path):
    print("There is no {} folder".format(path))
list_of_files=[element.rsplit(("'")[0])[1] for element in os.listdir(path)]
label_train=list_of_files[0]
X=np.load("{}\\M1_3D-['{}'].npy".format(path, label_train))
Y=np.load("{}\\M1_L-['{}'].npy".format(path, label_train))
def define_model(dim3, A):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import InputLayer
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling1D
    from tensorflow.keras.layers import GaussianNoise
    if A==3:
        model=Sequential([
        InputLayer(input_shape=(1000, dim3)), #batch_size missing
        BatchNormalization(),
        Conv2D(96, 11, strides=4), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(2, strides=2), #Overlapping pooling
        #Layer
        BatchNormalization(),
        Conv2D(256, 5, strides=2), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(2, strides=2),
        #Layer
        BatchNormalization(),
        Conv2D(384, 3), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        #Layer
        BatchNormalization(),
        Conv2D(384, 3), #Removing stride could improve?
        Activation("relu"),#Inverting order with poolsing
        #Layer
        BatchNormalization(),
        Conv2D(256, 3), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(2, strides=2),
        #Layer
        Flatten(),
        #Layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(2048, activation='relu'),
        #Layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(2048, activation='relu'),
                #Layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(2048, activation='relu'),
                #Layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(2048, activation='relu'),
        Dense(1, activation="sigmoid")
        ])
    if A==6:
        model=Sequential([
        InputLayer(input_shape=(1000, dim3)), #batch_size missing
        BatchNormalization(),
        Conv2D(96, 11, strides=4), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(3, strides=2), #Overlapping pooling
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(256, 5, strides=2), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        MaxPooling1D(3, strides=2),
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(384, 3), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(384, 3), #Removing stride could improve?
        Activation("relu"),#Inverting order with poolsing
        #Layer
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(256, 3), #Removing stride could improve?
        Activation("relu"),#Inverting order with pooling
        #Layer
        Flatten(),
        #Layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(2048, activation='relu'),
        #Layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(2048, activation='relu'),
                #Layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(2048, activation='relu'),
                #Layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(2048, activation='relu'),
        Dense(1, activation="sigmoid")
        ])
    return model


print(label_train)
