from colored import fg
print(fg(245)) # material innecesario en gris
import numpy as np
import cv2
import psutil, shutil
import os
import keras
from keras import layers
from keras import models
from keras import utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tensorflow as tf
from os import listdir
from os.path import isfile, join

# --------------------------------- PARAMETROS ---------------------------------

'''
La lista con las funciones esta al final del fichero

listaFunciones = []
'''


# arquitectura de la red
kernel_size_3  = (3, 3)
kernel_size_5  = (5, 5)
kernel_size_7  = (7, 7)
kernel_size_9  = (9, 9)
kernel_size_11 = (11, 11)
pool_size_2 = (2, 2)


###########################################################       INDICE       ###########################################################

'''
Red 0:
    - Entrada de (480, 640)
    - Salida de 4 categorias
    - 4.200.000  parametros entrenables
    - 90,9% val_accuracy:
        10 epochs
        batch_size = 2
        learning_rate = 0.00001
        datos normalizados [0,1]


Red 1:
    - Entrada de (230, 510)
    - Salida de 3 categorias
    - 1.589.939  parametros
    - 92,56% val_accuracy:
        10 epochs
        batch_size = 2
        learning_rate = 0.00001
        datos normalizados [0,3000]

    - 93,98% val_accuracy:
        10 epochs
        batch_size = 10
        learning_rate = 0.00001
        datos normalizados [0,3000]


Red 2:
    - Entrada de (230, 510)
    - Salida de 3 categorias
    - 28.037.939  parametros (2 convolucionales menos que el M1)
    - 91,27% val_accuracy:
        10 epochs
        batch_size = 10
        learning_rate = 0.00001
        datos normalizados [0,3000]


Red 3:
    - Entrada de (230, 510)
    - Salida de 3 categorias
    - 228.467  parametros (M1 con menos kernels)
    - 93,08% val_accuracy:
        30 epochs
        batch_size = 10
        learning_rate = 0.00001
        datos normalizados [0,3000]
        (muy poco overfitting)


Red 4:
    - Entrada de (230, 510)
    - Salida de 3 categorias
    - 992.851  parametros (M3 con mas kernel en la ultima convolucion y mas dense)
    - 94,08% val_accuracy:
        30 epochs
        batch_size = 10
        learning_rate = 0.00001
        datos normalizados [0,3000]
        (poco estable, fluctua mucho)


Red 5:
    - Entrada de (230, 510)
    - Salida de 3 categorias
    - 29.163  parametros (M4 con solo 8 kernels)
    - 92,28% val_accuracy:
        100 epochs
        batch_size = 10
        learning_rate = 0.00001
        datos normalizados [0,3000]
        (MUY estable, sin apenas overfitting)


Red 6:
    - Entrada de (230, 510)
    - Salida de 3 categorias
    - 495.363  parametros (equilibrada, menos kernels)
    - 94,08% val_accuracy:
        10 epochs
        batch_size = 10
        learning_rate = 0.00001
        datos normalizados [0,3000]
        (MUY estable, sin apenas overfitting)



'''






###########################################################   MODELOS NUMERADOS   ###########################################################

def cargarModelo(id):
    '''Carga el modelo con el identificador numerico dado.'''

    if 0 <= id < len(listaFunciones):
        return listaFunciones[id]()

    else:
        B = "\033[;37m"
        R = "\x1b[1;31m"
        print(R + "[!] El numero de modelo " +B+ str(id) +R+ " no existe. Introduzca un numero de modelo entre "+B+"0"+R+" y " +B+ str(len(listaFunciones)-1) +R+ "." +B )




###################################################################       ###################################################################
###################################################################   0   ###################################################################
###################################################################       ###################################################################

def M0():
    model = models.Sequential()


    #VGG16 like

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3,input_shape=(480, 640, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))

    model.add(layers.Flatten())
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(4, activation='softmax'))

    return model




###################################################################       ###################################################################
###################################################################   1   ###################################################################
###################################################################       ###################################################################

def M1():
    model = models.Sequential()


    #VGG16 like

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3,input_shape=(230, 510, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))

    model.add(layers.Flatten())
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(3, activation='softmax'))

    return model


###################################################################       ###################################################################
###################################################################   2   ###################################################################
###################################################################       ###################################################################

def M2():
    model = models.Sequential()


    #VGG16 like

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3,input_shape=(230, 510, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))

    model.add(layers.Flatten())
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(3, activation='softmax'))

    return model





###################################################################       ###################################################################
###################################################################   3   ###################################################################
###################################################################       ###################################################################

def M3():
    model = models.Sequential()


    #VGG16 like

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3,input_shape=(230, 510, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))

    model.add(layers.Flatten())
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(3, activation='softmax'))

    return model



###################################################################       ###################################################################
###################################################################   4   ###################################################################
###################################################################       ###################################################################

def M4():
    model = models.Sequential()


    #VGG16 like

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3,input_shape=(230, 510, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(3, activation='softmax'))

    return model

###################################################################       ###################################################################
###################################################################   5   ###################################################################
###################################################################       ###################################################################

def M5():
    model = models.Sequential()


    #VGG16 like

    model.add(layers.Conv2D(8, kernel_size=kernel_size_3,input_shape=(230, 510, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(8, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(8, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(8, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(8, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(8, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(3, activation='softmax'))

    return model


###################################################################       ###################################################################
###################################################################   6   ###################################################################
###################################################################       ###################################################################

def M6():
    model = models.Sequential()


    #VGG16 like

    model.add(layers.Conv2D(16, kernel_size=kernel_size_3,input_shape=(230, 510, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(16, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(3, activation='softmax'))

    return model



###################################################################       ###################################################################
###################################################################   7   ###################################################################
###################################################################       ###################################################################

def M7():
    model = models.Sequential()


    #VGG16 like

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3,input_shape=(230, 510, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_5, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(3, activation='softmax'))

    return model



###################################################################       ###################################################################
###################################################################   8   ###################################################################
###################################################################       ###################################################################

def M8():
    model = models.Sequential()

    distribucion = keras.initializers.he_normal(seed=None)


    #VGG16 like

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3,input_shape=(230, 510, 1) , activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=pool_size_2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size=kernel_size_3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size_2))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(3, activation='softmax'))

    return model


############################################################   lista de modelos   ############################################################


listaFunciones = [M0,M1,M2,M3,M4,M5,M6,M7,M8]
