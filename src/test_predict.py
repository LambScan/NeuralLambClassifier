import numpy as np
import cv2
import os
from keras import models

# funcion para rellenar con 0
def cropWithZeros(in_array, x, y, h, w):
    in_array = np.array(in_array)
    shape = in_array.shape
    crop = in_array[y:y+h, x:x+w]
    bx = shape[0]-x-h
    by = shape[1]-y-w
    padding = ((x,bx),(y,by))
    return np.pad(crop, padding)




def predecir(imagen):
    parent_folder = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(parent_folder, "modeloYpesos_50.h5") # ruta al archivo .h5 con la red
    model = models.load_model(path)

    #crop
    img = cropWithZeros(imagen, 38, 102, 230, 503)

    #predict
    img = np.array([img.reshape(480,640,1)])
    res = model.predict(img, verbose=0, use_multiprocessing=True)

    i = np.argmax(np.array(res))

    if i == 0:
        label = "lamb"
    elif i == 1:
        label = "empty"
    elif i == 2:
        label = "wrong"
    else:
        label = "fly"

    return label

