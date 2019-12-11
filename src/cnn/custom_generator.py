""" Custom Generator Class"""

from sklearn.utils import shuffle
import numpy as np
from keras.utils import Sequence
import cv2
import os



class My_Custom_Generator(Sequence):
    """ Custom generator. """

    def __init__(self, parent_folder, image_filenames, labels, batch_size, target_size, use_shuffle=True):
        """
        Ctor.

            - parent_folder: ruta absoluta al directorio raiz del proyecto
            - image_filenames: lista con todos los nombres de las imagenes (sin extension, pues esta se añade despues)
            - labels: etiquetas de las imagenes. Deben estar ordenadas con los nombres de las imagenes.
            - batch_size: conjunto de datos a devolver en cada llamada al iterador.
            - target_size: tamaño objetivo de las imagenes a devolver.
            - use_shuffle: baraja las listas de entrada para aleatorizar los datos. (mantiene el orden entre imagen y etiqueta)
        """
        if use_shuffle:
            seed = np.random.randint(1000000)
            self.image_filenames = shuffle(image_filenames, random_state=seed)
            self.labels = shuffle(labels, random_state=seed)
        else:
            self.image_filenames = image_filenames
            self.labels = labels
        self.batch_size = batch_size
        self.parent_folder = parent_folder
        self.target_size = target_size
        self.transforms = []

    def add_transform(self, func):
        """ Añade una transformacion a la lista de transformaciones de Data Aumentation. """
        self.transforms.append(func)

    def get_num_transform(self):
        """ Devuelve el numero de transformaciones que van a aplicarse como Data Aumentation. (0 si no se aplica ninguna) """
        return len(self.transforms)

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        """ Iterador. """
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        xRet = []
        yRet = []

        for ind, file_name in enumerate(batch_x):
            img = cv2.imread(os.path.join(self.parent_folder, str(file_name) + ".png"), cv2.IMREAD_ANYDEPTH)
            xRet.append(img.reshape(self.target_size))
            yRet.append(batch_y[ind])
            for t in self.transforms:
                xRet.append(t(img).reshape(self.target_size))
                yRet.append(batch_y[ind])

        return np.array(xRet), np.array(yRet)
