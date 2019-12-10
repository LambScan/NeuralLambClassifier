""" Custom Generator Class"""

from sklearn.utils import shuffle
import numpy as np
from keras.utils import Sequence
import cv2
import os



class My_Custom_Generator(Sequence):
    """ Custom generator. """

    def __init__(self, parent_folder, image_filenames, labels, batch_size, use_shuffle=True):
        if use_shuffle:
            seed = np.random.randint(1000000)
            self.image_filenames = shuffle(image_filenames, random_state=seed)
            self.labels = shuffle(labels, random_state=seed)
        else:
            self.image_filenames = image_filenames
            self.labels = labels
        self.batch_size = batch_size
        self.parent_folder = parent_folder
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
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        xRet = []
        yRet = []

        for ind, file_name in enumerate(batch_x):
            img = cv2.imread(os.path.join(self.parent_folder, str(file_name) + ".png"), cv2.IMREAD_ANYDEPTH)
            xRet.append(img.reshape(230, 510, 1))
            yRet.append(batch_y[ind])
            for t in self.transforms:
                xRet.append(t(img).reshape(230, 510, 1))
                yRet.append(batch_y[ind])

        return np.array(xRet), np.array(yRet)
