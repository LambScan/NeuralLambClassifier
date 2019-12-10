from colored import fg
import os


'''
Tensorflow Message Debug Level:<

    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(fg(245))  # material innecesario en gris
import numpy as np
import cv2
import psutil, shutil
import keras

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.utils import shuffle

from os import listdir
from os.path import isfile, join

#project imports
from cnn.models import loadModels
from cnn.util import printProgressBar
from cnn.util import cropWithZeros
from cnn.custom_generator import My_Custom_Generator



class Model_constructor():
    """ Clase constructora de un modelo de NN completo. """

    def __init__(self, parent_folder, model_params, exec_color):
        """
            Ctor.
            Elementos necesarios:
                - model_params: diccionario con los valores de los parametros.
                - main_color: color destacado de la ejecucion.
        """
        self.ID_MODELO = model_params["id_modelo"]

        self.epochs = model_params["epochs"]
        self.batch_size = model_params["batch_size"]

        self.loading_batch_size = model_params["loading_batch_size"]
        self.learning_rate = model_params["learning_rate"]

        self.workers = model_params["workers"]
        self.RAM_PERCENT_LIMIT = model_params["ram_percent_limit"]

        # callbacks
        self.paciencia = model_params["paciencia"]

        # porcentaje de uso del dataset en entrenamiento (el resto va a la validacion)
        self.train_percent = model_params["train_percent"]

        self.num_train = -1
        self.num_val = -1

        self.C = exec_color["main"]
        self.B = exec_color["default"]

        #paths
        self.parent_folder = parent_folder

        self.label_numpy_path = os.path.join(self.parent_folder, "dataset", "dataCUS", "labels.npy")
        self.data_path = os.path.join(self.parent_folder, "dataset", "dataCUS")

    def create_model(self):
        print(self.B + "Usando Modelo " + self.C + str(self.ID_MODELO) + self.B)
        return loadModels(self.ID_MODELO)


    def compile_model(self, model):
        adam = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model


    def get_generators(self):
        label_list = np.load(self.label_numpy_path, allow_pickle=True)
        img_names_list = np.arange(0, len(label_list), 1)

        seed = np.random.randint(1000000)
        label_list = shuffle(label_list, random_state=seed)
        img_names_list = shuffle(img_names_list, random_state=seed)

        frontera = int(round(len(img_names_list) * self.train_percent, 0))
        train_img_names = np.array(img_names_list[:frontera])
        train_labels = np.array(label_list[:frontera])
        val_img_names = np.array(img_names_list[frontera:])
        val_labels = np.array(label_list[frontera:])

        # creamos los generadores
        train_it = My_Custom_Generator(self.data_path, train_img_names, train_labels, self.loading_batch_size)
        train_it.add_transform(lambda x: cv2.flip(x, 0))
        train_it.add_transform(lambda x: cv2.flip(x, 1))
        train_it.add_transform(lambda x: cv2.flip(x, -1))

        val_it = My_Custom_Generator(self.data_path, val_img_names, val_labels, self.loading_batch_size)
        val_it.add_transform(lambda x: cv2.flip(x, 0))
        val_it.add_transform(lambda x: cv2.flip(x, 1))
        val_it.add_transform(lambda x: cv2.flip(x, -1))


        self.num_train = train_img_names.size * (train_it.get_num_transform() + 1)
        self.num_val = val_img_names.size * (val_it.get_num_transform() + 1)

        print(self.B + "Imagenes de entrenamiento " + self.C + str(self.num_train) + self.B)
        print(self.B + "Imagenes de validacion " + self.C + str(self.num_val) + self.B)

        return train_it, val_it


    def get_dataset(self):
        """ Devuelve un numpy con el dataset cargado en RAM. """
        data_folder = os.path.join(self.parent_folder, "dataset", "dataNP")

        onlyfiles = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]

        all_data = []
        all_label = []

        print(self.B + "Numero de elementos encontrados: " + self.C + str(len(onlyfiles)), end='\n\n')
        printProgressBar(0, len(onlyfiles), prefix='Carga del dataset:',
                         suffix='Completado (' + self.C + str(psutil.virtual_memory()[2]) + self.B + '% RAM)',
                         length=100)
        i = 0
        for fil in onlyfiles:

            # cargamos el archivo
            l = np.load(os.path.join(data_folder, str(fil)), allow_pickle=True)
            # separamos datos y etiquetas
            l = l.tolist()
            all_data.append(cropWithZeros(l[0], 38, 102, 230, 503).reshape(480, 640, 1))
            all_label.append(l[1])
            # progress bar
            i = i + 1
            printProgressBar(i, len(onlyfiles), prefix='Carga del dataset:',
                             suffix='Completado (' + self.C + str(psutil.virtual_memory()[2]) + self.B + '% RAM)',
                             length=100)
            if psutil.virtual_memory()[2] > self.RAM_PERCENT_LIMIT:
                break

        print(self.B)

        # dataset preparado
        frontera = int(round(len(all_data) * self.train_percent, 0))

        train_images = np.array(all_data[:frontera])
        train_labels = np.array(all_label[:frontera])

        test_images = np.array(all_data[frontera:])
        test_labels = np.array(all_label[frontera:])

        print(
            "\n\n==============================================================================\n\nEl dataset se compone de " + self.C + str(
                len(all_data)) + self.B + " elementos."
            + "\nSe utilizara el " + self.C + str(
                round(self.train_percent * 100, 0)) + self.B + "% del dataset en el entrenamiento. (" + self.C + str(
                len(train_labels)) + self.B + " elementos)"
            + "\nEl otro " + self.C + str(
                round((1 - self.train_percent) * 100,
                      0)) + self.B + "% se utilizara en la validacion del modelo. (" + self.C + str(
                len(test_labels)) + self.B + " elementos)"
            + "\n\n==============================================================================\n\n")

        return (train_images, train_labels), (test_images, test_labels)




    def fit_model(self, model, data_input, use_generators, use_tensorboard=False):
        print(self.B
              + "#############################################"
              + self.C + "    Entrenando la red    " + self.B
              + "#############################################")

        # Callbacks para el entrenamiento


        calls = [
            keras.callbacks.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=self.paciencia,
                                                    verbose=0,
                                                    mode='auto', baseline=None, restore_best_weights=True),
            keras.callbacks.callbacks.LambdaCallback(on_epoch_begin=lambda e, l: print(fg(np.random.randint(130, 232))),
                                                     on_epoch_end=lambda e, l: print(fg(248)), on_batch_begin=None,
                                                     on_batch_end=None, on_train_begin=None, on_train_end=None)]

        if use_tensorboard:
            board_path = os.path.join(self.parent_folder, "last_exec_data")
            if os.path.exists(board_path):
                shutil.rmtree(board_path)
            board_call = keras.callbacks.TensorBoard(log_dir=board_path)
            calls.append(board_call)

        history = []

        if use_generators:
            # entrenamiento del modelo iterativo
            train_it, val_it = data_input
            stepsXepoch = self.num_train // self.batch_size
            val_steps   = self.num_val  // self.batch_size
            history = model.fit_generator(train_it,
                                            steps_per_epoch=stepsXepoch,
                                            epochs = self.epochs,
                                            verbose = 1,
                                            validation_data = val_it,
                                            validation_steps = val_steps,
                                            use_multiprocessing = True,
                                            workers = self.workers,
                                            shuffle = True,
                                            callbacks = calls)
        else:
            # entrenamiento tradicional
            train_images, train_labels = data_input[0]
            test_images, test_labels   = data_input[1]
            history = model.fit(train_images,
                                    train_labels,
                                    batch_size = self.batch_size,
                                    epochs = self.epochs,
                                    validation_data = (test_images,test_labels),
                                    verbose = 1,
                                    use_multiprocessing = True,
                                    callbacks = calls,
                                    workers = self.workers)

        return history



    def show_plot(self, history):

        ent_loss = history.history['loss']
        val_loss = history.history['val_loss']
        ent_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        Gepochs = range(1, len(ent_loss) + 1)

        plt.style.use('dark_background')
        fig, axs = plt.subplots(2)
        fig.suptitle('Loss & Accuracy')

        axs[0].set_ylim(top=1) # MAX_Y_LOSS

        axs[0].plot(Gepochs, ent_loss, 'lightcoral', label='Training Loss')
        axs[0].plot(Gepochs, val_loss, 'sandybrown', label='Test Loss')
        axs[1].plot(Gepochs, ent_acc, 'limegreen', label='Training Accuracy')
        axs[1].plot(Gepochs, val_acc, 'greenyellow', label='Test Accuracy')
        plt.xlabel('Epochs')
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].legend()
        axs[1].legend()

        plt.show()


    def save_model(self, model, keyword):

        iter_name = keyword

        model_path = os.path.join(self.parent_folder, "modelos")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save(
            model_path + "/" + "modeloYpesos_" + iter_name + "_M" + str(self.ID_MODELO) + "_epochs" + str(self.epochs) + "_batch" + str(
                self.batch_size) + ".h5")


        print(self.B + "Modelo " + self.C + "Guardado" + self.B + ".\n")
