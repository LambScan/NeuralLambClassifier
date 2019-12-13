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
                - parent_folder: ruta absoluta al directorio raiz del proyecto.
                - model_params: diccionario con los valores de los parametros.
                - exec_color: color de los elementos destacados impresos por pantalla.
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

        # best_score
        self.last_score = 999999999


    def create_model(self):
        """ Carga un modelo de la lista de modelos predefinidos y lo devuelve. """
        print("#################################################################\n")
        print(self.B + "Usando Modelo " + self.C + str(self.ID_MODELO) + self.B, end='\n\n')
        return loadModels(self.ID_MODELO)


    def compile_model(self, model, regression=False):
        """
        Compila un modelo dado.

            - Optimizador usado: Adam
            - regression: Adapta el optimizador a un problema de regresion
        """
        if regression:
            adam = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
            model.compile(loss="mean_absolute_percentage_error", optimizer=adam)
            return model
        else:
            adam = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
            return model


    def get_generators(self, dataset_path, target_size, data_aumentation=True):
        """
        Devuelve un par de generadores de entrenamiento y validacion.

            - dataset_path: ruta absoluta al directorio que contiene el dataset en el formato CUS. (nombreDirectorioCUS)
            - target_size: tupla de python con el tamaño que se espera que entre en la red. (Ej.: (640,480,1))
            - data_aumentation: Aumenta el dataset a base de realizar modificaciones. Aumento del 400%
        """
        #path
        label_numpy_path = os.path.join(dataset_path, "labels.npy")


        label_list = np.load(label_numpy_path, allow_pickle=True)
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
        train_it = My_Custom_Generator(dataset_path, train_img_names, train_labels, self.loading_batch_size, target_size)
        if data_aumentation:
            train_it.add_transform(lambda x: cv2.flip(x, 0))
            train_it.add_transform(lambda x: cv2.flip(x, 1))
            train_it.add_transform(lambda x: cv2.flip(x, -1))

        val_it = My_Custom_Generator(dataset_path, val_img_names, val_labels, self.loading_batch_size, target_size)
        if data_aumentation:
            val_it.add_transform(lambda x: cv2.flip(x, 0))
            val_it.add_transform(lambda x: cv2.flip(x, 1))
            val_it.add_transform(lambda x: cv2.flip(x, -1))


        self.num_train = train_img_names.size * (train_it.get_num_transform() + 1)
        self.num_val = val_img_names.size * (val_it.get_num_transform() + 1)

        print(self.B + "Imagenes de entrenamiento " + self.C + str(self.num_train) + self.B)
        print(self.B + "Imagenes de validacion " + self.C + str(self.num_val) + self.B)

        return train_it, val_it


    def get_dataset(self):
        """
         Devuelve un numpy con el dataset cargado en RAM.
         El dataset debe componerse de .npy con [imagen,label]
        """
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




    def fit_model(self, model, data_input, use_generators, use_tensorboard=False, regression=False, evaluate_each_epoch=False, color=-1, restore_best_weights=False):
        """
        Entrena el modelo con los datos de entrada, devolviendo la historia del entrenamiento.

            - data_input: datos de entrada.
                -- SI se usan generadores, esta variable contendra una tupla con los iteradores. (train_iterator, validation_iterator)
                -- si NO se usan generadores, esta variable contendra una tupla con 4 numpys. (train_images, train_labels, test_images, test_labels)
            - use_generators: Establece si van a usarse generadores.
            - use_tensorboard: Establece si va a utilizarse TensorBoard, añadiendo el callback.
            - regression: Establece si el problema a entrenar es de tipo Regresion.
            - color: Color de la salida durante el entrenamiento.
                --   -1  = rainbow
                -- 0-255 = color equivalente a fg(n)
        """
        print(self.B
              + "#############################################"
              + self.C + "    Entrenando la red    " + self.B
              + "#############################################")

        # Callbacks para el entrenamiento


        calls = []
        if regression:
            calls.append(keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=self.paciencia,
                                                    verbose=0,
                                                    mode='auto', baseline=None, restore_best_weights=True))
        else:
            calls.append(keras.callbacks.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=self.paciencia,
                                                    verbose=0,
                                                    mode='auto', baseline=None, restore_best_weights=True))

        if color == -1:
            calls.append(keras.callbacks.callbacks.LambdaCallback(on_epoch_begin=lambda e, l: print(fg(np.random.randint(130, 232))),
                                                     on_epoch_end=lambda e, l: print(fg(248)), on_batch_begin=None,
                                                     on_batch_end=None, on_train_begin=None, on_train_end=None))
        elif color >= 0 and color < 256:
            calls.append(keras.callbacks.callbacks.LambdaCallback(on_epoch_begin=lambda e, l: print(fg(color)),
                                                     on_epoch_end=lambda e, l: print(fg(248)), on_batch_begin=None,
                                                     on_batch_end=None, on_train_begin=None, on_train_end=None))

        if use_tensorboard:
            board_path = os.path.join(self.parent_folder, "last_exec_data")
            if os.path.exists(board_path):
                shutil.rmtree(board_path)
            board_call = keras.callbacks.TensorBoard(log_dir=board_path)
            calls.append(board_call)



        # callback para poder evaluar la red entre epocas
        # TODO -> recoger la informacion de la evaluacion para poder representarla mas tarde en una grafica
        if evaluate_each_epoch:
            class PredictionCallback(keras.callbacks.Callback):
                def __init__(self, MC, C, B, restore_best_weights):
                    self.MC = MC
                    self.C = C
                    self.B = B
                    self.restore_best_weights = restore_best_weights

                def on_epoch_end(self, epoch, logs={}):
                    print("\033[2A" + "\033[108C", end='')
                    eva = self.MC.evaluate_regression_model(model, val_it, use_generators=True, is_callback=True)
                    print("m: " + self.C + '%.2f' % eva[0] + self.B + ", d: " +
                          self.C + '%.2f' % eva[1] + self.B + "", end='')
                    if self.restore_best_weights:
                        # comprobamos si el score ha mejorado
                        if self.MC.last_score > eva[0]:
                            # guardamos el modelo
                            self.MC.save_model(model, "==TEMP==", is_temp_file=True, quiet=True)
                            # actualizamos el score
                            self.MC.last_score = eva[0]
                            # imprimimos una sutil marca de mejora
                            print(" *\n")
                        else:
                            print("\n")
                    else:
                        print("\n")

            calls.append(PredictionCallback(self, self.C, self.B, restore_best_weights))




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



        if restore_best_weights:
            temp_path = os.path.join(self.parent_folder, "models", "==TEMP==.h5")
            # comprobamos si esta el archivo
            if os.path.exists(temp_path):
                # cargamos el modelo
                model = self.load_model(temp_path, quiet=True)
                # borramos el archivo
                os.remove(temp_path)


        return history, model


    def evaluate_regression_model(self, model, validation_data, use_generators=False, is_callback=False, extended_info=False, num_examples=1):
        """
         Evalua la red contrastando las predicciones que realiza con los datos de validacion de entrada.

            - validation_data: datos de validacion. Esta variable se compone de una tupla o generador con las imagenes de
                                validacion y sus etiquetas. (val_images, val_labels)

            - use_generators: establece el uso de generadores como enbtrada de datos.
            - is_callback: adapta la salida por pantalla para el uso entre epocas durante el entrenamiento.
            - extended_info: devuelve mas info.
                        - True: (media, desviacion, maximo, minimo, mediana, average, varianza, [num_examples x (true_label, prediction)] )
                        - False: (media, desviacion_tipica)

            - num_examples: numero de ejemplos devueltos con la informacion extendida.

            = Return: info
        """

        val_labels = []
        preds      = []


        bar_length = 30
        bar_prefix = 'Evaluando red:'
        bar_suffix = 'Completado '
        bar_end    = "\r"
        progress_end = "\n"

        if is_callback:
            bar_length = 15
            bar_prefix = 'Validation:'
            bar_suffix = ''
            #guardamos la posicion de comienzo del cursor para poder volver a ella imprimiendo bar_end
            print("\033[s", end='')
            bar_end    = "\033[u"
            progress_end = "\033[48C"

        if use_generators:

            printProgressBar(0, len(validation_data), prefix=bar_prefix,
                             suffix=bar_suffix+ '(' + self.C + str(psutil.virtual_memory()[2]) + self.B + '% RAM)',
                             length=bar_length, printEnd=bar_end, color=self.C, print_finish=progress_end)
            i = 0
            for data in validation_data:
                # separamos datos y etiquetas
                val_img, val_lab = data

                # realizamos las predicciones
                preds.append(model.predict(val_img))
                val_labels.append(val_lab)

                i = i+1
                printProgressBar(i, len(validation_data), prefix=bar_prefix,
                                 suffix=bar_suffix + '(' + self.C + str(psutil.virtual_memory()[2]) + self.B + '% RAM)',
                                 length=bar_length, printEnd=bar_end, color=self.C, print_finish=progress_end)
        else:
            # separamos datos y etiquetas
            val_imgs, val_labels = validation_data

            # realizamos las predicciones
            print("Predicting...")
            preds = model.predict(val_imgs)


        if len(preds) > 0:
            # contrastamos las predicciones con las etiquetas y calculamos el porcentaje de error
            diff = np.array(preds).flatten() - val_labels
            #percentDiff = (diff / val_labels) * 100
            #absPercentDiff = np.abs(percentDiff)
            absDiff = np.abs(diff)

            # calculamos la media de error y su desviacion tipica
            mean = np.mean(absDiff)
            std = np.std(absDiff)


            #establecemos el return
            ret = [mean, std]

            if extended_info:
                # añadimos la informacion estadistica extra
                ret.append(np.amax(absDiff))
                ret.append(np.amin(absDiff))
                ret.append(np.median(absDiff))
                ret.append(np.var(absDiff))


                #añadimos los ejemplos
                indx = np.random.randint(len(validation_data)-num_examples)
                ret.append([])
                for i in range(0, num_examples):
                    ret[-1].append([val_labels[indx + i], preds[indx + i]])


            return ret


    def show_plot(self, history, regression=False, max_y_value=-1):
        """ Muestra por pantalla una grafica con el historial del entrenamiento. """

        if regression:
            ent_loss = history.history['loss']
            val_loss = history.history['val_loss']

            Gepochs = range(1, len(ent_loss) + 1)

            plt.style.use('dark_background')
            fig, axs = plt.subplots(1)
            fig.suptitle('Loss & Accuracy')

            if max_y_value >= 0:
                axs.set_ylim(top=max_y_value)  # MAX_Y_LOSS

            axs.plot(Gepochs, ent_loss, 'lightcoral', label='Training Loss')
            axs.plot(Gepochs, val_loss, 'sandybrown', label='Test Loss')

            plt.xlabel('Epochs')
            axs.xaxis.set_major_locator(MaxNLocator(integer=True))
            axs.legend()

            plt.show()
        else:
            ent_loss = history.history['loss']
            val_loss = history.history['val_loss']
            ent_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            Gepochs = range(1, len(ent_loss) + 1)

            plt.style.use('dark_background')
            fig, axs = plt.subplots(2)
            fig.suptitle('Loss & Accuracy')

            if max_y_value >= 0:
                axs[0].set_ylim(top=max_y_value) # MAX_Y_LOSS


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


    def save_model(self, model, keyword, is_temp_file=False, quiet=False):
        """ Guarda el modelo y sus pesos. """

        iter_name = keyword

        model_path = os.path.join(self.parent_folder, "models")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if is_temp_file:
            model.save(
                model_path + "/" + keyword + ".h5")
        else:
            model.save(
                model_path + "/" + "modeloYpesos_" + iter_name + "_M" + str(self.ID_MODELO) + "_epochs" + str(self.epochs) + "_batch" + str(
                    self.batch_size) + ".h5")

        if not quiet:
            print(self.B + "Modelo " + self.C + "Guardado" + self.B + ".\n")


    def load_model(self, path, quiet=False):
        """ Carga un modelo de un fichero dada su ruta absoluta. """

        model = keras.models.load_model(path)
        if not quiet:
            print(self.B + "Cargando modelo:\n\n " + self.C + str(path) + self.B + "\n")
        return model
    
    
    def print_final_evaluation(self, model, val_gen, num_examples=1):
        # Evaluacion final
        print("\n\nEvaluacion final:\n")
        eva = self.evaluate_regression_model(model, val_gen, use_generators=True, extended_info=True, num_examples=num_examples)

        print()
        print(" - Media:      " + self.C + '%.2f' % eva[0] + self.B)
        print(" - Desviacion: " + self.C + '%.2f' % eva[1] + self.B)
        print(" - Maximo:     " + self.C + '%.2f' % eva[2] + self.B)
        print(" - Minimo:     " + self.C + '%.2f' % eva[3] + self.B)
        print(" - Mediana:    " + self.C + '%.2f' % eva[4] + self.B)
        print(" - Varianza:   " + self.C + '%.2f' % eva[5] + self.B)



        true_labels = ''
        preds = ''
        for par in eva[-1]:
            true_labels += '%.2f' % par[0] + '\t\t'
            preds       += '%.2f' % par[1] + '\t\t'

        #quitamos los tabuladores extra
        true_labels = true_labels[:-2]
        preds = preds[:-2]

        print("\nEjemplos de prediccion (true/" + self.C + "prediction" + self.B + "):\n" + true_labels + "\n" + self.C + preds + self.B + "\n")

