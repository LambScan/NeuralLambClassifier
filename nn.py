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


'''
Tensorflow Message Debug Level:

    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from os import listdir
from os.path import isfile, join
from modelos import cargarModelo

V = fg(118)
B = fg(15)

# PROGRESS BAR 
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = V + fill * filledLength + B + "-" * (length - filledLength)
    print("\r%s |%s|" % (prefix, bar) + V + "%s" % (percent) + B + "%% %s" % (suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



# funcion para rellenar con 0
def cropWithZeros(in_array, x, y, h, w):
    in_array = np.array(in_array)
    shape = in_array.shape
    crop = in_array[y:y+h, x:x+w]
    bx = shape[0]-x-h
    by = shape[1]-y-w
    padding = ((x,bx),(y,by))
    return np.pad(crop, padding)



# clase generator personalizada
class My_Custom_Generator(keras.utils.Sequence) :

    def __init__(self, parent_folder, image_filenames, labels, batch_size) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.parent_folder = parent_folder


    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)


    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        #print(batch_x)

        return np.array([cv2.imread(os.path.join(self.parent_folder, str(file_name) + ".png"), cv2.IMREAD_ANYDEPTH).reshape(230, 510, 1) for file_name in batch_x]), np.array(batch_y)



print("\n\n\n" + B + "============================================================================================================")
print(V + "============================================================================================================" + B)
print("============================================================================================================\n")

print("Keras version: " + V + str(keras.__version__) + B, end='\n\n\n')




# --------------------------------- PARAMETROS ---------------------------------

ID_MODELO = 8
USAR_GENERADORES = True


epochs = 50
batch_size = 1   #2

loading_batch_size  = 10
learning_rate = 0.00001  # 0.00001



workers = 8 # hilos para el multiprocessing
RAM_PERCENT_LIMIT = 80 #%


#callbacks
paciencia = 200

# porcentaje de uso del dataset en entrenamiento (el resto va a la validacion)
train_percent = 0.8  # 80%

# valor maximo a visualizar en la grafica de perdida
MAX_Y_LOSS = 1

# arquitectura de la red
kernel_size_3  = (3, 3)
kernel_size_5  = (5, 5)
kernel_size_7  = (7, 7)
kernel_size_9  = (9, 9)
kernel_size_11 = (11, 11)
pool_size = (2, 2)


# ================================================     Montamos la Red     ================================================
print(B + "\n\n=====================  " + V + "Montando la red" + B + "  =====================\n")
print(fg(245)) # material innecesario en gris

model = cargarModelo(ID_MODELO)


print(fg(159))
model.summary()
print(B)



# ================================================     Compilamos la Red     ================================================
print(B + "\n\n=====================  " + V + "Compilando" + B + "  =====================\n")

adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])


print(B + "Usando Modelo " + V + str(ID_MODELO) + B)



# ================================================     Preparamos los Datos     ================================================
print(B + "\n\n=====================  " + V + "Preparando los datos" + B + "  =====================\n")


parent_folder = os.path.abspath(os.path.dirname(__file__))


if USAR_GENERADORES:

    #### dataset iterativo ####
    '''
    train_folder = os.path.join(parent_folder, "dataDIR", "train")
    validation_folder = os.path.join(parent_folder, "dataDIR", "validation")
    datagen = ImageDataGenerator()


    #  training dataset
    train_it = datagen.flow_from_directory(train_folder,
                                            class_mode='categorical',
                                            batch_size=loading_batch_size,
                                            target_size=(230, 510),
                                            color_mode='grayscale')

    #  validation dataset
    val_it = datagen.flow_from_directory(validation_folder,
                                            class_mode='categorical',
                                            batch_size=loading_batch_size,
                                            target_size=(230, 510),
                                            color_mode='grayscale')


    '''

    #obtenemos las listas
    label_numpy_path = os.path.join(parent_folder, "dataCUS", "labels.npy")
    data_path = os.path.join(parent_folder, "dataCUS")


    label_list   = np.load(label_numpy_path, allow_pickle=True)
    img_names_list = np.arange(0, len(label_list), 1)



    frontera = int(round(len(img_names_list)*train_percent, 0))
    train_img_names = np.array(img_names_list[:frontera])
    train_labels = np.array(label_list[:frontera])
    val_img_names = np.array(img_names_list[frontera:])
    val_labels = np.array(label_list[frontera:])

    #creamos los generadores
    train_it = My_Custom_Generator(data_path, train_img_names, train_labels, loading_batch_size)

    val_it   = My_Custom_Generator(data_path, val_img_names,   val_labels,   loading_batch_size)


    print("------------------------------------------")

    num_train = train_img_names.size
    num_val = val_img_names.size

    '''
    num_train = 0
    for root, dirs, files in os.walk(train_folder):
        num_train += len(files)

    num_val = 0
    for root, dirs, files in os.walk(validation_folder):
        num_val += len(files)
    '''

    print(B + "Imagenes de entrenamiento " + V + str(num_train) + B)
    print(B + "Imagenes de validacion " + V + str(num_val) + B)
    print("------------------------------------------")

    # numero de imagenes usadas en cada batch (batch_size * stepsXepoch debe ser igual al numero de imagenes de entrenamiento)
    stepsXepoch = num_train//batch_size
    val_steps   = num_val//batch_size



else:
    #### dataset tradicional ####


    parent_folder = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(parent_folder, "dataNP")

    onlyfiles = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]

    all_data     = []
    all_label    = []


    print(B + "Numero de elementos encontrados: " + V + str(len(onlyfiles)), end='\n\n')
    printProgressBar(0, len(onlyfiles), prefix = 'Carga del dataset:', suffix = 'Completado (' + V + str(psutil.virtual_memory()[2]) + B + '% RAM)', length = 100)
    i = 0
    for fil in onlyfiles:

        # cargamos el archivo
        l = np.load(os.path.join(data_folder,str(fil)), allow_pickle=True)
        #separamos datos y etiquetas
        l = l.tolist()
        all_data.append (cropWithZeros(l[0], 38, 102, 230, 503).reshape(480,640,1))
        all_label.append(l[1])
        #progress bar
        i = i+1
        printProgressBar(i, len(onlyfiles), prefix = 'Carga del dataset:', suffix = 'Completado (' + V + str(psutil.virtual_memory()[2]) + B + '% RAM)', length = 100)
        if psutil.virtual_memory()[2] > RAM_PERCENT_LIMIT:
            break

    print(B)



    # dataset preparado
    frontera = int(round(len(all_data)*train_percent, 0))


    train_images = np.array(all_data[:frontera])
    train_labels = np.array(all_label[:frontera])


    test_images = np.array(all_data[frontera:])
    test_labels = np.array(all_label[frontera:])




    print("\n\n==============================================================================\n\nEl dataset se compone de " + V + str(len(all_data)) + B + " elementos."
           + "\nSe utilizara el " + V + str(round(train_percent*100, 0)) + B + "% del dataset en el entrenamiento. (" + V + str(len(train_labels))  + B + " elementos)"
           + "\nEl otro " + V + str(round((1-train_percent)*100, 0)) + B + "% se utilizara en la validacion del modelo. (" + V + str(len(test_labels)) + B + " elementos)"
           + "\n\n==============================================================================\n\n")


# ================================================     Entrenamos la Red     ================================================
print(B + "\n\n=====================  " + V + "Entrenando la red" + B + "  =====================\n")

#TensorBoard
board_path = os.path.join(parent_folder, "last_exec_data")

#Callbacks para el entrenamiento
if os.path.exists(board_path):
    shutil.rmtree(board_path)
calls = [
          keras.callbacks.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=paciencia, verbose=0, mode='auto', baseline=None, restore_best_weights=True),
          keras.callbacks.callbacks.LambdaCallback(on_epoch_begin=lambda e,l:print(fg(np.random.randint(130,232))), on_epoch_end=lambda e,l:print(fg(248)), on_batch_begin=None, on_batch_end=None,  on_train_begin=None, on_train_end=None),
          keras.callbacks.TensorBoard(log_dir=board_path, batch_size=batch_size)]

history = []

if USAR_GENERADORES:
    # entrenamiento del modelo iterativo
    history = model.fit_generator(train_it,
                                    steps_per_epoch=stepsXepoch,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=val_it,
                                    validation_steps=val_steps,
                                    use_multiprocessing=True,
                                    workers = workers,
                                    shuffle=True,
                    callbacks=calls)
else:


    # entrenamiento tradicional
    history=model.fit(train_images, train_labels,
              batch_size=batch_size,
              epochs=epochs,validation_data=(test_images, test_labels),
              verbose=1, use_multiprocessing=True, callbacks=calls,workers = workers)




# ================================================     Metricas y Graficas     ================================================
print(B + "\n\n=====================  " + V + "Mostrando los resultados" + B + "  =====================\n")

ent_loss = history.history['loss']
val_loss = history.history['val_loss']
ent_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

Gepochs = range(1, len(ent_loss) + 1)


plt.style.use('dark_background')
fig, axs = plt.subplots(2)
fig.suptitle('Loss & Accuracy')

axs[0].set_ylim(top=MAX_Y_LOSS)

axs[0].plot(Gepochs, ent_loss, 'lightcoral',  label='Training Loss')
axs[0].plot(Gepochs, val_loss, 'sandybrown',  label='Test Loss')
axs[1].plot(Gepochs, ent_acc,  'limegreen',        label='Training Accuracy')
axs[1].plot(Gepochs, val_acc,  'greenyellow', label='Test Accuracy')
plt.xlabel('Epochs')
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].legend()
axs[1].legend()

plt.show()


# ================================================     Guardamos el modelo     ================================================
print(B + "\n\n=====================  " + V + "Guardamos el modelo" + B + "  =====================\n")

iter_name = ""
if USAR_GENERADORES:
    iter_name = "DIR"
else:
    iter_name = "NP"


model_path = os.path.join(parent_folder, "modelos")
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save(model_path + "/" + "modeloYpesos_" + iter_name + "_M" + str(ID_MODELO) + "_epochs" + str(epochs) + "_batch" + str(batch_size) + ".h5")







