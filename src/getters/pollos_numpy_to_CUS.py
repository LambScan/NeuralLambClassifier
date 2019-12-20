import os, cv2, psutil
from os import listdir
from colored import fg
import numpy as np
from util import printProgressBar

B = fg(15)
C = fg(45)

parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


dataset_path = os.path.join(parent_folder, "dataset", "pollos")
dest_path = os.path.join(parent_folder, "dataset", "pollosCUS")

# -------------  PARAMETROS  ------------
OVERRIDE_MAXVALUE = 1000  # IMAGE MAXVALUE


NORMALIZE = True


def normalize(in_value, min_value, max_value, a, b):
    """ Funcion que mapea un valor de entrada para que encaje en el rango [a,b] definido. """
    first_up   = in_value - min_value
    second_up  = b - a
    up    = first_up * second_up
    down  = max_value - min_value
    group = up / down
    ret   = a + group
    return ret


##################################################        Init        ##################################################


# obtenemos la lista de .npy
onlyfiles = [f for f in listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]


print(B + "Numero de elementos encontrados: " + C + str(len(onlyfiles)) + B, end='\n\n')


max_value = 0
min_value = 9999999
i = 0
print("\nNormalizar datos: " + C + str(NORMALIZE) + B + "\n")
if NORMALIZE:
    for fil in onlyfiles:
        # cargamos el archivo
        npy = np.load(os.path.join(dataset_path, str(fil)), allow_pickle=True)
        # separamos datos y etiquetas
        img, label = npy

        if label[0] < min_value:
            min_value = label[0]
        if label[0] > max_value:
            max_value = label[0]

        i = i + 1
        printProgressBar(i, len(onlyfiles), prefix='Buscando maximo y minimo:',
                         suffix='Completado (' + C + str(psutil.virtual_memory()[2]) + B + '% RAM)',
                         length=100, color=45)

print("\nMaximo: " + C + str(max_value) + B)
print("Minimo: " + C + str(min_value) + B + "\n")


#lista para guardar labels
label_list = []

#para cada .npy
i = 0
for fil in onlyfiles:
    # cargamos el archivo
    npy = np.load(os.path.join(dataset_path, str(fil)), allow_pickle=True)
    # separamos datos y etiquetas
    img, label = npy


    # normalizamos los labels
    if NORMALIZE:
        label[0] = normalize(label[0], min_value, max_value, -1, 1)

    # guardamos el label
    label_list.append(label[0]) # todo -> para la regresion de prueba guardamos solo la primera coordenada

    # todo -> resize para agilizar la red
    escala = 0.50
    img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_LANCZOS4)



    #guardamos la imagen
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    cv2.imwrite(os.path.join(dest_path, str(i) + ".png"), np.uint16(img*1000))

    i = i+1
    printProgressBar(i, len(onlyfiles), prefix='Estructurando dataset:',
                     suffix='Completado (' + C + str(psutil.virtual_memory()[2]) + B + '% RAM)',
                     length=100, color=45)



# guardamos las respuestas
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
d = open(os.path.join(dest_path, "labels.npy"), "wb+")
np.save(d, np.array(label_list), allow_pickle = True)
d.close()
