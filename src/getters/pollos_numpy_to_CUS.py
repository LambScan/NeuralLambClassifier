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
OVERRIDE_MAXVALUE = 1000  # -1 = auto detect





##################################################        Init        ##################################################


# obtenemos la lista de .npy
onlyfiles = [f for f in listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]


print(B + "Numero de elementos encontrados: " + C + str(len(onlyfiles)), end='\n\n')
printProgressBar(0, len(onlyfiles), prefix='Estructurando dataset:',
                 suffix='Completado (' + C + str(psutil.virtual_memory()[2]) + B + '% RAM)',
                 length=100, color=45)

#lista para guardar labels
label_list = []

#para cada .npy
i = 0
for fil in onlyfiles:
    # cargamos el archivo
    npy = np.load(os.path.join(dataset_path, str(fil)), allow_pickle=True)
    # separamos datos y etiquetas
    img, label = npy
    # guardamos el label
    label_list.append(label[0]) # todo -> para la regresion de prueba guardamos solo la primera coordenada

    # todo -> resize para agilizar la red
    escala = 0.25
    img = cv2.resize(img, None, fx=escala, fy=escala)

    #guardamos la imagen
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    cv2.imwrite(os.path.join(dest_path, str(i) + ".png"), np.uint16(img*1000))

    i = i+1
    printProgressBar(i, len(onlyfiles), prefix='Estructurando dataset:',
                     suffix='Completado (' + C + str(psutil.virtual_memory()[2]) + B + '% RAM)',
                     length=100, color=45)


#guardamos las respuestas
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
d = open(os.path.join(dest_path, "labels.npy"), "wb+")
np.save(d, np.array(label_list), allow_pickle = True)
d.close()
