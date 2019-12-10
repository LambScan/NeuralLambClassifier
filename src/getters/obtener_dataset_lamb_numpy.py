import os, json, cv2, sys, psutil
from colored import fg
from shutil import copyfile
import numpy as np
B = fg(15)
V = fg(118)

parent_folder = os.path.abspath(os.path.dirname(__file__))

json_path = os.path.join(parent_folder, "dataset_labeled.json")
dataset_path = os.path.join(parent_folder, os.pardir,"TrainLamb", "TrainLamb")
dest_path = os.path.join(parent_folder, "dataNP")

# -------------  PARAMETROS  ------------
OVERRIDE_MAXVALUE = 3000 # -1 = auto detect

MAX_FLY = 2000 # -1 = no limit

#------------- PROGRESS BAR  -------------------------------------------------------------------
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
    bar = fg(118) + fill * filledLength + fg(15) + "-" * (length - filledLength)
    print("\r%s |%s|" % (prefix, bar) + fg(118) + "%s" % (percent) + fg(15) + "%% %s" % (suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


 
    
    
    


#------------------------------------------------------------------------------------------------------


f = open(json_path)
j = json.load(f)


i  = 0
l1 = 0
l2 = 0
l3 = 0
l4 = 0

for key in j:
    i = i+1

    if j[key]["label"] == "lamb":
        l1 = l1 + 1
    elif j[key]["label"] == "empty":
        l2 = l2 + 1
    elif j[key]["label"] == "wrong":
        l3 = l3 + 1
    elif j[key]["label"] == "fly":
        l4 = l4 + 1
    else:
        print("hay alguna etiqueta mala: " + str(j[key]["label"]))


print("Imagenes etiquetadas: " + V + str(i) + B)
i = i - (l4-MAX_FLY)
print("Imagenes usadas: " + V + str(i) + B)
print("lamb: "  + V + str(l1) + B)
print("empty: " + V + str(l2) + B)
print("wrong: " + V + str(l3) + B)
print("fly: "   + V + str(l4) + B)


######################################   buscamos el valor maximo   ######################################
#valor maximo del dataset para poder normalizar
#solo buscamos si OVERRIDE_MAXVALUE = -1, si no se usa OVERRIDE_MAXVALUE
MAXIMUN = 0
num_fly = 0

if OVERRIDE_MAXVALUE == -1:

    printProgressBar(0, i, prefix = 'Buscando el maximo:', suffix = 'Completado', length = 100)
    i2 = 0
    for key in j:
        no_es_fly = j[key]["label"] != "fly"
        if num_fly < MAX_FLY or no_es_fly:
            # cargamos cada imagen
            img = cv2.imread(dataset_path + j[key]["path_depth"], flags=cv2.IMREAD_ANYDEPTH).flatten()

            for pixel in img:
                if pixel > MAXIMUN:
                    MAXIMUN = pixel


            i2 = i2 + 1
            printProgressBar(i2, i, prefix = 'Buscando el maximo:', suffix = 'Completado', length = 100)
            if not no_es_fly:
                num_fly = num_fly + 1


    print("\nValor maximo encontrado en el dataset: " + V + str(MAXIMUN) + B)
else:
    MAXIMUN = OVERRIDE_MAXVALUE

    print("\nValor maximo establecido manualmente: " + V + str(MAXIMUN) + B)

######################################   montamos el dataset   ######################################
printProgressBar(0, i, prefix = 'Copia y estructuracion del dataset:', suffix = 'Completado', length = 100)
i2 = 0
img_list = []
label_list = []
num_fly   = 0
num_lamb  = 0
num_wrong = 0
num_empty = 0
for key in j:
    no_es_fly = j[key]["label"] != "fly"
    if num_fly < MAX_FLY or no_es_fly:
        # cargamos cada imagen y cada respuesta y las guardamos ordenadas en un numpy
        img = cv2.imread(dataset_path + j[key]["path_depth"], flags=cv2.IMREAD_ANYDEPTH)

        #creamos la respuesta normalizada
        label = j[key]["label"]
        if   label == "lamb":
            label = [1,0,0,0]
            num_lamb = num_lamb + 1
        elif label == "empty":
            label = [0,1,0,0]
            num_empty = num_empty + 1
        elif label == "wrong":
            label = [0,0,1,0]
            num_wrong = num_wrong + 1
        elif label == "fly":
            label = [0,0,0,1]
            num_fly = num_fly + 1
        else:
            print("hay alguna etiqueta mala: " + str(j[key]["label"]))

        #normalizamos la imagen a [0,1]
        imgOut = img/MAXIMUN


        # agrupamos y guardamos
        res = np.array([imgOut, label])

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        d = open(os.path.join(dest_path, "ld_" + str(i2) + ".npy"), "wb+")
        np.save(d, res, allow_pickle = True)
        d.close()




        i2 = i2 + 1
        printProgressBar(i2, i, prefix = 'Copia y estructuracion del dataset:', suffix = 'Completado', length = 100)


print("El dataset resultado se compone de: ")
print("lamb: "  + V + str(num_lamb) + B)
print("empty: " + V + str(num_empty) + B)
print("wrong: " + V + str(num_wrong) + B)
print("fly: "   + V + str(num_fly) + B)


