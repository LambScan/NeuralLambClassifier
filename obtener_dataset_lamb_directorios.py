import os, json, cv2, sys, psutil
from colored import fg
from shutil import copyfile
import numpy as np
B = fg(15)
V = fg(118)

parent_folder = os.path.abspath(os.path.dirname(__file__))

json_path = os.path.join(parent_folder, "dataset_labeled.json")
dataset_path = os.path.join(parent_folder, os.pardir,"TrainLamb", "TrainLamb")
dest_path = os.path.join(parent_folder, "dataDIR")

# -------------  PARAMETROS  ------------
OVERRIDE_MAXVALUE = 3000 # -1 = auto detect

TRAINING_PERCENT = 0.8


'''
Añade al dataset una copia de cada imagen rotada 180º (Data Aumentation)
'''
ROTATION = False



'''
Debido a que se elimina la categoria fly, se tomaran 0 imagenes de fly
'''
MAX_FLY = 0 # -1 = no limit

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




# funcion para rellenar con 0 (ahora no se usa. En lugar de eso se utiliza solo la linea que hace el recorte)
def cropWithZeros(in_array, x, y, h, w):
    in_array = np.array(in_array)
    shape = in_array.shape
    crop = in_array[y:y+h, x:x+w]
    bx = shape[0]-x-h
    by = shape[1]-y-w
    padding = ((x,bx),(y,by))
    return np.pad(crop, padding)



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

da = "Ninguno"
if ROTATION:
    da = "Rotacion 180º"

print(B + "\nData Aumentation usado: " + V + da + B)


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
    wigo = "  Debug: "
    no_es_fly = j[key]["label"] != "fly"
    if num_fly < MAX_FLY or no_es_fly:
        # cargamos cada imagen
        img = cv2.imread(dataset_path + j[key]["path_depth"], flags=cv2.IMREAD_ANYDEPTH)
        #actualizamos los contadores
        label = j[key]["label"]
        if   label == "lamb":
            num_lamb = num_lamb + 1
        elif label == "empty":
            num_empty = num_empty + 1
        elif label == "wrong":
            num_wrong = num_wrong + 1
        elif label == "fly":
            num_fly = num_fly + 1
        else:
            print("hay alguna etiqueta mala: " + str(j[key]["label"]))

        #preparamos la imagen para ser normalizada posteriormente (todo pixel que supere OVERRIDE_MAXVALUE se queda en OVERRIDE_MAXVALUE)
        if OVERRIDE_MAXVALUE != -1:
            img = np.clip(img, None, OVERRIDE_MAXVALUE)

        #le hacemos crop
        #img = cropWithZeros(img, 38, 102, 230, 510)
        x = 38
        y = 102
        h = 230
        w = 510
        img = img[y:y+h, x:x+w]

        #invertimos la imagen
        #img = OVERRIDE_MAXVALUE-img



        if i2/i < TRAINING_PERCENT:
            wigo = wigo + "T fly: " + str(num_fly)
            if no_es_fly:
                labeled_path = os.path.join(dest_path, "train", j[key]["label"])
            elif num_fly < (MAX_FLY*TRAINING_PERCENT):
                wigo = wigo + " <80%"
                labeled_path = os.path.join(dest_path, "train", j[key]["label"])
            else:
                wigo = wigo + " 80%"
                #cuando hayamos superado el TRAINING_PERCENT% de imagenes fly, las guardamos en validation hasta llegar a MAX_FLY
                labeled_path = os.path.join(dest_path, "validation", j[key]["label"])
        else:
            wigo = wigo + "V fly: " + str(num_fly) + "            "
            labeled_path = os.path.join(dest_path, "validation", j[key]["label"])

        if not os.path.exists(labeled_path):
            os.makedirs(labeled_path)
        cv2.imwrite(os.path.join(labeled_path, str(i2) + ".png"), np.uint16(img))

        if ROTATION:
            # guardamos una copia rotada 180º
            imgRot = np.rot90(np.rot90(img))
            cv2.imwrite(os.path.join(labeled_path, str(i2) + "_ROT.png"), np.uint16(imgRot))



        i2 = i2 + 1
        printProgressBar(i2, i, prefix = 'Copia y estructuracion del dataset:', suffix = 'Completado' + wigo, length = 100)

if ROTATION:
    (num_lamb, num_empty, num_wrong, num_fly) = (num_lamb*2, num_empty*2, num_wrong*2, num_fly*2)

print("El dataset resultado se compone de: ")
print("lamb: "  + V + str(num_lamb) + B)
print("empty: " + V + str(num_empty) + B)
print("wrong: " + V + str(num_wrong) + B)
print("fly: "   + V + str(num_fly) + B)

