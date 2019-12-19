import glob
import os, json, cv2
from util import printProgressBar
from colored import fg, attr
import numpy as np

B = fg(15)
V = fg(118)

parent_folder = os.path.abspath(
    os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

dataset_dest_folder = os.path.join(parent_folder, "dataset")
raw_dataset_path = os.path.join(parent_folder, "dataset", "lambsm")
dest_path = os.path.join(dataset_dest_folder, "lambsmCUS")

# -------------  PARAMETROS  ------------
OVERRIDE_MAXVALUE = 3000  # -1 = auto detect



# ------------------------------------------------------------------------------------------------------


num_images = 0

# contamos las imagenes totales
for json_number, json_name in enumerate(glob.glob(os.path.join(raw_dataset_path, '*.json'))):

    f = open(json_name)
    j = json.load(f)

    i = 0
    for key in j:
        num_images = num_images + 1
        i += 1
    print("Numero de Json: " + V + str(json_number) + B)
    print("Imagenes en el Json: " + V + str(i) + B, end='\n\n')



print(attr(4) + "Imagenes Totales:" + attr(0) + " " + V + str(num_images) + B, end='\n\n')



i2 = 0
label_list = []

num_fly   = 0
num_lamb  = 0
num_wrong = 0
num_empty = 0
num_ret_img=0
# para todos los .json del raw_dataset
for json_number, json_name in enumerate(glob.glob(os.path.join(raw_dataset_path, '*.json'))):

    f = open(json_name)
    j = json.load(f)


    # estructuramos las imagenes del json
    for key in j:
        img_abs_path = raw_dataset_path + j[key]["path_depth"]
        if j[key]["label"] != "fly" and os.path.exists(img_abs_path):
            # cargamos cada imagen
            img = cv2.imread(img_abs_path, flags=cv2.IMREAD_ANYDEPTH)

            # preparamos la imagen para ser normalizada posteriormente (todo pixel que supere OVERRIDE_MAXVALUE se queda en OVERRIDE_MAXVALUE)
            if OVERRIDE_MAXVALUE != -1:
                img = np.clip(img, None, OVERRIDE_MAXVALUE)

            # le hacemos crop
            x = 38
            y = 102
            h = 230
            w = 510
            img = img[y:y + h, x:x + w]

            # invertimos la imagen
            # if OVERRIDE_MAXVALUE != -1:
            #   img = OVERRIDE_MAXVALUE-img

            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            cv2.imwrite(os.path.join(dest_path, str(num_ret_img) + ".png"), np.uint16(img))

            # guardamos la etiqueta
            label = j[key]["label"]
            if label == "lamb":
                num_lamb = num_lamb + 1
                label = [1, 0, 0]
            elif label == "empty":
                num_empty = num_empty + 1
                label = [0, 1, 0]
            elif label == "wrong":
                num_wrong = num_wrong + 1
                label = [0, 0, 1]
            else:
                print("hay alguna etiqueta mala: " + str(j[key]["label"]))


            label_list.append(label)

            num_ret_img += 1

        i2 += 1
        printProgressBar(i2, num_images, prefix='Estructurando JSON ' + str(json_number) + ':', suffix='Completado',length=100,color=118)




# guardamos las etiquetas
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
d = open(os.path.join(dest_path, "labels.npy"), "wb+")
np.save(d, np.array(label_list), allow_pickle=True)
d.close()

print("El dataset resultado se compone de: ")
print("lamb:  "  + V + str(num_lamb) + B)
print("empty: " + V + str(num_empty) + B)
print("wrong: " + V + str(num_wrong) + B)
print("fly:   "   + V + str(num_fly) + B)
print("TOTAL: "   + V + str(num_ret_img) + B)



