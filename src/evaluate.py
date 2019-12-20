from colored import fg, attr
from cnn.nn import Model_constructor
import os


nombre_modelo  = "M10_95.5_val_accuracy.h5"
nombre_dataset = "lambsmCUS"
target_size = (230, 510, 1)



parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
model_path = os.path.join(parent_folder, "models", nombre_modelo)

################################################       PARAMETROS       ################################################


ID_MODELO = 14

epochs = 6
batch_size = 1  # 2

loading_batch_size = 1
learning_rate = 0.00001  # 0.00001

workers = 8  # hilos para el multiprocessing
RAM_PERCENT_LIMIT = 80  # %

# callbacks
paciencia = 200

# porcentaje de uso del dataset en entrenamiento (el resto va a la validacion)
train_percent = 0.8  # 80%


BLANCO = fg(15)
COLOR  = fg(201)


##################################################        Dicc        ##################################################
parametros = {}
parametros["id_modelo"] = ID_MODELO
parametros["epochs"] = epochs
parametros["batch_size"] = batch_size
parametros["loading_batch_size"] = loading_batch_size
parametros["learning_rate"] = learning_rate
parametros["workers"] = workers
parametros["ram_percent_limit"] = RAM_PERCENT_LIMIT
parametros["paciencia"] = paciencia
parametros["train_percent"] = train_percent

colores = {}
colores["main"] = COLOR
colores["default"] = BLANCO



##################################################        MAIN        ##################################################

# cargamos el modelo
MC = Model_constructor(parent_folder, parametros, colores)
model = MC.load_model(model_path)

# obtenemos los generadores
dataset_path = os.path.join(parent_folder, "dataset", nombre_dataset)
genetators = MC.get_generators(dataset_path, target_size, data_aumentation=False)


# evaluamos la red y mostramos los resultados
MC.print_final_classif_evaluation(model, genetators[1], target_size=target_size, num_examples=100)




