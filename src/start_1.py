from colored import fg
from cnn.nn import Model_constructor
import os


VERDE  = fg(118)
BLANCO = fg(15)
AZUL   = fg(45)
AZUL_CLARO = fg(159)


nombre_dataset = "lambsmCUS"
target_size = (230, 510, 1)



parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

################################################       PARAMETROS       ################################################


ID_MODELO = 10

epochs = 80
batch_size = 4  # 1

loading_batch_size = 4
learning_rate = 0.00001  # 0.00001

workers = 8  # hilos para el multiprocessing
RAM_PERCENT_LIMIT = 80  # %

# callbacks
paciencia = 500

# porcentaje de uso del dataset en entrenamiento (el resto va a la validacion)
train_percent = 0.8  # 80%





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
colores["main"] = VERDE
colores["default"] = BLANCO



##################################################        MAIN        ##################################################

# creamos el modelo
MC = Model_constructor(parent_folder, parametros, colores)
model = MC.create_model()

print(colores["main"])
model.summary()
print(colores["default"])
print("#################################################################")


# compilamos
model = MC.compile_model(model)


# obtenemos los generadores
dataset_path = os.path.join(parent_folder, "dataset", nombre_dataset)
genetators = MC.get_generators(dataset_path, target_size, data_aumentation=True)


# entrenamos
history, model = MC.fit_model(model, genetators, use_generators=True, regression=False, color=118, evaluate_each_epoch=False, restore_best_weights=True)

# evaluamos la red y mostramos los resultados
MC.print_final_classif_evaluation(model, genetators[1], target_size=target_size, num_examples=20)

# mostramos los resultados
MC.show_plot(history, regression=False, just_save=True, save_name='lambSM_2')
MC.save_model(model, "SM2")

