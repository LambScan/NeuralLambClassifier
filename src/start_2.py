from colored import fg, attr
from cnn.nn import Model_constructor
import os


VERDE  = fg(118)
BLANCO = fg(15)
AZUL   = fg(45)
AZUL_CLARO = fg(159)


nombre_dataset = "pollosCUS"
target_size = (160,120,1)



parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

################################################       PARAMETROS       ################################################


ID_MODELO = 13

epochs = 10
batch_size = 100  # 1

loading_batch_size = 1
learning_rate = 0.1  # 0.00001

workers = 8  # hilos para el multiprocessing
RAM_PERCENT_LIMIT = 80  # %

# callbacks
paciencia = 200

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
colores["main"] = AZUL
colores["default"] = BLANCO



##################################################        MAIN        ##################################################

# creamos el modelo
MC = Model_constructor(parent_folder, parametros, colores)
model = MC.create_model()

print(AZUL_CLARO)
model.summary()
print(BLANCO)

# compilamos
model = MC.compile_model(model, regression=True)


# obtenemos los generadores
dataset_path = os.path.join(parent_folder, "dataset", nombre_dataset)
genetators = MC.get_generators(dataset_path, target_size, data_aumentation=False)


# entrenamos
history = MC.fit_model(model, genetators, True, regression=True, color=45, evaluate_each_epoch=True)

# Evaluacion final
print("\n\n" + attr(4) + "Evaluacion final:\n" + attr(0))
eva = MC.evaluate_regression_model(model, genetators[1], True)
print("\nmedia: " + colores["main"] + '%.2f' % eva[0] + colores["default"] + "%, desviacion: " + colores["main"] + '%.2f' % eva[1] + colores["default"] + "%")

# mostramos los resultados
MC.show_plot(history, True)
MC.save_model(model, "pollos")

