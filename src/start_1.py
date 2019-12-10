from colored import fg
from cnn.nn import Model_constructor
import os


VERDE  = fg(118)
BLANCO = fg(15)
AZUL   = fg(45)
AZUL_CLARO = fg(159)

parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

################################################       PARAMETROS       ################################################


ID_MODELO = 11

epochs = 200
batch_size = 1  # 2

loading_batch_size = 1
learning_rate = 0.00001  # 0.00001

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
colores["main"] = VERDE
colores["default"] = BLANCO



##################################################        MAIN        ##################################################


MC = Model_constructor(parent_folder, parametros, colores)
model = MC.create_model()

print(AZUL_CLARO)
model.summary()
print(BLANCO)

model = MC.compile_model(model)
genetators = MC.get_generators()
history = MC.fit_model(model, genetators, True)

MC.show_plot(history)
MC.save_model(model, "Struct")

