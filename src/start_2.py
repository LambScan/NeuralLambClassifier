from colored import fg
from cnn.nn import Model_constructor
import os


VERDE  = fg(118)
BLANCO = fg(15)
AZUL   = fg(45)
AZUL_CLARO = fg(159)


nombre_dataset = "pollosCUS"
target_size = (320, 240, 1)



parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

################################################       PARAMETROS       ################################################


ID_MODELO = 13

epochs = 5
batch_size = 1  # 1

loading_batch_size = 1
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
colores["main"] = AZUL
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
model = MC.compile_model(model, regression=True)


# obtenemos los generadores
dataset_path = os.path.join(parent_folder, "dataset", nombre_dataset)
genetators = MC.get_generators(dataset_path, target_size, data_aumentation=False)


# entrenamos
history, model = MC.fit_model(model, genetators, True, regression=True, color=45, evaluate_each_epoch=True, restore_best_weights=True)

# evaluamos la red y mostramos los resultados
MC.print_final_regress_evaluation(model, genetators[1], num_examples=12)

# mostramos los resultados
MC.show_plot(history, regression=True, just_save=True, save_name='pollos')
MC.save_model(model, "pollos")



# todo -> comando para que se apague automaticamente al terminar la ejecucion (poner contraseña)
"""
sys_pass = ''
sudo = "echo \"" + sys_pass + "\" | sudo -S "  # es importante que haya un espacio despues de -S
os.system(sudo + "shutdown 0")
"""