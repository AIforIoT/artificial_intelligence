from os import path, environ
from artificial_inteligence.text2classification import train_model

# Variable for CPU Deep learning
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH = path.dirname(path.realpath(__file__))

# Check if there is a trained model
if not (path.isfile(PATH+"/bin/model.json") and path.isfile(PATH+"/bin/model.h5")):
    train_model.train(PATH)
