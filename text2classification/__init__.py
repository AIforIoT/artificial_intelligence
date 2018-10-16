from os import path
from text2classification import train_model

PATH = path.dirname(path.realpath(__file__))

# Check if there is a trained model
if not (path.isfile(PATH+"/bin/model.json") and path.isfile(PATH+"/bin/model.h5")):
	train_model.train()
	print("text2speech: Training the model")
else:
	print("text2speech: Model loaded")
