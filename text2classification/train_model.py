import tensorflow as tf
from tensorflow import keras
import numpy as np

# Dicctionary functions
from artificial_intelligence.text2classification.utils import codec

# Variables for our NN
vocab_size = 100
max_sentence_words = 10
num_outputs = 5
num_hiddenNodes = 16
num_hiddenLayers = 2
iterations = 1000

def train(PATH):
	# Read every column of the csv data file
	csv_data = np.genfromtxt(PATH+'/data/train_dataset.csv', delimiter=';', dtype=(str))
	train_command, label_status, label_location, label_light, label_blind, label_plug = list(csv_data.T)

	# Generate a vector of each row
	train_command = np.asarray(train_command).flatten()
	
	label_status = np.asarray(label_status).flatten()
	label_location = np.asarray(label_location).flatten()
	label_light = np.asarray(label_light).flatten()
	label_blind = np.asarray(label_blind).flatten()
	label_plug = np.asarray(label_plug).flatten()

	# Generate the training matrix (words * commands)
	train_data=[]
	train_labels=[]

	# Replace the words by numbers to prepare for the NN
	for i in range(0, len(train_command)):
                train_data.append(codec.encode_sentence(str(train_command[i])))
                train_labels.append([label_status[i] , label_location[i], label_light[i], label_blind[i], label_plug[i]])

	# Convert the number array to tensors (input of NN)
	train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding='post', maxlen = max_sentence_words)
	train_labels = keras.preprocessing.sequence.pad_sequences(train_labels, value=0, padding='post', maxlen = num_outputs)

	# Create the NN with a hidden layer of 16 nodes and 2 output
	model = keras.Sequential()

	# Input layer
	model.add(keras.layers.Embedding(vocab_size, num_hiddenNodes, name="input"))

	# Hidden layers
	model.add(keras.layers.GlobalAveragePooling1D(name = "hiddenPool"))
	for i in range(0, num_hiddenLayers):
		model.add(keras.layers.Dense(num_hiddenNodes, activation=tf.nn.relu, name=("hidden_"+str(i))))

	# Output layer
	model.add(keras.layers.Dense(num_outputs, activation=tf.nn.sigmoid, name="output"))

	model.summary()

	# Compile the NN
	model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

	# Split the data into training and validation
	x_val = train_data[:10]
	partial_x_train = train_data[10:]

	y_val = train_labels[:10]
	partial_y_train = train_labels[10:]

	# Train the model (epoch = iteration) (verbose to see training = 1)
	history = model.fit(partial_x_train, partial_y_train, epochs=iterations, batch_size=512, validation_data=(x_val, y_val), verbose=1)

	# Serialize model to JSON
	model_json = model.to_json()
	with open(PATH+"/bin/model.json", "w") as json_file:
		json_file.write(model_json)

	# Serialize weights to HDF5
	model.save_weights(PATH+"/bin/model.h5")
	print("Saved model to disk")
