# OS dependencies
import os
import sys

# Tensor flow dependencies
import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

# Dicctionary functions
from codec import encode_sentence, decode_sentence

# Tensor flow session
sess = tf.Session()

# Load json and create model
json_file = open('bin/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("bin/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# Using a csv file for example of commands and generated phrases
test_command = pd.read_csv("data/test_dataset.csv",sep=";", usecols=['command'])
test_command = np.asarray(test_command).flatten()

test_data=[]

for x in range(0, len(test_command)):
	test_command[x]=encode_sentence(test_command[x])
	test_data.append(test_command[x])
	
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=0, padding='post', maxlen=10)

# Create a on x 10 command and off x 10 command and a random command
on_command = np.full(10,7)
off_command = np.full(10,8)
random_command = np.random.randint(30, size=10)

# Print the random command
print("Generated phrases: ")
print(decode_sentence(random_command))
print(decode_sentence(on_command))
print(decode_sentence(off_command))

test_data = np.append(test_data, [on_command], axis = 0)
test_data = np.append(test_data, [off_command], axis = 0)
test_data = np.append(test_data, [random_command], axis = 0)

# Show results
print(test_data)
print(loaded_model.predict(test_data))