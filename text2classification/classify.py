# OS dependencies
import os
import sys
from os import path

# Tensor flow dependencies
import tensorflow as tf
from tensorflow import keras

# train model
from text2classification.utils import codec

#sentence = "switch on the light"
PATH = path.dirname(path.realpath(__file__))

def classify_sentence(sentence):

	# Load json and create model
	json_file = open(PATH+'/bin/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = keras.models.model_from_json(loaded_model_json)

	# Load weights into new model
	loaded_model.load_weights(PATH+"/bin/model.h5")

	# evaluate loaded model
	loaded_model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

	# Process the sentence to a tensor
	sentence_data=[]
	sentence = codec.encode_sentence(sentence)

	sentence_data.append(sentence)
	sentence_data = keras.preprocessing.sequence.pad_sequences(sentence_data, value=0, padding='post', maxlen=10)
	
	sess = tf.Session()
	sess.close()

	return loaded_model.predict(sentence_data)
