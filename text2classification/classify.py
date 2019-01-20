from os import path
import tensorflow as tf
from tensorflow import keras

from artificial_intelligence.text2classification.utils import codec

PATH = path.dirname(path.realpath(__file__))

# Load json and create model
json_file = open(PATH+'/bin/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights(PATH+"/bin/model.h5")
loaded_model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# Flask compatibility
loaded_model._make_predict_function()

def classify_sentence(sentence):

	# Process the sentence to a tensor
	sentence_data=[]
	sentence = codec.encode_sentence(sentence)

	if not len(sentence):
		return [[-1,-1,-1,-1,-1]]

	sentence_data.append(sentence)
	sentence_data = keras.preprocessing.sequence.pad_sequences(sentence_data, value=0, padding='post', maxlen=10)

	return loaded_model.predict(sentence_data)
