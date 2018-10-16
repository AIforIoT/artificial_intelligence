import json
from os import path

DICT_FILE = path.join(path.dirname(path.realpath(__file__)), "../data/iot_dictionary.json")

# Open the dictionary file
with open(DICT_FILE) as f:
	word_dict = json.loads(f.read())

# Methods for encoding the word
def encode_word(word):
	if word not in word_dict:
	    return None
	return word_dict[word]

def encode_sentence(text):
	result = []
	arr = text.split(" ")
	#arr = keras.preprocessing.text.text_to_word_sequence(text, lower=True, split=" ")
	for word in arr:
	    w = encode_word(word.lower())
	    if w is not None:
	        result.append(w)
	return result

def decode_word(number):
	for word in word_dict:
		if word_dict[word] == number:
			return word
	return None

def decode_sentence(numbers):
	result = []
	for number in numbers:
		word = decode_word(number)
		if word is not None:
			result.append(word)
	return result
