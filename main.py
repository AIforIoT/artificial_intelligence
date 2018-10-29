# System dependencies
import sys

# Packages import
from text2classification import classify
from speech2text import local, cloud, local_deep

# Check if there is new wav file in the folder

# Pre-process the audio file for a better understanding

# Try local-based speech2text
text = local.transcribe(sys.argv[1])
<<<<<<< HEAD
print("LOCAL SPHINX: " + str(text))

text = local_deep.transcribe(sys.argv[1])
print("LOCAL DEEPSPEECH " + str(text))
=======
print("LOCAL: " + str(text))
>>>>>>> 06802d9a5009777dbf53ef350ba6d90fa4e36970

# If sentence is not understandable - cloud speech2text
text = cloud.transcribe(sys.argv[1])
print("CLOUD: " + str(text))

# Classify the words with text2classification
classify.classify_sentence(text)

# Output
