# System dependencies
import sys

# Packages import
from text2classification import classify
from speech2text import local, cloud

# Check if there is new wav file in the folder

# Pre-process the audio file for a better understanding

# Try local-based speech2text
text = local.transcribe(sys.argv[1])
print("LOCAL: " + str(text))

# If sentence is not understandable - cloud speech2text
text = cloud.transcribe(sys.argv[1])
print("CLOUD: " + str(text))

# Classify the words with text2classification
classify.classify_sentence(text)

# Output
