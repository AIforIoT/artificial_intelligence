# System dependencies
import sys, os

# Packages import
from text2classification import classify
from speech2text import local, cloud, local_deep
from timeit import default_timer as timer

# Disable warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check if there is new wav file in the folder

# Pre-process the audio file for a better understanding

# Try local-based speech2text
t_i = timer()
text = local_deep.transcribe(sys.argv[1])
t_f = timer() - t_i
print("LOCAL DEEPSPEECH " + str(text) + " - " + str(t_f))

t_i = timer()
text = local.transcribe(sys.argv[1])
t_f = timer() - t_i
print("LOCAL SPHINX: " + str(text) + " - " + str(t_f))

# If sentence is not understandable - cloud speech2text
t_i = timer()
text = cloud.transcribe(sys.argv[1])
t_f = timer() - t_i
print("CLOUD: " + str(text) + " - " + str(t_f))

# Classify the words with text2classification
classify.classify_sentence(text)

# Output
