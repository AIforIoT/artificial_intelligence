# System dependencies
import sys, os

# Packages import
from text2classification import classify
#from speech2text import local, cloud, local_deep
from speech2text import cloud

# Time purposes, will be removed
from timeit import default_timer as timer

# Disable warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Sender variables
thresold = 0.8
actions = ['Status' , 'Location', 'Light', 'Blind', 'Plug'];
status = ['L', 'H'];

# Check if there is new wav file in the folder

# Pre-process the audio file for a better understanding

# Try local-based speech2text
#t_i = timer()
#text = local_deep.transcribe(sys.argv[1])
#t_f = timer() - t_i
#print("LOCAL DEEPSPEECH " + str(text) + " - " + str(t_f))

#t_i = timer()
#text = local.transcribe(sys.argv[1])
#t_f = timer() - t_i
#print("LOCAL SPHINX: " + str(text) + " - " + str(t_f))

# If sentence is not understandable - cloud speech2text
t_i = timer()
text = cloud.transcribe(sys.argv[1])
t_f = timer() - t_i
print("CLOUD: " + str(text) + " - Time: " + str(t_f))

# Classify the words with text2classification - [0] numpy bug
probabilites = classify.classify_sentence(text)[0]

# Get the state and location probability
state = probabilites[actions.index('Status')]
location = probabilites[actions.index('Location')]

# Use this probability to map the index of the status
print('Status: ' + status[state.round(decimals=0).astype(int)])
print('Location: ' + status[location.round(decimals=0).astype(int)])

# Get the type of ESP32 refered
for i in range(actions.index('Location') + 1, len(actions)):
    if probabilites[i] > thresold:
        print("ESP32 referred is a " + actions[i])

# Output
