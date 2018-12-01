# Packages import
from artificial_inteligence.text2classification import classify
#from artificial_inteligence.speech2text import local, local_deep, cloud
from artificial_inteligence.speech2text import cloud

# Sender variables
thresold = 0.8
actions = ['Status' , 'Location', 'Light', 'Blind', 'Plug'];
status = ['L', 'H'];
error = 'L', 'L', 'none', 'error'

def detect_cloud(AUDIO_FILE):

    text = cloud.transcribe(AUDIO_FILE)
    #print("CLOUD: " + str(text))
    
    if text is not -1:
        return classiy_transcripted_text(text)
    else:
        return error


def detect_local(AUDIO_FILE):

    #text = local.transcribe(AUDIO_FILE)
    #print("LOCAL SPHINX: " + str(text))
    
    #text = local_deep.transcribe(AUDIO_FILE)
    #print("LOCAL DEEPSPEECH " + str(text))
    
    return classiy_transcripted_text(text)
    

def classiy_transcripted_text(text):
    
    # Classify the words with text2classification - [0] numpy bug
    probabilites = classify.classify_sentence(text)[0]

    # Get the state and location probability
    state = probabilites[actions.index('Status')]
    location = probabilites[actions.index('Location')]

    # Get the type of ESP32 refered
    for i in range(actions.index('Location') + 1, len(actions)):
        if probabilites[i] > thresold:
            action = actions[i]

    # Output
    return status[state.round(decimals=0).astype(int)], status[location.round(decimals=0).astype(int)], action, text
    