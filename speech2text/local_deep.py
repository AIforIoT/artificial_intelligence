import sys
from deepspeech import Model
import scipy.io.wavfile as wav

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.50

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 2.10

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

model = "../models/output_graph.pbmm"
alphabet = "../models/alphabet.txt"

LANGUAGE_MODEL = "../models/lm.binary"
TRIE = "../models/trie"

ds = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)

def transcribe(AUDIO_FILE):
    
    fs, audio = wav.read(AUDIO_FILE)
    
    if fs != 16000:
        print('Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(fs), file=sys.stderr)
        fs, audio = convert_samplerate(args.audio)
	
    processed_data = ds.stt(audio, fs)

    try:
        return processed_data
    except sr.UnknownValueError:
        return -1
    except sr.RequestError as e:
        return -1

def loadModel():
    
    ds.enableDecoderWithLM(alphabet, LANGUAGE_MODEL, TRIE, LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)
