# Artifial Inteligence module for IoT

This repository contains the source code for the AI module responsible for speech recognition and text classification.
The code in this repository contains the following libraries:
* Speech2text: AI module based on Google services or local speech recognition to transform speech (wav file) to text.
* Text2classification: AI module to classify the text recognized in 3 labels (probabilities):
	* Status (H/L)
	* Localization usage (H/L)
	* Device: Light, Blind or Plug

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The libraries you must have to run this Module are:

```
Python 3
Tensorflow
DeepSpeech
H5py
tqdm
librosa
```

### Usage

This package can be used by importing the library:
```
from artificial_inteligence import detect_command as dc
```

To recognize the 3 labels and the recognized text:
```
status, location, device, transcription = dc.detect_cloud(AUDIO_FILE)
#status, location, device, transcription = dc.detect_local(AUDIO_FILE)
```

word = dk.detect(AUDIO_FILE)
