# Artifial Inteligence module for IoT

This repository contains the source code for the AI module responsible for speech recognition and text classification.
The code in this repository contains the following libraries:
* Speech2text: AI module based on Google services or local speech recognition to transform speech (wav file) to text.
* Text2classification: AI module to classify the text recognized in 2 labels (probabilities):
	* Light (on/off)
	* Localization usage (yes/no)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The libraries you must have to run this Module are:

```
Python 3
Tensorflow
H5py
tqdm
librosa
```

### Usage

To compile and run the Nerual Network type on your command line:

```
python3 main.py "path-to-wav-file"
```