import speech_recognition as sr

def transcribe(AUDIO_FILE):

	# Read the audio file 	
	r = sr.Recognizer()
	with sr.AudioFile(AUDIO_FILE) as source:
		audio = r.record(source)

	try:
		return r.recognize_sphinx(audio)
	except sr.UnknownValueError:
		return -1
	except sr.RequestError as e:
		return -1
