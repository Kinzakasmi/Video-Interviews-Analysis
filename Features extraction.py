# import library
import speech_recognition as sr
import wave
import pyaudio
import textdistance

filename = 'filename.wav'
referenceText = "Rappeur, slameur, écrivain, metteur en scène, D’ de Kabal arpente depuis près de vingt ans les scènes musicales, " \
                "les festivals, les théâtres et les ateliers d’écriture.Après s’être intéressé à la construction d’une identité dont " \
                "le mur porteur est l’histoire de l’esclavage colonial et ce qui en découle, après avoir questionné la figure de la " \
                "victime d’actes ou de propos racistes, il explore à présent un autre champ de pensée tout aussi proche de lui-même, " \
                "la notion de masculinité et les mécanismes de fabrication de celle-ci. Depuis 2015, D’ de Kabal écrit sur le sujet, " \
                "écoute, se documente, échange dans le cadre d’ateliers de parole qu’il appelle laboratoires de déconstruction et de " \
                "redéfinition du masculin par l’Art et le Sensible. « Il ne s’est pas agi de récolter des paroles d’hommes et d’en " \
                "faire un spectacle, ces laboratoires m’ont permis avant tout, de me rapprocher de moi-même et d’échanger sur des " \
                "sujets qui, jusque là, n’existaient dans aucun espace. »Pour donner corps à cette première création à La Colline, " \
                "D’ de Kabal convoque plusieurs figures marquantes tant par leur présence que par les instruments qu’ils /elles " \
                "utilisent (voix, machines, guitare électrique, corps, vidéo). Il questionne ainsi la construction de la masculinité " \
                "dans ses fondements, cette virilité portée comme une cuirasse, qui fabrique des dominants à la chaîne, porteurs de ce " \
                "qu’il nomme l’intégrisme masculin."

#Recording audio file

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = 'filename.wav'

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
print('file closed')



# Start Speech Recognition
# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Audio file as source
# listening the audio file and store in audio_text variable

def recognitionScore(text1, text2):
    return textdistance.overlap.normalized_similarity(text1, text2)


if __name__ == "__main__" :
    key = "YTR4JZSOXMEVIWYYX6MVC5HFKWWEJSWW"
    
    with sr.AudioFile(filename) as source:
        print('listening...')
        audio_text = r.record(source)
        print('Connecting to API...')

        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:

            # using google speech recognition

            text = r.recognize_wit(audio_text,key=key)
            print('Converting audio transcripts into text ...')
            print(text)
            print(recognitionScore(text, referenceText))
        except :
            raise
            print('Sorry, no speech detected.. run again...')

