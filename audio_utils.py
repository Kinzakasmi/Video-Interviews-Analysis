import pydub
import os

def load_audios(video_folder) :
    filenames = os.listdir(video_folder)
    audio = list(map(lambda f : pydub.AudioSegment.from_file(video_folder+f,'mp4'),filenames))
    return audios

def split_pauses(audio=None,filename=None,save=False) :
    if not audio and not filename :
        raise('You must provide either audio or filename') 
    
    if filename :
        audio = pydub.AudioSegment.from_file('videos/'+filename+'.mp4','mp4')
    
    audio_chunks = pydub.silence.split_on_silence(audio, 
        # must be silent for at least 1.5 second
        min_silence_len=1500,

        # consider it silent if quieter than -16 dBFS
        silence_thresh=-50
    )

    if save :
        print('There are',len(audio_chunks),'chunks')
        for i, chunk in enumerate(audio_chunks):
            out_file = 'splits/'+filename+"_chunk{0}.wav".format(i)
            print("Exporting", out_file)
            chunk.export(out_file, format="wav")