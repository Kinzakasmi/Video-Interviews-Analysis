import pydub
import os

def split_questions(video_folder,df_startend,filename):
    df_startend = df_startend[df_startend['mail']==filename.split('.mp4',2)[0]]
    #Read audio
    audio = pydub.AudioSegment.from_file(video_folder+filename,'mp4')
    #split audio
    audios = [audio[s:e] for (s,e) in zip(df_startend['start'],df_startend['end'])]
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