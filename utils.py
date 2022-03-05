import pandas as pd

def get_end_from_start(df_mail):
    end             = df_mail.iloc[1:,:]['start'].copy()
    end[len(end)+1] = -1
    df_mail['end']  = end.values
    return df_mail

def get_start_end_from_file(file):
    #Get start and end times
    df = pd.read_excel(file,1)
    df = df.rename(columns={'time':'start'})
    df['start'] = df['start']*1000 #in ms

    df = df.groupby('email').apply(get_end_from_start)
    return df
