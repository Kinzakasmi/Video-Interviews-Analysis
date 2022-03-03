import pandas as pd

def get_features(audios):

    feats = []
    for audio in audios: 
        spectral_features          = audio.spectral_features
        spectral_features['stat']  = spectral_features.index
        spectral_features['index'] = 0
        spectral_features          = pd.pivot_table(spectral_features,index='index',columns='stat',values=spectral_features.columns[:-2],aggfunc='first')
        
        features = pd.concat([audio.prosodic_features, spectral_features],axis=1)
        features['id']    = audio.email+'_'+str(audio.question)
        features = features.set_index('id')
        features['email'] = audio.email
        features['question'] = audio.question

        feats.append(features)

    feats = pd.concat(feats,axis=0)
    feats.columns = map(lambda c: '_'.join(c) if isinstance(c, tuple) else c, feats.columns)
    return feats

def get_scores(df_name):
    scores = [pd.read_excel(df_name,i) for i in range(4)]
    scores_all = pd.concat(scores)

    scores_all = scores_all.groupby(level=-1).mean()
    scores_all['email'] = scores[0]['email']
    return scores_all


def merge_scores_feats(scores,feats):
    df = feats.copy()
    df = df.merge(scores,on=['email','question']).set_index('email')

    feats  = df.drop(columns=['Q'+str(i+1) for i in range(21)])
    scores = df[['Q'+str(i+1) for i in range(21)]]

    feats  = feats.drop(columns=['question','bdd','time'])
    return feats, scores
