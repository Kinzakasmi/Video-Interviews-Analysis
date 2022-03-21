# Video-interviews-analysis
## 1. Pour installer toutes les librairies, lancez la commande suivante: 
python -m setup

## 2. Pour ajouter une/des nouvelle.s vidéo.s à la base d'entrainement et re-entrainer:

Ajouter la video mp4: 

*Télécharger la photo dans ./videos

Ajouter les scores pour chaque question:

*Dans notes_entretiens_all.xlsx:
    Ajouter une ligne pour chaque question et indiquer la BDD, le numéro de la question et les temps de début de chaque question en secondes
    Ajouter ensuite les scores pour chaque question

Lancer l'entrainement:

*Lancer le notebook train.ipynb

Les questions sont les suivantes:  Attention, 1 est toujours 'négatif' et 4 toujours 'positif', même si la question est tournée autrement.  
Q1  : Did the participant maintain a good speaking rate (neither too slow nor too fast) ?
Q2  : Did the participant use few filler words (not many filler words) ? (Fillers)  
Q3  : Did the participant use gestures to emphasize what is being said ? (Gestures)  
Q4  : Was he/she convincing/persuasive in delivering ? (Convincing)  
Q5  : Did the candidate speak without making grammatical mistakes, using precise words, etc. ?
Q6  : Was the candidate polite (did they say hello, thank you, etc.) ?  
Q7  : Was the candidate wearing appropriate clothing ?  
Q8  : Did the participant maintain a good posture ? (neither too closed/formal nor too open/informal) (Posture)  
Q9  : Was the candidate honest ?  
Q10 : Did the candidate extented his answer by giving practical adequate examples and information ?  
Q11 : Was the candidate organized ? (was their speech organized or random)  
Q12 : Did the candidate evoke qualities of a leader (proper voice intonation, good posture, confident)  
Q13 : Did you feel that the candidate wasn't "stunned" by the questions ?    
Q14 : Did you feel that the candidate was motivated for the role ?  
Q15 : Was the participant enthusiastic in the interview (gestures, tone, smile) ?  
Q16 : Did the candidate give you a good first impression ?  
Q17 : Did you feel that the candidate wasn't too harsh on themselves or didn't give negative information ?  
Q18 : Did the candidate give the impression that they were confident on their skills, that they could do the job ?  
Q19 : Was the candidate not over confident (quick answers, a little violent, too assertive without concrete examples) ?  
Q20 : How sympathetic/warm the person is in the interview ? (Warmness)  
Q21 : Was the participant expressive (neither too blank or exaggerated) ? (Face Exp.)  

