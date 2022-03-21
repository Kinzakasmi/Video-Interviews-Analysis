import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

def setupAll(list):
    for package in list:
        install(package)

if __name__ == '__main__':
    setupAll([
        'numpy',
        'pandas',
        'openpyxl',
        'tqdm',
        'ffmpeg',
        'pydub',
        'librosa',
        'praat-parselmouth',
        'lime',
        'spacy',
        'intel-openmp',
        'spacy-transformers'
    ])

    subprocess.call([sys.executable, "-m", "spacy", "download", "fr_dep_news_trf"])  
