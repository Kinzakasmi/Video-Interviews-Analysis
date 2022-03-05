import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


def setupAll(list):
    for package in list:
        install(package)
