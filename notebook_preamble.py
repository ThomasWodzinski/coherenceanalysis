# <codecell>
# for colab notebook

from pathlib import Path  # see https://docs.python.org/3/library/pathlib.html#basic-use
import os.path

# ! git clone --branch develop https://github.com/ThomasWodzinski/coherenceanalysis.git
! git clone https://github.com/ThomasWodzinski/coherenceanalysis.git
%cd coherenceanalysis/

## Define paths
# use data stored in own google drive location
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)

# Prerequiste: request access to shared folder 'coherenceanalysis' and add as a shortcut to own Google Drive
coherenceanalysis_folder_url = 'https://drive.google.com/drive/folders/1-s3HWOnOqqi_Z5OYDH2k3n_GIKJFvY47?usp=sharing'

if os.path.isdir('/content/gdrive/MyDrive/coherenceanalysis/data') == True:
    ! ln -s /content/gdrive/MyDrive/coherenceanalysis/data/useful ./data/useful
    ! ln -s /content/gdrive/MyDrive/coherenceanalysis/data/bgsubtracted ./data/bgsubtracted
else:    
    print('Please request access to the data folder located in Google Drive.')
