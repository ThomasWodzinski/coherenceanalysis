# <codecell>
# for colab notebook

from pathlib import Path  # see https://docs.python.org/3/library/pathlib.html#basic-use

# ! git clone --branch develop https://github.com/ThomasWodzinski/coherence-analysis.git
! git clone https://github.com/ThomasWodzinski/coherence-analysis.git
%cd coherence-analysis/

## Define paths
# use data stored in own google drive location
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)

# Prerequiste: shared folder 'coherence' has to be added as a shortcut to own Google Drive

if os.path.isdir('/content/gdrive/MyDrive/coherence/data') == True:
    ! ln -s /content/gdrive/MyDrive/coherence/data/useful ./data/useful
    ! ln -s /content/gdrive/MyDrive/coherence/data/bgsubtracted ./data/bgsubtracted
else:    
    print('Please request access to the data folder located in Google Drive by contacting thomas.wodzinski@gmail.com')
