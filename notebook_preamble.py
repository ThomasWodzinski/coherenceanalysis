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

# Directory containing the data:
data_dir = Path('/content/gdrive/MyDrive/coherence/data/')
# Directory containing the useful hdf5 files (cleaned)
useful_dir = Path('/content/gdrive/MyDrive/coherence/data/useful/')
# Directory containing the background-subtracted hdf5 files
bgsubtracted_dir = Path('/content/gdrive/MyDrive/coherence/data/bgsubtracted/')
# Directory for temporary files:
scratch_dir = Path('/content/gdrive/MyDrive/coherence/data/scratch_cc/')
# Directory for local temporary files:
local_scratch_dir = Path("/content/coherence-analysis/scratch/")
import os
if os.path.isdir(local_scratch_dir) == False:
    if os.path.isdir(local_scratch_dir.parent.absolute()) == False:
        os.mkdir(local_scratch_dir.parent.absolute())    
    os.mkdir(local_scratch_dir)
#prebgsubtracted_dir
#bgsubtracted_dir = Path.joinpath('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/','bgsubtracted')

