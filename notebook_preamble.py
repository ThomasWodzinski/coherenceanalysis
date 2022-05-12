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
data_dir = Path('/content/gdrive/MyDrive/PhD/coherence/data/')
useful_dir = Path('/content/gdrive/MyDrive/PhD/coherence/data/useful/')
bgsubtracted_dir = Path('/content/gdrive/MyDrive/PhD/coherence/data/bgsubtracted/')
print(useful_dir)
scratch_dir = Path('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/')
#prebgsubtracted_dir
#bgsubtracted_dir = Path.joinpath('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/','bgsubtracted')

