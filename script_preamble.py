# <codecell>

from pathlib import Path  # see https://docs.python.org/3/library/pathlib.html#basic-use

## Define paths

data_dir = Path("g:/My Drive/PhD/coherence/data/")
useful_dir = Path("g:/My Drive/PhD/coherence/data/useful/")
bgsubtracted_dir = Path("g:/My Drive/PhD/coherence/data/bgsubtracted/")
print(useful_dir)
scratch_dir = Path("g:/My Drive/PhD/coherence/data/scratch_cc/")
# prebgsubtracted_dir
# bgsubtracted_dir = Path.joinpath('/content/gdrive/MyDrive/PhD/coherence/data/scratch_cc/','bgsubtracted')
