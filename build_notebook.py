# adpated from https://stackoverflow.com/a/35720002

from nbformat import v3, v4 

with open("notebook_preamble.py") as fpin:
    text1 = fpin.read()

with open("main.py") as fpin:
    text2_list = fpin.readlines()[24:]
    text2 = "".join([l for l in text2_list]) 
# text2 = '# <codecell>\n%load main.py'

text = text1 + '\n'

text += text2


nbook = v3.reads_py(text)
nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

jsonform = v4.writes(nbook) + "\n"
with open("coherenceanalysis.ipynb", "w") as fpout:
    fpout.write(jsonform)
