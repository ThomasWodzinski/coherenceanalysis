
with open("script_preamble.py") as fpin:
    text1 = fpin.read()

with open("main.py") as fpin:
    text2 = fpin.read()

text = text1 + '\n'
text += text2

with open("coherenceanalysis.py", "w") as fpout:
    fpout.write(text)
