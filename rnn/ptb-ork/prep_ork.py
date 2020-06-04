""" Convert character sequences to "one-row-keyboard" sequences.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("datpath", type=str)
args = parser.parse_args()

alp2num = {"q": 1, "a": 1, "z": 1, "w": 2, "s": 2, "x": 2, "e": 3, "d": 3, "c": 3,
           "r": 4, "f": 4, "v": 4, "t": 5, "g": 5, "b": 5, "y": 6, "h": 6, "n": 6,
           "u": 7, "j": 7, "m": 7, "i": 8, "k": 8, "o": 9, "l": 9, "p": 0}

with open(args.datpath) as f:
    lines = [line.strip() for line in f]

for line in lines:
    chars_ = line.split()[:-1]
    chars = []
    ok = True
    for char_ in chars_:
        if char_ == "_":
            chars.append("_")
        elif char_ == "<unk>" or char_ == "<N>":
            ok = False
        else:
            chars.append(str(alp2num[char_]))
    if ok:
        print(" ".join(chars))
        print(" ".join(chars_))
