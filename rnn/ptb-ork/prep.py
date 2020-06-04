""" Prepare character sequences from origial corpus.
"""

import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("datpath", type=str)
args = parser.parse_args()

with open(args.datpath) as f:
    lines = [line.strip() for line in f]

for line in lines:
    chars = []
    words = line.split()
    for word in words:
        if word == "<unk>":
            chars.append("<unk>")
            chars.append("_")
            continue
        for char in word:
            if char == "N":
                chars.append("<N>")
            if re.match(r"[a-z]", char) is not None:
                chars.append(char)
        if chars and chars[-1] != "_":
            chars.append("_")
    
    chars = chars[:-1]  # remove last `_`
    chars.append("<eos>")
    print(" ".join(chars))
