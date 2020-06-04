""" Beam search and get accuracy on test set.
"""

import argparse
from operator import itemgetter
import torch
import numpy as np
from torch.nn.functional import log_softmax
from rnn import RNNLM, repackage_hidden

TEST_PATH = "./data/ptb.test.char.onerow.txt"
VOCAB_PATH = "./data/vocab.txt"
LOG_STEP = 50
#
VOCAB_SIZE = 30
EMB_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 3
TIE_WEIGHTS = True

parser = argparse.ArgumentParser()
parser.add_argument("-model_path", type=str)
parser.add_argument("-beam", type=int, default=3)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if args.cpu:
    device = "cpu"
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num2id = {"_": [0], "0": [19], "1": [20, 4, 29], "2": [26, 22, 27], "3": [8, 7, 6], "4": [21, 9, 25],
          "5": [23, 10, 5], "6": [28, 11, 17], "7": [24, 13, 16], "8": [12, 14], "9": [18, 15]}

with open(VOCAB_PATH) as f:
    lines = [line.strip() for line in f]
token2id = {}
for token_id, token in enumerate(lines):
    token2id[token] = token_id

model = RNNLM(vocab_size=VOCAB_SIZE, emb_size=EMB_SIZE, hidden_size=HIDDEN_SIZE,
              num_layers=NUM_LAYERS, tie_weights=TIE_WEIGHTS)
state_dict = torch.load(args.model_path, map_location=device)
model.load_state_dict(state_dict)
print(f"load model from {args.model_path}")
model.to(device)
model.eval()

with open(TEST_PATH) as f:
    lines = [line.strip() for line in f]
inputs_data, labels_data = [], []
for line in lines[0::2]:
    numbers = line.split()
    inputs_data.append(numbers)
for line in lines[1::2]:
    alphabets = line.split()
    ids = [int(token2id[a]) for a in alphabets]
    labels_data.append(ids)

corr = 0
total = 0

for step, (inputs, labels) in enumerate(zip(inputs_data, labels_data)):
    zeros = (torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, device=device), torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, device=device))
    beam_paths = [(0.0, [1], zeros)]  # (score, seq, hidden)
    # beam search
    for inp in inputs:
        curr_beam_paths = []
        for beam_path in beam_paths:
            score, seq, hidden = beam_path
            last = torch.tensor([[seq[-1]]], device=device)
            preds, hidden = model(last, hidden)
            logprobs = log_softmax(preds, dim=2)

            cand_token_ids = num2id[inp]
            for cand_token_id in cand_token_ids:
                new_score = score + logprobs[0, 0, cand_token_id]
                new_seq = seq + [cand_token_id]
                curr_beam_paths.append((new_score, new_seq, hidden))

        curr_beam_paths_sorted = sorted(curr_beam_paths, key=itemgetter(0), reverse=True)
        beam_paths = curr_beam_paths_sorted[:args.beam]

    best_score, best_seq, _ = beam_paths[0]
    seq_pred = np.array(best_seq[1:], dtype=int)
    seq_true = np.array(labels, dtype=int)
    assert len(seq_pred) == len(seq_true)
    corr += np.sum(seq_pred == seq_true)
    total += len(seq_true)

    # logging
    if (step + 1) % LOG_STEP == 0:
        print(f"step = {(step + 1):d} / {len(inputs_data):d} - accuracy: {(corr/total):.5f}")

print(f"accuracy: {(corr/total):.5f}")
