""" Train character-level RNN(LSTM) LM.
"""

import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from corpus import CorpusForTrain
from rnn import RNNLM, repackage_hidden

VOCAB_PATH = "./data/vocab.txt"
BATCH_SIZE = 50
NUM_EPOCHS = 50
SEQ_LEN = 128
LOG_STEP = 100
SAVE_STEP = 5
LEARNING_RATE = 1e-3
#
VOCAB_SIZE = 30
EMB_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 3
TIE_WEIGHTS = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument("-train_path", type=str, default="./data/ptb.train.char.txt")
parser.add_argument("-log_path", type=str, default="./log/train.log")
parser.add_argument("-save_path", type=str, default="./checkpoints/rnnlm")
args = parser.parse_args()

corpus = CorpusForTrain(path=args.train_path, vocab_path=VOCAB_PATH)
corpus.batchify(BATCH_SIZE)

model = RNNLM(vocab_size=VOCAB_SIZE, emb_size=EMB_SIZE, hidden_size=HIDDEN_SIZE,
              num_layers=NUM_LAYERS, tie_weights=TIE_WEIGHTS)
model.to(device)
model.train()

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss()

if args.debug:
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
else:
    logging.basicConfig(filename=args.log_path,
                        format="%(asctime)s %(message)s",
                        level=logging.DEBUG)

for epoch in range(NUM_EPOCHS):
    step = 0
    loss_step = 0
    total_steps = corpus.num_batchs // SEQ_LEN
    zeros = torch.zeros(NUM_LAYERS,
                        BATCH_SIZE,
                        HIDDEN_SIZE, device=device)
    hidden = (zeros, zeros)
    while corpus.next_batch(SEQ_LEN):
        batched = corpus.get_batch(SEQ_LEN)
        inputs = batched[:, :-1].to(device)
        targets = batched[:, 1:].to(device)

        optimizer.zero_grad()
        preds, hidden = model(inputs, hidden)
        hidden = repackage_hidden(hidden)
        loss = criterion(preds.contiguous().view(-1, VOCAB_SIZE), targets.contiguous().view(-1))
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        loss_step += loss.item()
        if (step + 1) % LOG_STEP == 0:
            logging.info(f"epoch = {(epoch + 1):d} step = {(step + 1):d} / {total_steps:d} loss = {loss_step:.3f}")
            loss_step = 0
        step += 1
    
    if (epoch + 1) % SAVE_STEP == 0:
        save_path = args.save_path + f".network.epoch{epoch + 1}"
        logging.info(f"model is saved to: {save_path}")
        torch.save(model.state_dict(), save_path)
