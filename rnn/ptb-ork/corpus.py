""" Provide batch for training LM.
"""

import torch

class CorpusForTrain:
    def __init__(self, path, vocab_path):
        self.tokens = self.tokenize(path)
        self.build_vocab(vocab_path)
        token_ids = self.convert_tokens_to_ids(self.tokens)
        self.token_ids = torch.tensor(token_ids)
        self.batch_id = 0
        # build_vocab
        self.token2id = None
        self.id2token = None
        # batchify
        self.num_batchs = None
        self.batched = None

    def tokenize(self, path):
        with open(path) as f:
            lines = [line.strip() for line in f]
        tokens = []
        for line in lines:
            tokens_ = line.split()
            tokens.extend(tokens_)
        return tokens
    
    def build_vocab(self, vocab_path):
        self.token2id = {}
        self.id2token = {}
        with open(vocab_path) as f:
            lines = [line.strip() for line in f]
        for token_id, token in enumerate(lines):
            self.token2id[token] = token_id
            self.id2token[token_id] = token
    
    def batchify(self, batch_size):
        self.num_batchs = self.token_ids.size(0) // batch_size
        # trim off remainders
        batched = self.token_ids.narrow(0, 0, self.num_batchs * batch_size)

        self.batched = batched.view(batch_size, -1)

    def get_batch(self, seq_len):
        # if batch does not remained, return None and prepare new one
        batch_id = self.batch_id
        self.batch_id += seq_len
        return self.batched[:, batch_id : (batch_id + seq_len)]
    
    def next_batch(self, seq_len):
        if self.batch_id + seq_len >= self.num_batchs:
            self.batch_id = 0
            return False
        return True
    
    def convert_token_to_id(self, token):
        return self.token2id[token]

    def convert_tokens_to_ids(self, tokens):
        return [self.convert_token_to_id(token) for token in tokens]
