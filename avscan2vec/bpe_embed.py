import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

class BPEEmbedding(nn.Module):
    def __init__(self, D, vocab_size, pad_token_id):
        super(BPEEmbedding, self).__init__()
        self.D = D
        self.pad_token_id = pad_token_id

        # init bpe tokenize from tiktoken
        # self.tokenizer = tiktoken.get_encoding("o200k_base")

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, self.D)

    def forward(self, input_ids):
        # print(input_ids)

        # Ensure input_ids is a tensor
        input_ids = torch.tensor(input_ids).to(self.embedding.weight.device)
        
        # Get embeddings
        embeddings = self.embedding(input_ids)
        return embeddings
