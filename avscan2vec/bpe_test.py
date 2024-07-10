import torch
import torch.nn as nn
import tiktoken

class BPEEmbedding(nn.Module):
    def __init__(self, D, vocab_size, PAD_idx):
        """
        BPE-style embeddings

        Arguments:
        D -- Embedding dimension
        vocab_size -- Size of BPE vocabulary
        PAD_idx -- Index of <PAD> in token vocabulary
        """
        super(BPEEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, D, padding_idx=PAD_idx)
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        
    def forward(self, text):
        """
        Generate BPE embeddings for the input text.

        Arguments:
        text -- Input text or batch of texts

        Returns:
        Tensor of embeddings with shape (batch_size, max_length, embedding_dim)
        """
        tokens = [self.tokenizer.encode(t) for t in text]
        max_length = max(len(t) for t in tokens)
        padded_tokens = [t + [self.embedding.padding_idx] * (max_length - len(t)) for t in tokens]
        token_tensor = torch.tensor(padded_tokens, dtype=torch.long)
        embeddings = self.embedding(token_tensor)
        return embeddings

# # Example usage
# D = 768  # Embedding dimension
# vocab_size = 50257  # Size of BPE vocabulary (example size, adjust as needed)
# PAD_idx = 0  # Index of <PAD> in token vocabulary

# bpe_embedding = BPEEmbedding(D, vocab_size, PAD_idx)

# # Example input
# text = ["Hello, world!", "This is a test.", "bingbong."]
# embeddings = bpe_embedding(text)
# print(embeddings.shape)