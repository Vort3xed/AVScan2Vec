import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveLogSoftmaxWithLoss

# remove, this is old characterbert implementation
from char_embed import CharCNN
from bpe_embed import BPEEmbedding

# from bpe_test import BPEEmbedding

from globalvars import *
import tiktoken
import sys


class PositionalEmbedding(nn.Module):

    # Original CharacterBERT
    # def __init__(self, A, L, D, n_chars, max_chars, PAD_idx):
    #     """CharacterBERT-style embeddings

    #     Arguments:
    #     A -- The number of AV products
    #     L -- Number of tokens per label
    #     D -- Embedding dimension
    #     n_chars -- Size of character dataset
    #     max_chars -- Max number of characters in token
    #     PAD_idx -- Index of <PAD> in token vocabulary
    #     """

    #     super(PositionalEmbedding, self).__init__()
    #     self.token_embd = CharCNN(D, n_chars, max_chars, PAD_idx)
    #     self.av_embd = nn.Embedding(A+1, D)
    #     self.pos_embd = nn.Embedding(A*L+1, D)
    #     self.layer_norm = nn.LayerNorm(D)

    #     positions = torch.arange(A*L+1, dtype=torch.long).reshape(1, -1) # (1, A*L+1)
    #     avs = torch.arange(A+1, dtype=torch.long) # (A+1)
    #     avs = avs.repeat(L)[L-1:].reshape(1, -1) # (1, A*L+1)
    #     self.register_buffer("positions", positions)
    #     self.register_buffer("avs", avs)

    #BPE implementation with BPE class
    # <CHANGE>
    def __init__(self, A, L, D, vocab_size, max_chars, PAD_idx):
        """BPE-style embeddings

        Arguments:
        A -- The number of AV products
        L -- Number of tokens per label
        D -- Embedding dimension
        vocab_size -- Size of BPE vocabulary
        max_chars -- Max number of characters in token
        PAD_idx -- Index of <PAD> in token vocabulary
        """

        # A: 89, L: 35, D: 768, vocab_size: 200000, PAD_idx: 0

        # super(PositionalEmbedding, self).__init__()
        # self.token_embd = BPEEmbedding(D, vocab_size, PAD_idx)
        # self.av_embd = nn.Embedding(A+1, D)
        # self.pos_embd = nn.Embedding(A*L+1, D)
        # self.layer_norm = nn.LayerNorm(D)
        # self.tokenizer = tiktoken.get_encoding("o200k_base")

        super(PositionalEmbedding, self).__init__()
        # self.token_embd = nn.Embedding(vocab_size, D)

        # 35000 should be reduced to the actual size of the vocabulary, wasting memory
        # self.token_embd = nn.Embedding(35000, D)
        print(f"Vocab size in positional embedding class: {vocab_size}")
        self.token_embd = nn.Embedding(8192, D)

        self.av_embd = nn.Embedding(A, D)
        self.pos_embd = nn.Embedding(L, D)
        self.layer_norm = nn.LayerNorm(D)
        self.tokenizer = tiktoken.get_encoding("o200k_base")

        positions = torch.arange(L, dtype=torch.long).reshape(1, L) # (1, L)
        positions = positions.repeat(A, 1).reshape(1, -1) # (1, A*L)
        # avs = torch.arange(A*L, dtype=torch.long).reshape(1, -1) # (1, A*L)

        avs = torch.arange(A, dtype=torch.long).reshape(A, 1) # (A)
        avs = avs.repeat(1, L).reshape(1, -1) # (1, A*L)

        # positions = torch.arange(A*L+1, dtype=torch.long).reshape(1, -1) # (1, A*L+1)
        # avs = torch.arange(A+1, dtype=torch.long) # (A+1)
        # avs = avs.repeat(L)[L-1:].reshape(1, -1) # (1, A*L+1)
        #why do we splice out the first L-1 elements?

        self.register_buffer("positions", positions)
        self.register_buffer("avs", avs)

        self.A = A
        self.L = L
        self.vocab_size = vocab_size

    # BPE implementation
    # def __init__(self, A, L, D, vocab_size, PAD_idx, model_name=None):
    #     """BPE-style embeddings using tiktoken with a built-in tokenizer

    #     Arguments:
    #     A -- The number of AV products
    #     L -- Number of tokens per label
    #     D -- Embedding dimension
    #     vocab_size -- Size of BPE vocabulary
    #     PAD_idx -- Index of <PAD> in token vocabulary
    #     model_name -- BPE tokenizer model (o200k_base)
    #     """
        
    #     super(PositionalEmbedding, self).__init__()
    #     self.D = D
    #     self.PAD_idx = PAD_idx

    #     # Initialize BPE tokenizer
    #     self.tokenizer = tiktoken.get_encoding("o200k_base")

    #     self.bpe_embd = nn.Embedding(vocab_size, D)
    #     self.av_embd = nn.Embedding(A + 1, D)
    #     self.pos_embd = nn.Embedding(A * L + 1, D)
    #     self.layer_norm = nn.LayerNorm(D)

    #     positions = torch.arange(A * L + 1, dtype=torch.long).reshape(1, -1)  # (1, A*L+1)
    #     avs = torch.arange(A + 1, dtype=torch.long)  # (A+1)
    #     avs = avs.repeat(L)[L - 1:].reshape(1, -1)  # (1, A*L+1)
    #     self.register_buffer("positions", positions)
    #     self.register_buffer("avs", avs)

    # CharacterBERT forward function for model ALSO WORKS FOR BPE
    def forward(self, X_scan):

        # Get batch size
        B = X_scan.shape[0]//self.A

        # Repeat positions and avs B times
        pos = self.positions.repeat(B, 1)
        avs = self.avs.repeat(B, 1)

        # X_scan_reshaped = X_scan.view(B * self.A, -1)
        av_reshaped = avs.view(B * self.A, -1)
        pos_reshaped = pos.view(B * self.A, -1)

        # print(f"X_scan size: {X_scan.size()}")
        # print(f"av_reshaped size: {av_reshaped.size()}")
        # print(f"pos_reshaped size: {pos_reshaped.size()}")

        # print(self.vocab_size)
        # print(torch.max(X_scan))
        # exit(0)
        # X_scan_reshaped = X_scan.view(B * self.A, -1)
    
        X_scan_embd = self.token_embd(X_scan) 
        av_embd = self.av_embd(av_reshaped)
        pos_embd = self.pos_embd(pos_reshaped)


        # Embed token
        # print(f"X_scan size (post embd): {X_scan_embd.size()}") # (B, A*L+1, max_chars, D)
        # print(f"avs size (post embd): {av_embd.size()}") # (B, A*L+1, D)
        # print(f"pos size (post embd): {pos_embd.size()}") # (B, A*L+1, D)

        token_embd = X_scan_embd + av_embd + pos_embd
        token_embd = self.layer_norm(token_embd)
        # print(f"token_embd size: {token_embd.size()}") # (B, A*L+1, D)
        # exit(0)
        return token_embd

    # basic bpe implementaion?
    # def forward(self, X_scan):

    #     # Get batch size
    #     B = X_scan.shape[0]

    #     print(self.positions.size())
    #     print(self.avs.size())

    #     # Repeat positions and avs B times
    #     pos = self.positions.repeat(B, 1)
    #     avs = self.avs.repeat(B, 1)

    #     print(pos.size())
    #     print(avs.size())

    #     # pos = pos.repeat(1, 1, 20)
    #     # avs = avs.repeat(1, 1, 20)

    #     # print(pos.size())
    #     # print(avs.size())

    #     # bpe_embd = self.bpe_embd(X_scan).view(100, 624, -1)
    #     bpe_embd = self.bpe_embd(X_scan).sum(dim=2)
    #     av_embd = self.av_embd(avs)
    #     pos_embd = self.pos_embd(pos)

    #     # av_embd = av_embd.repeat(1,1,20)
    #     # pos_embd = pos_embd.repeat(1,1,20)

    #     print(f"BPE embedding size: {bpe_embd.size()}")
    #     print(f"antivirus embedding size: {av_embd.size()}")
    #     print(f"positional embedding size: {pos_embd.size()}")

        
    #     # Embed token
    #     token_embd = bpe_embd + av_embd + pos_embd
    #     token_embd = self.layer_norm(token_embd)

    #     # token_embd = torch.cat((bpe_embd, av_embd, pos_embd), dim=-1)
        
    #     return token_embd


class PretrainEncoder(nn.Module):

    def __init__(self, A, L, D, H, tok_layers, PAD_idx, NO_AV_idx, token_embd):
        """Implements AVScan2Vec's forward pass during pre-training

        Arguments:
        A -- The number of AV products
        L -- Number of tokens per label
        D -- Embedding dimension
        H -- Hidden layer dimension
        tok_layers -- Number of layers in the token encoder
        PAD_idx -- Index of <PAD> in token vocabulary
        token_embd -- PositionalEmbedding object
        """

        super(PretrainEncoder, self).__init__()
        self.A = A
        self.L = L
        self.D = D
        self.H = H
        self.tok_layers = tok_layers


        # make sure PAD_idx is the index of the token in the vocabulary
        self.PAD_idx = PAD_idx

        # make sure NO_AV is the index of the AV in the vocabulary
        self.NO_AV_idx = NO_AV_idx

        print("pad idx in pretrain encoder", self.PAD_idx)

        # PositionalEmbedding object
        self.token_embd = token_embd

        # Define token transformer encoder
        encoderlayer_tok_args = {
            "d_model": D,
            "nhead": 8,
            "dim_feedforward": H,
            "batch_first": True
        }
        encoderlayer_tok = nn.TransformerEncoderLayer(**encoderlayer_tok_args)
        self.encoder_tok = nn.TransformerEncoder(encoderlayer_tok, num_layers=tok_layers)

        self.aggregator_tok = nn.Sequential(
            nn.Linear(D * L, D),
            nn.LeakyReLU(),
            nn.LayerNorm(D)
        )


        # Define av transformer encoder
        encoderlayer_av_args = {
            "d_model": D,
            "nhead": 8,
            "dim_feedforward": H,
            "batch_first": True
        }
        encoderlayer_av = nn.TransformerEncoderLayer(**encoderlayer_av_args)
        self.encoder_av = nn.TransformerEncoder(encoderlayer_av, num_layers=tok_layers)

        self.aggregator_av = nn.Sequential(
            nn.Linear(D * A, D),
            nn.LeakyReLU(),
            nn.LayerNorm(D)
        )


    def forward(self, X_scan, X_av):
        """Forward pass through AVScan2Vec pretrain model.

        Arguments:
        X_scan -- Batch of scan reports (B, A*L)
        X_scan -- Batch of scan reports (B, A*L)
        X_av -- AVs with labels in batch (B, A)
        """
        
        B = X_scan.size(0)
        X_scan = X_scan.reshape(B, self.A, self.L) # (B, A, L)
        X_scan = X_scan.reshape(B * self.A, self.L) # (B*A, L)

        # Get mask indicating <PAD> tokens in X_scan
        with torch.no_grad():
            token_mask = (X_scan == self.PAD_idx) # (B * A, L)


        # Apply positional and segment embeddings
        # pass X_scan into the PositionalEmbedding model and get the embeddings
        X_scan_embd = self.token_embd(X_scan) # (B * A, L, D)

        # Encode X_scan using token encoder
        X_tok_enc = self.encoder_tok(X_scan_embd, src_key_padding_mask=token_mask) # (B * A, L, D)
        # X_tok_enc = X_tok_enc.reshape(B, self.A, self.L, self.D)
        X_tok_enc = X_tok_enc.reshape(B * self.A, self.L * self.D) # (B * A, L * D)

        # Aggregate X_tok_enc to shape (B, A, D)
        X_agg_tok = self.aggregator_tok(X_tok_enc) # (B * A, D)
        # print(f"X_agg size: {X_agg_tok.size()}")

        X_agg_tok = X_agg_tok.reshape(B, self.A, self.D) # (B, A, D)

        with torch.no_grad():
            av_mask = (X_av == self.NO_AV_idx)
        
        X_av_enc = self.encoder_av(X_agg_tok, src_key_padding_mask=av_mask) # (B, A, D)

        X_av_enc = X_av_enc.reshape(B, self.A * self.D) # (B, A * D)

        # Aggregate X_tok_av to shape (B, D)
        X_agg_av = self.aggregator_av(X_av_enc) # (B, D)

        # print(f"X_agg_av size: {X_agg_av.size()}")

        return X_agg_av


class PretrainLoss(nn.Module):
    def __init__(self, A, L, D, H, tok_layers, encoder, dataset):
        """
        Compute loss for Masked Token Prediction and Masked Label Prediction.

        Arguments:
        A -- The number of AV products
        L -- Number of tokens per label
        D -- Embedding dimension
        H -- Hidden layer dimension
        tok_layers -- Number of layers in the token encoder
        encoder -- AVScan2Vec PretrainEncoder object
        dataset -- AVScanDataset object
        """

        super(PretrainLoss, self).__init__()
        self.A = A
        self.L = L
        self.D = D
        self.H = H
        self.tok_layers = tok_layers
        self.encoder = encoder
        self.dataset = dataset

        print(f"Vocab size in pretrain loss: {dataset.vocab_size}")
        self.vocab_size = dataset.vocab_size

        # Define network for aggregating outputs of transformer encoder for LSTM hidden state
        self.h_init = nn.Sequential(
            nn.Linear(D, D*tok_layers//2),
            nn.LeakyReLU(),
            nn.LayerNorm(D*tok_layers//2),
            nn.Linear(D*tok_layers//2, D*tok_layers),
            nn.LeakyReLU(),
            nn.LayerNorm(D*tok_layers),
        )

        # Initial cell state of decoder
        _c_init = torch.zeros(tok_layers, 1, D)
        self.register_buffer("c_init", _c_init) # (tok_layers, 1, D)

        # <SOS> token to use as initial decoder input
        print(torch.cat([self.dataset.tok_to_tensor(tok) for tok in dataset.SOS_toks]).size())
        sys.stdout.flush()
        
        _X_SOS = torch.cat([self.dataset.tok_to_tensor(tok) for tok in dataset.SOS_toks]).reshape(A, -1) # (A, max_chars)        
        self.register_buffer("X_SOS", _X_SOS) # (A, max_chars)
        
        # Define the transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=D,
            nhead=8,
            # dim_feedforward=H,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=tok_layers,
        )

        print(f"Vocab size in pretrain loss: {dataset.vocab_size}")
        self.linear = nn.Linear(D, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)


        # Define token prediction network
        self.predict_token = nn.Sequential(
            nn.Linear(D, H),
            nn.LeakyReLU(),
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.LeakyReLU(),
            nn.LayerNorm(H),
        )

        # Define label prediction network
        self.predict_label = nn.Sequential(
            nn.Linear(D, H),
            nn.LeakyReLU(),
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.LeakyReLU(),
            nn.LayerNorm(H),
        )

        # Adaptive softmax with loss
        # self.alswl = AdaptiveLogSoftmaxWithLoss(H, self.vocab_size, cutoffs=[750, 5000, 20000])
        # self.alswl = AdaptiveLogSoftmaxWithLoss(H, self.vocab_size, cutoffs=[750, 2500, 6000])

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').

        Arguments:
        sz -- Size of the mask
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def CrossEntropyLossTime(self, Y_hat, Y, padding_idx=0):

        # padding_idx = 1
        
        cel = nn.CrossEntropyLoss(ignore_index=padding_idx)
        T = min(Y_hat.size(1), Y.size(1))
        loss = 0.0
        for t in range(T):

            if (torch.all(Y[:,t] == padding_idx)):
                continue

            loss += cel(Y_hat[:,t,:], Y[:,t])

            # torch.set_printoptions(profile="full")

            # print("y_hat: ", Y_hat[:,t,:])
            # print(loss)
            # if (torch.any(torch.isnan(loss))):
            #     exit(0)

        return loss

    def forward(self, X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av):
        """Compute masked token prediction and masked label prection loss.
        If not being trained, also returns predicted tokens and labels

        Arguments:
        X_scan -- Batch of scan reports (B, A*L+1)
        X_av -- AVs with labels in batch (B, A)
        Y_scan -- Randomly selected tokens to predict (?)
        Y_idxs -- Indices of tokens in to predict (B, A)
        Y_label -- Batch of randomly selected labels to predict (B, L)
        Y_av -- Batch of AVs which produced the labels to predict (B)
        """

        # Get batch size
        B = X_scan.shape[0]

        # print(f"X_scan size at pretrain loss: {X_scan.size()}")

        # Encode X_scan
        # X_tok_enc = self.encoder(X_scan, X_av)
        X_vec = self.encoder(X_scan, X_av) # (B, D)

        y_label_embd = self.encoder.token_embd.token_embd(Y_label) # (B, L, D)

        input_mask = self.generate_square_subsequent_mask(Y_label.size(1)).to("cuda:0")

        # reshape X_vec to (B, 1, D)
        X_vec = X_vec.reshape(B, 1, self.D) # (B, 1, D)

        # repeat X_vec L times to get decoder input
        X_vec = X_vec.repeat(1, Y_label.size(1), 1) # (B, L, D)

        # print("X_vec size in pretrainloss: ", X_vec.size())

        X_label_decoded = self.decoder(tgt=y_label_embd, memory=X_vec, tgt_mask=input_mask, memory_mask=input_mask) # (B, L, D)

        X_label_decoded = self.linear(X_label_decoded) # (B, L, vocab_size=8192)
        # X_label_decoded = self.softmax(X_label_decoded) # (B, L, vocab_size=8192)

        # print("X_label_decoded size in pretrain loss: ", X_label_decoded.size())

        # run loss function given X_label_decoded and Y_scan here
        label_loss = self.CrossEntropyLossTime(X_label_decoded, Y_label)

        print("label loss: ", label_loss.item())

        token_loss = torch.zeros(1).to("cuda:0")



        # # Get encoding of CLS token
        # X_vec = X_tok_enc[:, 0, :] # (B, D)
        # print(f"X_vec size in pretrainloss: {X_vec.size()}")

        # # Reshape X_tok_enc to (B*A, L, D) (Drop CLS token)
        # # X_tok_sel = X_tok_enc[:, 1:, :]
        # X_tok_sel = X_tok_enc

        # print(f"X_tok_sel size in pretrainloss: {X_tok_sel.size()}")

        # # X_tok_sel = X_tok_enc[:, 1:, :]
        # X_tok_sel = X_tok_enc

        # print(f"X_tok_sel size in pretrainloss: {X_tok_sel.size()}")

        # X_tok_sel = X_tok_sel.reshape(B*self.A, self.L, self.D) # (B*A, L, D)

        # # Select elements from X_scan_enc for prediction
        # Y_idxs = Y_idxs.reshape(B*self.A) # (B*A)
        # X_tok_pred = X_tok_sel[Y_idxs > 0] # (?, L, D)
        # Y_idxs = Y_idxs[Y_idxs > 0] # (?)
        # X_tok_pred = X_tok_pred[torch.arange(Y_idxs.shape[0]), Y_idxs] # (?, D)

        # # Get masked token prediction loss
        # pred_token_logits = self.predict_token(X_tok_pred) # (?, D)
        # _, token_loss = self.alswl(pred_token_logits, Y_scan)

        # Repeat X_scan_vec n_layers times to get decoder hidden state
        # h_decoder = self.h_init(X_vec) # (B, D*tok_layers)
        # print(f"h_decoder size: {h_decoder.size()}") # (B*A, D*tok_layers)
        # print(f"tok_layers: {self.tok_layers}")
        # # exit(0)
        
        # print(f"h_decoder size: {h_decoder.size()}") # (B*A, D*tok_layers)
        # print(f"tok_layers: {self.tok_layers}")
        # # exit(0)
        
        # h_decoder = h_decoder.reshape(B, self.D, self.tok_layers)
        # h_decoder = h_decoder.permute(2, 0, 1).contiguous() # (tok_layers, B, D)

        # # Repeat c_init B times to get decoder cell state
        # with torch.no_grad():
        #     c_decoder = self.c_init.repeat(1, B, 1) # (tok_layers, B, D)

        # Embed the <SOS> token as the first input to the decoder
        # with torch.no_grad():
        #     Y_pos = Y_av * self.L
        #     X_SOS_tok = self.X_SOS[Y_av] # (B, max_chars)
        #     X_decoder = self.encoder.token_embd.token_embd(X_SOS_tok) # (B, D)
        #     X_decoder = X_decoder + self.encoder.token_embd.av_embd(Y_av+1)
        #     X_decoder = X_decoder + self.encoder.token_embd.pos_embd(Y_pos+1)
        #     X_decoder = self.encoder.token_embd.layer_norm(X_decoder)

        #     # added sum here (remove)
        #     # X_decoder = X_decoder.sum(2)[:, 0:1, :]

        #     # print(f"X_decoder size: {X_decoder.size()}")

        #     X_decoder = X_decoder.reshape(B, 1, self.D) # (B, 1, D)
            

        # Determine whether to use teacher forcing (50% when training)
        # teacher_forcing = False
        # if self.training:
        #     teacher_forcing = np.random.choice((True, False))

        # # Iterate over each timestep
        # pred_tokens = None
        # pred_labels = []
        # label_loss = 0.0
        # for t in range(self.L):

        #     # Get output for decoder at current timestep
        #     decoder_out, (h_decoder, c_decoder) = self.decoder(X_decoder, (h_decoder, c_decoder))
        #     # print(f"decoder_out size: {decoder_out.size()}")
        #     # i have no clue what im doing THIS IS DEFINITELY WRONG. this is just some random crap i did to force x_decoder to be the right shape
        #     # decoder_out = decoder_out[:, 0:1, :].reshape(B, self.D)
        #     decoder_out = decoder_out.reshape(B, self.D)

        #     # Predict token to use as decoder's next input
        #     pred_label_logits = self.predict_label(decoder_out)
        #     with torch.no_grad():
        #         if teacher_forcing:
        #             X_next = Y_label[:, t]
        #         else:
        #             X_next = self.alswl.predict(pred_label_logits)
        #         if not self.training:
        #             pred_labels.append(X_next)

        #     # Embed predicted token
        #     if t+1 < self.L:
        #         X_next_tok = torch.stack([self.dataset.tok_to_tensor(self.dataset.token_vocab_rev[tok.item()]) for tok in X_next])
        #         X_next_tok = X_next_tok.to(next(self.parameters()).device)
        #         X_decoder = self.encoder.token_embd.token_embd(X_next_tok).reshape(B, -1, self.D) # (B, 1, D)

        #     # Get timestep loss
        #     _, timestep_loss = self.alswl(pred_label_logits, Y_label[:, t])
        #     label_loss = label_loss + timestep_loss

        # # Predict tokens/labels if evaluating model
        # if not self.training:
        #     with torch.no_grad():
        #         pred_tokens = self.alswl.predict(pred_token_logits)
        #         pred_labels = torch.stack(pred_labels, dim=1)
        #     return token_loss, label_loss, pred_tokens, pred_labels

        return token_loss, label_loss, None, None


class FinetuneEncoder(nn.Module):

    def __init__(self, pretrain_encoder):
        """Class for fine-tuning AVScan2Vec

        Arguments:
        pretrain_encoder -- Trained AVScan2Vec PretrainEncoder object
        """

        super(FinetuneEncoder, self).__init__()
        self.pretrain_encoder = pretrain_encoder

        # Freeze token embedding, all but last two layers of token encoder
        for param in pretrain_encoder.token_embd.parameters():
            param.requires_grad = False
        for layer in pretrain_encoder.encoder_tok.layers[:2]:
            for param in layer.parameters():
                param.requires_grad = False
            layer.dropout = nn.Dropout(p=0.0, inplace=True)


    def forward(self, X_scan, X_av):
        """Perform forward pass of AVScan2Vvec fine-tune model.

        Arguments:
        X_scan -- Batch of scan reports (B, A*L+1, max_chars)
        X_av -- AVs with labels in batch (B, A)
        """

        # Encode scan report using pretrained encoder
        X_tok_enc = self.pretrain_encoder(X_scan, X_av) # (B, A*L+1, D)

        # Return batch of vectors correspoding to <CLS> token
        X_vec = X_tok_enc[:, 0, :]
        return X_vec


class FinetuneLoss(nn.Module):

    def __init__(self, finetune_encoder):
        """Computes Multiple Negatives Ranking Loss for AVScan2Vec finetune model.

        Arguments:
        finetune_encoder -- AVScan2Vec FinetuneEncoder object
        """

        super(FinetuneLoss, self).__init__()
        self.finetune_encoder = finetune_encoder


    def forward(self, X_scan_anc, X_av_anc, X_scan_pos, X_av_pos):
        """Loss based on https://github.com/UKPLab/sentence-transformers/issues/373

        Arguments
        X_scan_anc -- Batch of anchor scan reports (B, A*L+1)
        X_scan_pos -- Batch of positive scan reports (B, A*L+1)
        X_av_anc -- AVs with labels in X_scan_anc (B, A)
        X_av_pos -- AVs with labels in X_scan_pos (B, A)
        """

        X_anc = self.finetune_encoder(X_scan_anc, X_av_anc)
        X_pos = self.finetune_encoder(X_scan_pos, X_av_pos)

        # Compute MNR loss
        scores = torch.matmul(X_pos, X_anc.t())
        diagonal_mean = torch.mean(torch.diag(scores))
        mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
        return -diagonal_mean + mean_log_row_sum_exp
