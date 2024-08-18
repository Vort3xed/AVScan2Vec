import os
import sys
import copy
import json
import mmap
import torch
import pickle
import random
import numpy as np
from bpe import Encoder
from datetime import datetime as dt
from torch.utils.data import Dataset

from globalvars import *
from utils import tokenize_label, read_supported_avs


class AVScanDataset(Dataset):

    def __init__(self, data_dir, max_tokens=20, max_chars=10, max_vocab=10000000):
        """Base dataset class for AVScan2Vec.

        Arguments:
        data_dir -- Path to dataset directory
        max_tokens -- Maximum number of tokens per label
        max_chars -- Maximum number of chars per token
        max_vocab -- Maximum number of tokens to track (for masked token / label prediction)
        """

        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.max_vocab = max_vocab

        # Read supported AVs
        av_path = os.path.join(data_dir, "avs.txt")
        self.supported_avs = read_supported_avs(av_path)
        self.avs = sorted(list(self.supported_avs))
        self.av_vocab_rev = [NO_AV] + self.avs
        self.num_avs = len(self.avs)

        # Map each AV to a unique index
        self.av_vocab = {av: idx for idx, av in enumerate(self.av_vocab_rev)}

        # Construct character alphabet
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        self.alphabet = [char for char in self.alphabet]
        self.SOS_toks = ["<SOS_{}>".format(av) for av in self.avs]
        self.special_tokens = [PAD, CLS, EOS, ABS, BEN] + self.SOS_toks + [SOW, EOW, MASK, UNK]
        self.special_tokens_set = set(self.special_tokens)
        # print(self.special_tokens_set)
        print("<BENIGN>" in self.special_tokens_set)
        # print(self.special_tokens_set)
        print("<BENIGN>" in self.special_tokens_set)
        self.alphabet = self.special_tokens + self.alphabet
        self.alphabet_rev = {char: i for i, char in enumerate(self.alphabet)}

        # Load token vocabulary
        vocab_path = os.path.join(data_dir, "vocab.txt")
        self.token_vocab_rev = []
        with open(vocab_path, "r") as f:
            for line in f:
                if len(self.token_vocab_rev) >= self.max_vocab:
                    break
                tok = line.strip()
                self.token_vocab_rev.append(tok)

        # Map each token to a unique index
        self.token_vocab = {tok: idx for idx, tok in enumerate(self.token_vocab_rev)}

        # Zipf distribution for sampling tokens
        self.vocab_size = len(self.token_vocab_rev)
        self.zipf_vals = np.arange(5, self.vocab_size)
        self.zipf_p = 1.0 / np.power(self.zipf_vals, 2.0)
        self.zipf_p /= np.sum(self.zipf_p)

        # Load line offsets
        line_path = os.path.join(data_dir, "line_offsets.pkl")
        with open(line_path, "rb") as f:
            self.line_offsets = pickle.load(f)
        self.line_paths = sorted(list(self.line_offsets.keys()))

        # Get total number of scan reports
        self.num_reports = sum([len(v) for v in self.line_offsets.values()])

        self.test_corpus = ""
        with open('/home/agneya/AVScan2Vec/avscan2vec/test_corpus.txt', 'r') as file:
            self.test_corpus = file.read()

        # self.encoder = Encoder(200, pct_bpe=0.88)
        # print("self.vocab_size in dataset", self.vocab_size)
        # self.encoder = Encoder(self.vocab_size, pct_bpe=1.2)
        # print("self.vocab_size:", self.vocab_size)
        # self.encoder.fit(self.test_corpus.split('\n'))





        # prints the ldjson path: /media/data1/labels/SOREL_110000/SOREL_labels.ldjson
        # print(self.line_paths)

        # test_corpus = []

        # with open(self.line_paths[0], 'r') as file:
        #     for line in file.readlines():
        #         report = json.loads(line)
        #         for av in report["scans"].keys():
        #             scan_info = report["scans"][av]
        #             if scan_info.get("result") is None:
        #                 continue
        #             else:
        #                 label = scan_info["result"]
        #                 tokens = tokenize_label(label)
        #                 test_corpus += tokens

        # print("test_corpus:", len(test_corpus))

        # string_test_corpus = " ".join(test_corpus)

        # self.encoder.fit(string_test_corpus)

        # print("bpe vocab size:", len(self.encoder.bpe_vocab))

        with open('../testing/encoder.pkl', 'rb') as file:
            self.encoder = pickle.load(file)

        self.vocab_size = self.encoder.vocab_size
        print("vocab size in dataset:", self.vocab_size)
            


    def sanitize(bpe_bits):
        return [element for element in bpe_bits if element not in ["__sow", "__eow"]]
    

    def parse_scan_report(self, idx):

        # Find the file path that contains the target scan report
        line_path = None
        for file_path in self.line_paths:
            
            if idx - len(self.line_offsets[file_path]) < 0:
                line_path = file_path
                break
            idx -= len(self.line_offsets[file_path])

        # Seek to first byte of that scan report in file path
        start_byte = self.line_offsets[line_path][idx]

        # Read report from file
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as f_mmap:
                f_mmap.seek(start_byte)
                line = f_mmap.readline()
                report = json.loads(line)
        md5 = report["md5"]
        sha1 = report["sha1"]
        sha256 = report["sha256"]
        scan_date = report["scan_date"]
        # scan_date = dt.fromtimestamp(scan_date).strftime("%Y-%m-%d")

        scan_date = dt.strptime(scan_date, "%Y-%m-%d %H:%M:%S")
        scan_date = scan_date.strftime("%Y-%m-%d")

        # Parse AVs and tokens from scan report
        av_tokens = {}
        
        for av in report["scans"].keys():

            # Normalize name of AV
            scan_info = report["scans"][av]
            av = AV_NORM.sub("", av).lower().strip()

            # Skip AVs that aren't supported
            if av not in self.supported_avs:
                continue

            # Use <BEN> special token for AVs that detected file as benign
            if scan_info.get("result") is None:
                tokens = [BEN]
            else:
                label = scan_info["result"]
                tokens = tokenize_label(label)[:self.max_tokens-2]
            av_tokens[av] = tokens

        return av_tokens, md5, sha1, sha256, scan_date


    # def tok_to_tensor(self, tok):
    #     """Return a tensor representing each char in a token"""
    #     if tok in self.special_tokens_set:
    #         tok = [SOW, tok, EOW]
    #     else:
    #         tok = tok[:self.max_chars-2]
    #         tok = [SOW] + [char for char in tok] + [EOW]
    #         print(tok)
    #     tok += [PAD]*(self.max_chars-len(tok))
    #     print([self.alphabet_rev[char] for char in tok])
        # X_tok = torch.LongTensor([self.alphabet_rev[char] for char in tok])
        # return X_tok
    
    def extract(self, input_list):
        if "<EOS>" in input_list:
            eos_index = input_list.index("<EOS>")
            return input_list[:eos_index]
        else:
            return input_list  # Return the whole list if "<EOS>" is not found
        

    #X_scan is a 1d array :(
    
    def extract(self, input_list):
        if "<EOS>" in input_list:
            eos_index = input_list.index("<EOS>")
            return input_list[:eos_index]
        else:
            return input_list  # Return the whole list if "<EOS>" is not found
        

    #X_scan is a 1d array :(
    def tok_to_tensor(self, tok):
        """Return a tensor representing each char in a token"""

        if tok in self.special_tokens_set:
            tok = [tok]

            num_rep_label = [self.alphabet_rev[char] for char in tok]
            # num_rep_label += [0]*(self.max_chars-len(num_rep_label))
            # num_rep_label += [0]*(self.max_chars-len(num_rep_label))
            return torch.LongTensor(num_rep_label)
        # else:
        #     #why am i substracting 2 here?
        #     tok = tok[:self.max_chars-2]
        #     tok = [] + self.encoder.tokenize(tok)[1:-1]
        # print(tok)
        # else:
        #     #why am i substracting 2 here?
        #     tok = tok[:self.max_chars-2]
        #     tok = [] + self.encoder.tokenize(tok)[1:-1]
        # print(tok)
        # tok += [PAD]*(self.max_chars-len(tok))
        # # X_tok = torch.LongTensor([self.alphabet_rev[char] for char in tok])
        # # X_tok = torch.LongTensor([self.alphabet_rev[char] for char in tok])
        # print(tok)
        # X_tok = torch.LongTensor(next(self.encoder.transform(tok)))
        # X_tok = torch.LongTensor(next(self.encoder.transform(tok)))
        #should convert ['ge', 'ne', 'ri', 'c'] into like [23, 53, 65, 21] (which is a longTensor)
        #transform the token and remove the __SOW and __EOW
        X_tok = next(self.encoder.transform([tok]))[1:-1]
    
        # X_tok = X_tok[:self.max_chars]
        #padding
        # X_tok += [0]*(self.max_chars-len(X_tok))
        X_tok = torch.LongTensor(X_tok)
        #transform the token and remove the __SOW and __EOW
        X_tok = next(self.encoder.transform([tok]))[1:-1]
    
        # X_tok = X_tok[:self.max_chars]
        #padding
        # X_tok += [0]*(self.max_chars-len(X_tok))
        X_tok = torch.LongTensor(X_tok)

        # print(next(encoder.transform(tok)))
        # exit(0)
        # print("X_TOKEN:", tok)
        # sys.stdout.flush()
        # print("X_TOKEN:", tok)
        # sys.stdout.flush()
        # print(X_tok.shape)
        # sys.stdout.flush()
        # sys.stdout.flush()
        # print(X_tok)
        # sys.stdout.flush()
        # sys.stdout.flush()
        return X_tok
    
    # new implementation of bpe based on emailed word document
    def __getitem__(self, idx):

        # Parse scan report
        av_tokens, md5, sha1, sha256, scan_date = self.parse_scan_report(idx)

        # AV_tokens looks like this: [[malware, win32, xyz], [trojan, linux, abc], [benign, win32, def]]
        # print(av_tokens)
        # print(av_tokens)
        # Construct X_scan from scan report
        X_scan = []
        # print(av_tokens)

        #runs 89 times (for every AV?)
        # print(av_tokens)

        #runs 89 times (for every AV?)
        for av in self.avs:
            if av_tokens.get(av) is None:
                Xi = ["<SOS_{}>".format(av), ABS, EOS]
            else:

                # token_sentence = ""
                # # Build sentence from tokens
                # for token in av_tokens[av]:
                #     token_sentence += token + " "
                # token_sentence = ""
                # # Build sentence from tokens
                # for token in av_tokens[av]:
                #     token_sentence += token + " "

                # # break sentence up into BPE bits and remove __EOW AND __SOW from tokenization 
                # tokenized_sentence = self.sanitize(self.encoder.tokenize(token_sentence))
                # # break sentence up into BPE bits and remove __EOW AND __SOW from tokenization 
                # tokenized_sentence = self.sanitize(self.encoder.tokenize(token_sentence))

                # Xi = ["<SOS_{}>".format(av)] + tokenized_sentence + [EOS]
                # [<SOS_av>, mal, ware, w, in, 32, <EOS>]

                Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
                # [<SOS_av>, malware, win32, <EOS>]
                # Xi = ["<SOS_{}>".format(av)] + tokenized_sentence + [EOS]
                # [<SOS_av>, mal, ware, w, in, 32, <EOS>]

                Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
                # [<SOS_av>, malware, win32, <EOS>]

                print(Xi)
                sys.stdout.flush()

            character_length_xi = 0
            for tok in Xi:
                character_length_xi += len(tok)

            Xi += [PAD]*(self.max_tokens-character_length_xi)
            print(Xi)
            character_length_xi = 0
            for tok in Xi:
                character_length_xi += len(tok)

            Xi += [PAD]*(self.max_tokens-character_length_xi)
            print(Xi)
            X_scan += Xi

            # X_SCAN:
            # [<SOS_av>, malware, win32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>,
            # <SOS_av>, malware, win32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>,
            # <SOS_av>, malware, win32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>]

            # X_SCAN:
            # [<SOS_av>, malware, win32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>,
            # <SOS_av>, malware, win32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>,
            # <SOS_av>, malware, win32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>]

        # Convert X_scan to tensor of characters
        X_scan_bpe = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        X_scan_bpe = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        for tok in X_scan:
            # X_scan_char = np.concatenate((X_scan_char, torch.LongTensor(next(self.encoder.transform(tok))).reshape(1, -1)))
            # print(X_scan)
            X_scan_bpe = np.concatenate((X_scan_bpe, self.tok_to_tensor(tok).reshape(1, -1)))
            #should have the transformed label from every antivirus scanner?
            # print(X_scan)
            X_scan_bpe = np.concatenate((X_scan_bpe, self.tok_to_tensor(tok).reshape(1, -1)))
            #should have the transformed label from every antivirus scanner?

        # print(X_scan_char.shape)
        # print(X_scan_char)
        # sys.stdout.flush()

        # Construct X_av from list of AVs in report
        X_scan = torch.as_tensor(X_scan_bpe)
        
        # X_SCAN:
        # [<SOS_av>, mal, ware, w, in, 32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>,
        # <SOS_av>, mal, ware, w, in, 32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>,
        # <SOS_av>, mal, ware, w, in, 32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>]


        X_scan = torch.as_tensor(X_scan_bpe)
        
        # X_SCAN:
        # [<SOS_av>, mal, ware, w, in, 32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>,
        # <SOS_av>, mal, ware, w, in, 32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>,
        # <SOS_av>, mal, ware, w, in, 32, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>]


        X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]
        X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
        return X_scan, X_av, md5, sha1, sha256, scan_date
    
    # def __getitem__(self, idx):

    #     # Parse scan report
    #     av_tokens, md5, sha1, sha256, scan_date = self.parse_scan_report(idx)

    #     # AV_tokens looks like this: [[malware, win32, xyz], [trojan, linux, abc], [benign, win32, def]]

    #     # Construct X_scan from scan report
    #     X_scan = []
    #     for av in self.avs:
    #         if av_tokens.get(av) is None:
    #             Xi = ["<SOS_{}>".format(av), ABS, EOS]
    #         else:

    #             bpe_tokenized_sentence = []
    #             # BPE encoded tokens:
    #             for token in av_tokens[av]:
    #                 bpe_tokenized_sentence += self.encoder.tokenize(token)[1:-1]

    #             Xi = ["<SOS_{}>".format(av)] + bpe_tokenized_sentence + [EOS]
    #         Xi += [PAD]*(self.max_tokens-len(Xi))
    #         X_scan += Xi

    #     # Convert X_scan to tensor of characters
    #     X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
    #     for tok in X_scan:
    #         X_scan_char = np.concatenate((X_scan_char, torch.LongTensor(next(self.encoder.transform(tok))).reshape(1, -1)))

    #     # print(X_scan_char.shape)
    #     # print(X_scan_char)
    #     # sys.stdout.flush()

    #     # Construct X_av from list of AVs in report
    #     X_scan = torch.as_tensor(X_scan_char)
    #     X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]
    #     X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
    #     return X_scan, X_av, md5, sha1, sha256, scan_date


    #OLD CHARACTERBERT IMPLEMENTATION
    # def __getitem__(self, idx):

    #     # Parse scan report
    #     av_tokens, md5, sha1, sha256, scan_date = self.parse_scan_report(idx)

    #     # Construct X_scan from scan report
    #     X_scan = []
    #     for av in self.avs:
    #         if av_tokens.get(av) is None:
    #             Xi = ["<SOS_{}>".format(av), ABS, EOS]
    #         else:
    #             Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
    #         Xi += [PAD]*(self.max_tokens-len(Xi))
    #         X_scan += Xi

    #     # Convert X_scan to tensor of characters
    #     X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
    #     for tok in X_scan:
    #         X_scan_char = np.concatenate((X_scan_char, self.tok_to_tensor(tok).reshape(1, -1)))

    #     print(X_scan_char.shape)
    #     print(X_scan_char)
    #     sys.stdout.flush()

    #     # Construct X_av from list of AVs in report
    #     X_scan = torch.as_tensor(X_scan_char)
    #     X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]
    #     X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
    #     return X_scan, X_av, md5, sha1, sha256, scan_date

    def __len__(self):
        return self.num_reports


class PretrainDataset(AVScanDataset):

    def __init__(self, data_dir, max_tokens):
        """AVScan2Vec dataset class for pre-training."""
        super().__init__(data_dir, max_tokens=max_tokens)

    # def squish(self, input_list):
    #     new_list = torch.tensor([])
    #     print("tensor sizes-----")
    #     for tensor in input_list:
    #         print(tensor.size())
    #     print("tensor sizes-----")
    #     for tensor in input_list:
    #         new_list = torch.cat((new_list, tensor), 0)

    #     return new_list

    def squish(self, input_list):
        if len(input_list) == 0:
            return []
        new_list = input_list[0].unsqueeze(0) if input_list[0].dim() == 0 else input_list[0]
        for tensor in input_list[1:]:
            # make sure tensor is atleast 1d
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            new_list = torch.cat((new_list, tensor), dim=0)

        return new_list
    
    def true_length(self, input_list):
        length = 0
        for tensor in input_list:
            length += len(tensor)
        return length
    
    def pad_to_number(self, input_list, number):
        while len(input_list) < number:
            input_list = torch.cat((input_list, torch.tensor([self.tok_to_tensor(PAD)])), 0)
        return input_list
    # def squish(self, input_list):
    #     new_list = torch.tensor([])
    #     print("tensor sizes-----")
    #     for tensor in input_list:
    #         print(tensor.size())
    #     print("tensor sizes-----")
    #     for tensor in input_list:
    #         new_list = torch.cat((new_list, tensor), 0)

    #     return new_list

    def squish(self, input_list):
        if len(input_list) == 0:
            return []
        new_list = input_list[0].unsqueeze(0) if input_list[0].dim() == 0 else input_list[0]
        for tensor in input_list[1:]:
            # make sure tensor is atleast 1d
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            new_list = torch.cat((new_list, tensor), dim=0)

        return new_list
    
    def true_length(self, input_list):
        length = 0
        for tensor in input_list:
            length += len(tensor)
        return length
    
    def pad_to_number(self, input_list, number):
        while len(input_list) < number:
            input_list = torch.cat((input_list, torch.tensor([self.tok_to_tensor(PAD)])), 0)
        return input_list

    def __getitem__(self, idx):

        # Parse scan report
        av_tokens, md5, _, _, scan_date = self.parse_scan_report(idx)

        # Randomly select one AV to hold out (train only)
        # Construct Y_label from held-out AV's label
        Y_label = []
        Y_av = random.choice(list(av_tokens.keys()))
        Y_label = ["<SOS_{}>".format(Y_av)] + av_tokens[Y_av] + [EOS]
        Y_label = [self.tok_to_tensor(tok) for tok in Y_label]
        Y_label += [self.tok_to_tensor(PAD)]*(self.max_tokens-self.true_length(Y_label))
        # Y_label = [self.tok_to_tensor(tok) for tok in Y_label]
        # Y_label += [self.tok_to_tensor(PAD)]*(self.max_tokens-self.true_length(Y_label))
        av_tokens[Y_av] = None

        # Randomly select 5% of tokens to be replaced with MASK
        # MOVE THIS DOWN LATER
        # MOVE THIS DOWN LATER
        Y_idxs = [0] * self.num_avs
        rand_nums = np.random.randint(0, 100, size=self.num_avs)

        #list of numeric representations

        #list of numeric representations
        pred_tokens = set()

        # for i, (av, tokens) in enumerate(av_tokens.items()):
        #     if tokens is None:
        #         continue
        #     if rand_nums[i] < 5:
        #         token_idxs = [i+1 for i, tok in enumerate(tokens) if not
        #                       tok.startswith("<") and not tok.endswith(">")]
        #         if not len(token_idxs):
        #             continue
        #         Y_idx = random.choice(token_idxs)
                # Y_idxs[self.av_vocab[av]-1] = Y_idx
        #         pred_tokens.add(tokens[Y_idx-1])

        # ADDED CLS HERE!
        # X_scan = [torch.tensor([1])]
        X_scan = []

        # for i, (av, tokens) in enumerate(av_tokens.items()):
        #     if tokens is None:
        #         continue
        #     if rand_nums[i] < 5:
        #         token_idxs = [i+1 for i, tok in enumerate(tokens) if not
        #                       tok.startswith("<") and not tok.endswith(">")]
        #         if not len(token_idxs):
        #             continue
        #         Y_idx = random.choice(token_idxs)
                # Y_idxs[self.av_vocab[av]-1] = Y_idx
        #         pred_tokens.add(tokens[Y_idx-1])

        # ADDED CLS HERE!
        # X_scan = [torch.tensor([1])]
        X_scan = []

        # Construct X_scan from scan report
        for i,av in enumerate(self.avs):
            if av_tokens.get(av) is None:
                Xi = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
            # [torch([1, 3]), torch([4, 6, 85]), torch([7])]
            # Xi = [self.tok_to_tensor(tok) for tok in Xi]
            if rand_nums[i] < 5:
                token_idxs = [i+1 for i in range(len(Xi) - 1)]
                # abstain and benign can be added here
                Y_idx = random.choice(token_idxs)
                Y_idxs[self.av_vocab[av]-1] = Y_idx
                pred_tokens.add(Xi[Y_idx])
            #Padding at the label level 

            # print(Xi)
            # print(len(Xi))
            # print(self.max_tokens)
            # print(" ")

            # print(Xi)
            
            # print(character_length_xi)
            # print("amt of padding added:", self.max_tokens-character_length_xi)
            # Xi += [self.tok_to_tensor(PAD)]*(self.max_tokens-self.true_length(Xi))
            # print(Xi)

            # print("xi length:", self.true_length(Xi))
            # print("xi after padding:", Xi)
            # [torch([1, 3]), torch([4, 6, 85]), torch([7])]
            Xi = [self.tok_to_tensor(tok) for tok in Xi]
            if rand_nums[i] < 5:
                token_idxs = [i+1 for i in range(len(Xi) - 1)]
                # abstain and benign can be added here
                Y_idx = random.choice(token_idxs)
                Y_idxs[self.av_vocab[av]-1] = Y_idx
                pred_tokens.add(Xi[Y_idx])
            #Padding at the label level 

            # print(Xi)
            # print(len(Xi))
            # print(self.max_tokens)
            # print(" ")

            # print(Xi)
            
            # print(character_length_xi)
            # print("amt of padding added:", self.max_tokens-character_length_xi)
            Xi += [self.tok_to_tensor(PAD)]*(self.max_tokens-self.true_length(Xi))
            # print(Xi)

            # print("xi length:", self.true_length(Xi))
            # print("xi after padding:", Xi)
            X_scan += Xi
        # print("size of x_scan:", len(X_scan))

        # exit(0)
        # print("size of x_scan:", len(X_scan))

        # exit(0)

        # Construct X_av from list of AVs in report
        X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]

        # X_scan = self.squish(X_scan)

        # Y_scan holds out BPE BITS, not tokens!
        X_scan = self.squish(X_scan)

        # Y_scan holds out BPE BITS, not tokens!
        # Construct Y_scan from 5% of held-out tokens
        Y_scan = []
        for i, av in enumerate(self.avs):
            Y_idx = Y_idxs[i]
            if Y_idx > 0:
                # print(X_scan.size())
                # print(i*self.max_tokens+Y_idx)
                # print(X_scan.size())
                # print(i*self.max_tokens+Y_idx)
                Y_scan.append(X_scan[i*self.max_tokens+Y_idx])

        # MASK any tokens in pred_tokens 80% of the time
        # 10% of the time, replace with a random token
        # 10% of the time, leave the token alone
        rand_nums = np.random.randint(0, 100, size=self.num_avs*self.max_tokens)
        for i, tok in enumerate(X_scan):
            if tok in pred_tokens:
                if rand_nums[i] < 80:
                    X_scan[i] = self.tok_to_tensor(MASK)
                    X_scan[i] = self.tok_to_tensor(MASK)
                elif rand_nums[i] < 90:
                    #TO-DO REPLACE WITH RANDOM BPE TOKEN NUMBER
                    # X_scan[i] = self.token_vocab_rev[random.randint(5, self.vocab_size-1)]
                    pass
                    #TO-DO REPLACE WITH RANDOM BPE TOKEN NUMBER
                    # X_scan[i] = self.token_vocab_rev[random.randint(5, self.vocab_size-1)]
                    pass

        # Convert X_scan to tensor of characters
        # X_scan_bpe = self.tok_to_tensor(CLS).reshape(1, -1) # (1, L)

        # X_scan_bpe = self.tok_to_tensor(CLS).reshape(1, 1) # (1, 1)
        # for tok in X_scan:
        #     X_scan_bpe = np.concatenate((X_scan_bpe, self.tok_to_tensor(tok).reshape(1, -1)))

        # ### MAKE EMPTY TENSOR FOR Y_SCAN_BPE
        # Y_scan_bpe = np.empty.reshape(1, -1) # (1, L)
        # for tok in Y_scan:
        #     Y_scan_bpe = np.concatenate((Y_scan_bpe, self.tok_to_tensor(tok).reshape(1, -1)))

        # ### MAKE EMPTY TENSOR FOR Y_SCAN_BPE
        # Y_label_bpe = np.empty.reshape(1, -1) # (1, L)
        # for tok in Y_label:
        #     Y_label_bpe = np.concatenate((Y_label_bpe, self.tok_to_tensor(tok).reshape(1, -1)))

        # FIX X_SCAN, Y_LABEL AND Y_SCAN DIMENSIONS
        # X_scan_bpe = self.tok_to_tensor(CLS).reshape(1, -1) # (1, L)

        # X_scan_bpe = self.tok_to_tensor(CLS).reshape(1, 1) # (1, 1)
        # for tok in X_scan:
        #     X_scan_bpe = np.concatenate((X_scan_bpe, self.tok_to_tensor(tok).reshape(1, -1)))

        # ### MAKE EMPTY TENSOR FOR Y_SCAN_BPE
        # Y_scan_bpe = np.empty.reshape(1, -1) # (1, L)
        # for tok in Y_scan:
        #     Y_scan_bpe = np.concatenate((Y_scan_bpe, self.tok_to_tensor(tok).reshape(1, -1)))

        # ### MAKE EMPTY TENSOR FOR Y_SCAN_BPE
        # Y_label_bpe = np.empty.reshape(1, -1) # (1, L)
        # for tok in Y_label:
        #     Y_label_bpe = np.concatenate((Y_label_bpe, self.tok_to_tensor(tok).reshape(1, -1)))

        # FIX X_SCAN, Y_LABEL AND Y_SCAN DIMENSIONS

        # Convert to LongTensor

        #Xscan shape: (A*max_tokens) or (A*max_tokens, 1)
        
        # X_scan = torch.LongTensor(X_scan)

        
        # print(X_scan.size())
        
        # X_scan = self.pad_to_number(X_scan, self.num_avs*self.max_tokens)

        # X_scan = torch.LongTensor(X_scan_elements)

        #Xscan shape: (A*max_tokens) or (A*max_tokens, 1)
        
        # X_scan = torch.LongTensor(X_scan)

        
        # print(X_scan.size())
        
        # X_scan = self.pad_to_number(X_scan, self.num_avs*self.max_tokens)

        # X_scan = torch.LongTensor(X_scan_elements)
        X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
        # Y_scan = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_scan])
        Y_scan = self.squish(Y_scan)
        # Y_scan = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_scan])
        Y_scan = self.squish(Y_scan)
        Y_idxs = torch.LongTensor(Y_idxs)
        # Y_label = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_label])
        # Y_label = torch.LongTensor(Y_label)

        Y_label = self.squish(Y_label)

        # Y_label = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_label])
        # Y_label = torch.LongTensor(Y_label)

        Y_label = self.squish(Y_label)

        Y_av = torch.LongTensor([self.av_vocab[Y_av]-1])


        # return X_SCAN BPE ETC

        # return X_SCAN BPE ETC
        return X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av, md5, scan_date

# class PretrainDataset(AVScanDataset):
#     def __init__(self, data_dir, max_tokens):
#         """AVScan2Vec dataset class for pre-training."""
#         super().__init__(data_dir, max_tokens=max_tokens)

#     def __getitem__(self, idx):

#         # Parse scan report
#         av_tokens, md5, _, _, scan_date = self.parse_scan_report(idx)

#         # Randomly select one AV to hold out (train only)
#         # Construct Y_label from held-out AV's label
#         Y_label = []
#         Y_av = random.choice(list(av_tokens.keys()))
#         Y_label = ["<SOS_{}>".format(Y_av)] + av_tokens[Y_av] + [EOS]
#         Y_label += [PAD]*(self.max_tokens-len(Y_label))
#         av_tokens[Y_av] = None

#         # Randomly select 5% of tokens to be replaced with MASK
#         Y_idxs = [0] * self.num_avs
#         rand_nums = np.random.randint(0, 100, size=self.num_avs)
#         pred_tokens = set()
#         for i, (av, tokens) in enumerate(av_tokens.items()):
#             if tokens is None:
#                 continue
#             if rand_nums[i] < 5:
#                 token_idxs = [i+1 for i, tok in enumerate(tokens) if not
#                               tok.startswith("<") and not tok.endswith(">")]
#                 if not len(token_idxs):
#                     continue
#                 Y_idx = random.choice(token_idxs)
#                 Y_idxs[self.av_vocab[av]-1] = Y_idx
#                 pred_tokens.add(tokens[Y_idx-1])

#         # Construct X_scan from scan report
#         X_scan = []
#         for av in self.avs:
#             if av_tokens.get(av) is None:
#                 Xi = ["<SOS_{}>".format(av), ABS, EOS]
#             else:
#                 Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
#             Xi += [PAD]*(self.max_tokens-len(Xi))
#             X_scan += Xi

#         # Construct X_av from list of AVs in report
#         X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]

#         # Construct Y_scan from 5% of held-out tokens
#         Y_scan = []
#         for i, av in enumerate(self.avs):
#             Y_idx = Y_idxs[i]
#             if Y_idx > 0:
#                 Y_scan.append(X_scan[i*self.max_tokens+Y_idx])

#         # MASK any tokens in pred_tokens 80% of the time
#         # 10% of the time, replace with a random token
#         # 10% of the time, leave the token alone
#         rand_nums = np.random.randint(0, 100, size=self.num_avs*self.max_tokens)
#         for i, tok in enumerate(X_scan):
#             if tok in pred_tokens:
#                 if rand_nums[i] < 80:
#                     X_scan[i] = MASK
#                 elif rand_nums[i] < 90:
#                     X_scan[i] = self.token_vocab_rev[random.randint(5, self.vocab_size-1)]

#         # Convert X_scan to tensor of characters
#         X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
#         for tok in X_scan:
#             # print("TOK")
#             # print(X_scan)
#             # exit(0)
#             X_scan_char = np.concatenate((X_scan_char, self.tok_to_tensor(tok).reshape(1, -1)))

#         # Convert to LongTensor
#         X_scan = torch.as_tensor(X_scan_char)
#         X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
#         Y_scan = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_scan])
#         Y_idxs = torch.LongTensor(Y_idxs)
#         Y_label = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_label])
#         Y_av = torch.LongTensor([self.av_vocab[Y_av]-1])

#         return X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av, md5, scan_date

# class PretrainDataset(AVScanDataset):
#     def __init__(self, data_dir, max_tokens):
#         """AVScan2Vec dataset class for pre-training."""
#         super().__init__(data_dir, max_tokens=max_tokens)

#     def __getitem__(self, idx):

#         # Parse scan report
#         av_tokens, md5, _, _, scan_date = self.parse_scan_report(idx)

#         # Randomly select one AV to hold out (train only)
#         # Construct Y_label from held-out AV's label
#         Y_label = []
#         Y_av = random.choice(list(av_tokens.keys()))
#         Y_label = ["<SOS_{}>".format(Y_av)] + av_tokens[Y_av] + [EOS]
#         Y_label += [PAD]*(self.max_tokens-len(Y_label))
#         av_tokens[Y_av] = None

#         # Randomly select 5% of tokens to be replaced with MASK
#         Y_idxs = [0] * self.num_avs
#         rand_nums = np.random.randint(0, 100, size=self.num_avs)
#         pred_tokens = set()
#         for i, (av, tokens) in enumerate(av_tokens.items()):
#             if tokens is None:
#                 continue
#             if rand_nums[i] < 5:
#                 token_idxs = [i+1 for i, tok in enumerate(tokens) if not
#                               tok.startswith("<") and not tok.endswith(">")]
#                 if not len(token_idxs):
#                     continue
#                 Y_idx = random.choice(token_idxs)
#                 Y_idxs[self.av_vocab[av]-1] = Y_idx
#                 pred_tokens.add(tokens[Y_idx-1])

#         # Construct X_scan from scan report
#         X_scan = []
#         for av in self.avs:
#             if av_tokens.get(av) is None:
#                 Xi = ["<SOS_{}>".format(av), ABS, EOS]
#             else:
#                 Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
#             Xi += [PAD]*(self.max_tokens-len(Xi))
#             X_scan += Xi

#         # Construct X_av from list of AVs in report
#         X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]

#         # Construct Y_scan from 5% of held-out tokens
#         Y_scan = []
#         for i, av in enumerate(self.avs):
#             Y_idx = Y_idxs[i]
#             if Y_idx > 0:
#                 Y_scan.append(X_scan[i*self.max_tokens+Y_idx])

#         # MASK any tokens in pred_tokens 80% of the time
#         # 10% of the time, replace with a random token
#         # 10% of the time, leave the token alone
#         rand_nums = np.random.randint(0, 100, size=self.num_avs*self.max_tokens)
#         for i, tok in enumerate(X_scan):
#             if tok in pred_tokens:
#                 if rand_nums[i] < 80:
#                     X_scan[i] = MASK
#                 elif rand_nums[i] < 90:
#                     X_scan[i] = self.token_vocab_rev[random.randint(5, self.vocab_size-1)]

#         # Convert X_scan to tensor of characters
#         X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
#         for tok in X_scan:
#             # print("TOK")
#             # print(X_scan)
#             # exit(0)
#             X_scan_char = np.concatenate((X_scan_char, self.tok_to_tensor(tok).reshape(1, -1)))

#         # Convert to LongTensor
#         X_scan = torch.as_tensor(X_scan_char)
#         X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
#         Y_scan = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_scan])
#         Y_idxs = torch.LongTensor(Y_idxs)
#         Y_label = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_label])
#         Y_av = torch.LongTensor([self.av_vocab[Y_av]-1])

#         return X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av, md5, scan_date


class FinetuneDataset(AVScanDataset):

    def __init__(self, data_dir, max_tokens):
        """AVScan2Vec dataset class for fine-tuning."""
        super().__init__(data_dir, max_tokens=max_tokens)

        # Load idxs of similar files
        similar_idx_path = os.path.join(data_dir, "similar_ids.pkl")
        with open(similar_idx_path, "rb") as f:
            similar_idxs = pickle.load(f)
        self.similar_idxs = {idx1: idx2 for idx1, idx2 in similar_idxs}
        self.num_reports = len(self.similar_idxs.keys())

    def __getitem__(self, idx):

        # Parse scan reports
        av_tokens_anc, md5, _, _, scan_date = self.parse_scan_report(idx)
        idx_pos = self.similar_idxs[idx]
        av_tokens_pos, md5_pos, _, _, _ = self.parse_scan_report(idx_pos)

        # Construct X_scan_anc and X_scan_pos
        X_scan_anc = []
        X_scan_pos = []
        for av in self.avs:
            if av_tokens_anc.get(av) is None:
                Xi_anc = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi_anc = ["<SOS_{}>".format(av)] + av_tokens_anc[av] + [EOS]
            if av_tokens_pos.get(av) is None:
                Xi_pos = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi_pos = ["<SOS_{}>".format(av)] + av_tokens_pos[av] + [EOS]
            Xi_anc += [PAD]*(self.max_tokens-len(Xi_anc))
            X_scan_anc += Xi_anc
            Xi_pos += [PAD]*(self.max_tokens-len(Xi_pos))
            X_scan_pos += Xi_pos

        # Convert X_scan_anc and X_scan_pos to tensors of characters
        X_scan_char_anc = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        X_scan_char_pos = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        for tok in X_scan_anc:
            X_scan_char_anc = np.concatenate((X_scan_char_anc, self.tok_to_tensor(tok).reshape(1, -1)))
        for tok in X_scan_pos:
            X_scan_char_pos = np.concatenate((X_scan_char_pos, self.tok_to_tensor(tok).reshape(1, -1)))

        # Construct X_av_anc, X_av_pos from lists of AVs in reports
        X_av_anc = [av if av_tokens_anc.get(av) is not None else NO_AV for av in self.avs]
        X_av_pos = [av if av_tokens_pos.get(av) is not None else NO_AV for av in self.avs]

        X_scan_anc = torch.as_tensor(X_scan_char_anc)
        X_av_anc = torch.LongTensor([self.av_vocab[av] for av in X_av_anc])
        X_scan_pos = torch.as_tensor(X_scan_char_pos)
        X_av_pos = torch.LongTensor([self.av_vocab[av] for av in X_av_pos])
        return X_scan_anc, X_av_anc, X_scan_pos, X_av_pos, md5, md5_pos, scan_date


    def __len__(self):
        return self.num_reports
