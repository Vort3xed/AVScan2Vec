from bpe import Encoder

test_corpus = ""

with open('/home/agneya/AVScan2Vec/avscan2vec/test_corpus.txt', 'r') as file:
    test_corpus = file.read()

encoder = Encoder(388616, pct_bpe=1.20)  # params chosen for demonstration purposes

encoder.fit(test_corpus.split('\n'))

example = "breaking bad is a really good show im not even going to lie!"
example2 = ["breaking", "bad", "is", "a", "really", "good", "show", "im", "not", "even", "going", "to", "lie!"]
example3 = "trojan"
example4 = ["trojan"]
# print(encoder.tokenize(example))

print("normal bpe tokenization using a space")
print(next(encoder.transform([example3])))
print(encoder.tokenize(example3))

# print("tokenization using a list of words (the same sentence)")
# print(next(encoder.transform(example2)))

token_list = []
for word in example4:
    token_list += next(encoder.transform([word]))

print("tokenization using a list of tokens")
print(token_list)


"""
X_TOKEN: attribute
torch.Size([20])
tensor([117, 203, 450, 303,   6,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0])
X_TOKEN: inject
torch.Size([20])
tensor([ 47, 358, 243,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0])
X_TOKEN: trojan
torch.Size([20])
tensor([203, 334,  55,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0])
X_TOKEN: win32
torch.Size([20])
tensor([374, 831,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0])
X_TOKEN: generic
torch.Size([20])
tensor([45, 46, 51, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0])
"""