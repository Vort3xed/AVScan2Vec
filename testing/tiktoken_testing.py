import tiktoken
from tiktoken._educational import *

sentence = "breaking bad is a great show."

tokenizer = tiktoken.get_encoding("o200k_base")
# tokenizer = SimpleBytePairEncoding.from_tiktoken("cl100k_base")

print(tokenizer.encode(sentence))

# print(tokenizer.decode([tokenizer.encode(sentence)[0]]))
# print(tokenizer.decode([tokenizer.encode(sentence)[1]]))
# print(tokenizer.decode([tokenizer.encode(sentence)[2]]))
# print(tokenizer.decode([tokenizer.encode(sentence)[3]]))
# print(tokenizer.decode([tokenizer.encode(sentence)[4]]))
# print(tokenizer.decode([tokenizer.encode(sentence)[5]]))