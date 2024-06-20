import tiktoken

encoder = tiktoken.get_encoding("o200k_base")

print(encoder.encode("hello world"))
print(encoder.decode(encoder.encode("hello world")))