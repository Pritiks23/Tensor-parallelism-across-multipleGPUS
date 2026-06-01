from transformers import GPT2TokenizerFast

_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def encode(text):
    return _tokenizer.encode(text, return_tensors="pt")

def decode(ids):
    return _tokenizer.decode(ids, skip_special_tokens=True)
