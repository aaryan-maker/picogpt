import tiktoken


class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.eot = self.enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab
