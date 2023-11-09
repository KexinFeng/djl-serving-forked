from typing import Tuple, List

from transformers import AutoTokenizer

class DetokenizedTritonResponse:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prefix_offset = 0
        self.read_offset = 0
        self.all_input_ids = []

    def decode_token(self) -> str:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.tokenizer.decode(
            self.all_input_ids[self.prefix_offset:self.read_offset], skip_special_tokens=False
        )
        new_text = self.tokenizer.decode(
            self.all_input_ids[self.prefix_offset:], skip_special_tokens=False
        )
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            new_text = new_text[len(prefix_text):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.all_input_ids)
            return new_text
        else:
            return ""

    def fetch(self, id):
        self.all_input_ids.append(id)
        return self.decode_token()

def decode_token(
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
) -> Tuple[str, int, int]:
    """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
    # The prefix text is necessary only to defeat cleanup algorithms in the decode
    # which decide to add a space or not depending on the surrounding ids.
    prefix_text = tokenizer.decode(
        all_input_ids[prefix_offset:read_offset], skip_special_tokens=False
    )
    new_text = tokenizer.decode(
        all_input_ids[prefix_offset:], skip_special_tokens=False
    )

    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        new_text = new_text[len(prefix_text) :]
        return new_text, read_offset, len(all_input_ids)
    else:
        return "", prefix_offset, read_offset

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")
input_txt = "Hello, this is Qing speaking to Rohith. How are you? I introduced a new tokenizing way for our model. As the sun dipped below the horizon, painting the in hues of crimson and gold, a tranquil stillness settled over the landscape. The night awakened with a symphony of crickets and the soft glow of stars overhead. The birds chirped and the wind"
feeder = tokenizer(input_txt).input_ids
all_input_ids = []
prefix_offset = 0
read_offset = 0

effective_decoding = []
effective_decoding_pair = []
not_effective_decoding = []
not_effective_decoding_pair = []
toolkit_decoding = []
response = DetokenizedTritonResponse(tokenizer)

for i, id in enumerate(feeder):
    all_input_ids.append(id)
    # toolkit_decoding.append(response.fetch(id))
    # Generated token
    next_token_text, prefix_offset, read_offset = decode_token(
        all_input_ids,
        prefix_offset,
        read_offset,
    )
    effective_decoding.append(next_token_text)
    effective_decoding_pair.append((i, next_token_text))
    not_effective_decoding.append(tokenizer.decode(id))
    not_effective_decoding_pair.append((i, tokenizer.decode(id)))

print(f"Effective Decoding {''.join(effective_decoding)}")
print(f"not effective Decoding {' '.join(not_effective_decoding)}")
print(f"Toolkit Decoding {''.join(toolkit_decoding)}")
print(f"Ground truth {tokenizer.decode(all_input_ids)}")

print(f"Effective Decoding {effective_decoding_pair}")
print(f"not effective Decoding {not_effective_decoding_pair}")
