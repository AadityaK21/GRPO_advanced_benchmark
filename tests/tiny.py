"""Tiny stand-ins for Qwen2.5 and its tokenizer, built from a config, so every test runs offline in seconds.

The model is a real Qwen2 architecture (the same code as Qwen2.5-0.5B), just with 2 layers, width 32
and a 64-token vocabulary. The tokenizer splits on spaces: token "tK" has id K + 2.
"""
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

VOCAB = 64
PAD = 0
EOS = 1


def tid(k):
    """Id of token "tK"."""
    return k + 2


def make_tokenizer():
    vocab = {"<pad>": PAD, "<eos>": EOS, **{f"t{i}": tid(i) for i in range(VOCAB - 2)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


def make_model(seed=0, logit_scale=1.0):
    """A tiny Qwen2 in fp32 with no dropout. logit_scale > 1 makes its next-token distributions peakier."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=False, attn_implementation="eager", pad_token_id=PAD, eos_token_id=EOS,
    )
    model = Qwen2ForCausalLM(config).float().eval()
    if logit_scale != 1.0:
        with torch.no_grad():
            model.lm_head.weight.mul_(logit_scale)
    model.generation_config.pad_token_id = PAD
    model.generation_config.eos_token_id = EOS
    return model
