import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.mm_utils import tokenizer_image_token

__all__ = [
    "tokenize_conversation",
    "preprocess_conversation",
    "infer_stop_tokens",
]


SENTINEL = "[VILA-SENTINEL]"
DUMMY_CONVERSATION = [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "answer"},
] * 10


def tokenize_conversation_legacy(
    messages: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    overrides: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Skip the first message if it is not from human
    if messages[0]["from"] != "human":
        messages = messages[1:]

    conv.messages = []
    for turn, message in enumerate(messages):
        role = roles[message["from"]]
        assert role == conv.roles[turn % 2]
        if overrides is not None and message["from"] in overrides:
            conv.append_message(role, overrides[message["from"]])
        else:
            conv.append_message(role, message["value"])

    return tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt")


def tokenize_conversation(
    messages: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    overrides: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    supported_tokenizers = ["qwen2tokenizer"]
    if not any(name in tokenizer.__class__.__name__.lower() for name in supported_tokenizers):
        return tokenize_conversation_legacy(messages, tokenizer, overrides=overrides)

    conversation = []
    for m in messages:
        message = {}
        if m["from"] == "human":
            message["role"] = "user"
        elif m["from"] == "gpt":
            message["role"] = "assistant"
        else:
            raise ValueError(f"Unexpected sender '{m['from']}' in conversation entry.")

        message["content"] = m["value"]
        if overrides is not None and m["from"] in overrides:
            message["content"] = overrides[m["from"]]
        conversation.append(message)

    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    return tokenizer_image_token(text, tokenizer, return_tensors="pt")


def preprocess_conversation(
    conversation: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, Any]:
    inputs = tokenize_conversation(conversation, tokenizer)
    labels = torch.ones_like(inputs) * IGNORE_INDEX

    # Generate the template by replacing the assistant's response with a sentinel.
    template = tokenize_conversation(conversation, tokenizer, overrides={"gpt": SENTINEL})
    sentinel = tokenizer(SENTINEL, add_special_tokens=False).input_ids
    sentinel = torch.tensor(sentinel, dtype=template.dtype)

    # Remove sentinel tokens from the template.
    mask = torch.ones_like(template, dtype=torch.bool)
    for k in range(template.size(0) - sentinel.size(0)):
        if torch.equal(template[k : k + sentinel.size(0)], sentinel):
            mask[k : k + sentinel.size(0) + 1] = False  # +1 to include the stop token
    template = template[mask]

    # Match the tokenized conversation with the template (with no assistant's response).
    # Every token that is not matched will be included in the label for training.
    p = 0
    for k in range(inputs.size(0)):
        if p < template.size(0) and inputs[k] == template[p]:
            p += 1
        else:
            labels[k] = inputs[k]

    # Mask all tokens in the label if the template is not fully matched.
    if p < template.size(0):
        logging.warning("Failed to process the conversation. All tokens will be masked in the label.")
        labels[:] = IGNORE_INDEX

    return {"input_ids": inputs, "labels": labels}


def infer_stop_tokens(tokenizer: transformers.PreTrainedTokenizer) -> List[str]:
    template = tokenize_conversation(DUMMY_CONVERSATION, tokenizer, overrides={"gpt": SENTINEL})
    sentinel = tokenizer(SENTINEL, add_special_tokens=False).input_ids
    sentinel = torch.tensor(sentinel, dtype=template.dtype)

    stop_tokens = {tokenizer.eos_token}
    for k in range(template.size(0) - sentinel.size(0)):
        if torch.equal(template[k : k + sentinel.size(0)], sentinel):
            stop_token = tokenizer.decode(template[k + sentinel.size(0)])
            stop_tokens.add(stop_token)
    return list(stop_tokens)
