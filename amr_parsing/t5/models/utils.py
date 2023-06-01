import torch
from transformers import TensorType

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_data(batch_text, tokenizer, max_length, pad_id=None, return_overflowing_tokens=True):
    """
    Prepare datasets
    :param batch_text: a list of source sequences ["I like monkey", "How are you"]
    :param tokenizer: tokenizer
    :param max_length: maximum source models length
    :param pad_id: pad_id
    :param return_overflowing_tokens: if true return overflowing tokens
    :return: a dict with sequence_ids and sequence_attention_mask
    """
    input_encodings = tokenizer.batch_encode_plus(batch_text,
                                                  padding=True,
                                                  truncation=True if max_length is not None else False,
                                                  max_length=max_length,
                                                  return_attention_mask=True,
                                                  # return_overflowing_tokens=return_overflowing_tokens,
                                                  return_tensors=TensorType.PYTORCH)

    # Convert to tensors
    input_ids = torch.LongTensor(input_encodings['input_ids']).to(device)
    attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(device)

    if pad_id is not None:
        mask = input_ids == tokenizer.pad_token_id
        input_ids[mask] = pad_id

    return input_ids, attention_mask
