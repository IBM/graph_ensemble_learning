# this provides an implementation of different Language Generation models
from abc import ABC

import torch
from torch.nn.functional import log_softmax
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from torch.nn import DataParallel, CrossEntropyLoss
from transformers.utils import logging
from .utils import prepare_data

logger = logging.get_logger(__name__)


class LG(torch.nn.Module, ABC):
    def __init__(self,
                 model_name,
                 max_source_length,
                 max_target_length,
                 target_pad_id=-100,
                 model_type='t5'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        if model_type == 'bart':
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.pad_token_id = self.model.config.pad_token_id
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.target_pad_id = target_pad_id

        self.parallel()

        self.model.to(self.device)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def forward(self, batch_source, batch_target, ret_batch_loss='loss'):
        """
        Forward function
        :param batch_source: a list of source sequences ["I like monkey", "How are you"]
        :param batch_target: a list of target sequences ["ok", "Hello, I am fine!"]
        :param ret_batch_loss: whether to reduce the loss with mean function
        :return: mean CE loss or batch loss each for a models in the batch
        """
        source_ids, source_mask = prepare_data(batch_source, self.tokenizer, self.max_source_length)
        target_ids, target_mask = prepare_data(batch_target, self.tokenizer, self.max_target_length, self.target_pad_id)
        outputs = self.model(input_ids=source_ids,
                             attention_mask=source_mask,
                             decoder_input_ids=None,
                             decoder_attention_mask=target_mask,
                             labels=target_ids,
                             return_dict=True)

        loss = outputs.loss
        if ret_batch_loss == 'loss':
            return loss.mean()
        elif ret_batch_loss == 'batch_loss':
            masked = target_ids != -100
            batch_size = masked.shape[0]
            logits = outputs.logits
            loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
            masked_loss = loss_fct(logits.permute(0, 2, 1), target_ids)
            masked_loss = torch.cat([masked_loss[i][masked[i]].mean().unsqueeze(0) for i in range(batch_size)])
            return masked_loss
        else:
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=2)
            masked = target_ids != -100
            sequence_scores = []
            for i in range(probs.shape[0]):
                indices = target_ids[i][masked[i]].unsqueeze(-1)
                prob = probs[i][masked[i]]
                prob_gather = torch.gather(prob, 1, indices)
                sequence_scores.append(prob_gather.mean(dim=0))
            sequence_scores = torch.cat(sequence_scores)
            return sequence_scores, outputs.loss

    def generate(self, texts, max_decoder_batch_size=256, num_beams=4, num_ret_seq=1):
        start_idx = 0
        results = []
        for j in range(len(texts)):
            if j % max_decoder_batch_size == max_decoder_batch_size - 1 or j == len(texts) - 1:
                end_idx = j + 1
                input_ids, attention_mask = prepare_data(texts[start_idx:end_idx], self.tokenizer, self.max_source_length)
                # Generate
                if torch.cuda.device_count() > 1:
                    model = self.model.module
                else:
                    model = self.model
                outs = model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      max_length=self.max_target_length,
                                      early_stopping=True,
                                      num_beams=num_beams,
                                      num_return_sequences=num_ret_seq)
                results.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs])
                start_idx = end_idx
        return results

    def generate_and_score(self, texts, max_decoder_batch_size, num_beams=4, num_ret_seq=1):

        input_ids, attention_mask = prepare_data(texts, self.tokenizer, self.max_source_length)
        repeat_input_ids = torch.cat([ids.unsqueeze(0).repeat(num_ret_seq, 1) for ids in input_ids])
        repeat_attention_mask = torch.cat([ids.unsqueeze(0).repeat(num_ret_seq, 1) for ids in attention_mask])

        generated_texts = self.generate(texts, max_decoder_batch_size, num_beams, num_ret_seq)
        sequence_scores = []
        start_idx = 0
        for j in range(len(generated_texts)):
            if j % max_decoder_batch_size == max_decoder_batch_size-1 or j == len(generated_texts)-1:
                end_idx = j + 1
                generated_ids, generated_attention_mask = prepare_data(generated_texts[start_idx:end_idx],
                                                                       self.tokenizer,
                                                                       self.max_target_length,
                                                                       self.target_pad_id)

                outputs = self.model(input_ids=repeat_input_ids[start_idx:end_idx],
                                     attention_mask=repeat_attention_mask[start_idx:end_idx],
                                     decoder_input_ids=None,
                                     decoder_attention_mask=generated_attention_mask,
                                     labels=generated_ids,
                                     return_dict=True)

                logits = outputs.logits
                log_probs = log_softmax(logits, dim=2)

                masked = generated_ids != self.target_pad_id

                for i in range(log_probs.shape[0]):
                    indices = generated_ids[i][masked[i]].unsqueeze(-1)
                    log_prob = log_probs[i][masked[i]]
                    log_prob_gather = torch.gather(log_prob, 1, indices)
                    sequence_scores.append(log_prob_gather.mean(dim=0))
                start_idx = end_idx
            j += 1
        sequence_scores = torch.cat(sequence_scores)
        return generated_texts, sequence_scores

    def parallel(self):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = DataParallel(self.model)
            self.model.to(self.device)
