#other models for text classification
import random

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import XLNetModel, XLNetTokenizer
from transformers.modeling_utils import SequenceSummary

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers import RobertaTokenizer, RobertaModel

class RandomDataSampler(torch.utils.data.Sampler):
    def __init__(self, num_samples, total_samples):
        index_permutation = [i for i in range(total_samples)]
        random.shuffle(index_permutation)
        self.mask = index_permutation[:num_samples]

    def __iter__(self):
        return (i for i in self.mask)

    def __len__(self):
        return len(self.mask)


class AnotherNet(nn.Module):
    def __init__(self, model_name="XLNet", num_labels=2):
        super().__init__()
        self.model_name = model_name
        if model_name == "XLNet":
            self.transformer = XLNetModel.from_pretrained('xlnet-base-cased')
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        elif model_name == "Roberta":
            self.transformer = RobertaModel.from_pretrained('roberta-base')
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        self.config = self.transformer.config
        self.config.num_labels = num_labels

        if model_name == "XLNet":
            self.classification_head = nn.Sequential(
                SequenceSummary(self.config),
                nn.Linear(self.config.d_model, self.config.num_labels)
            )
        elif model_name == "Roberta":
            self.classification_head = RobertaClassificationHead(self.config)
    
    def forward(self, input_ids):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs = self.transformer(input_ids)
        output = transformer_outputs[0]
        logits = self.classification_head(output)

        return logits



