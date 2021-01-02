#other models for text classification
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import XLNetConfig, XLNetModel, XLNetTokenizer
from transformers.modeling_utils import SequenceSummary

class AnotherNet(nn.Module):
    def __init__(self, model_name="XLNet", num_labels=2):
        super().__init__()
        self.model_name = model_name
        if model_name == "XLNet":
            self.transformer = XLNetModel.from_pretrained('xlnet-base-cased')
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        
        self.config = self.transformer.config
        self.config.num_labels = num_labels
    
        self.sequence_summary = SequenceSummary(self.config)
        self.logits_proj = nn.Linear(self.config.d_model, self.config.num_labels)
    
    def forward(self, input_ids):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs = self.transformer(input_ids)
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        return logits



