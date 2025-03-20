import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs  # Catch any extra arguments
    ):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]  # [batch_size, seq_length, hidden_size]
        sequence_output = self.dropout(sequence_output)

        # Pass through classifier to get emissions
        emissions = self.classifier(sequence_output)  # [batch_size, seq_length, num_labels]

        if labels is not None:
            # Training: Mask is 1 only where labels != -100
            crf_mask = (labels != -100).long()  # Convert boolean to 0/1 tensor
            tags = labels.clone()
            tags[labels == -100] = 0    
            loss = -self.crf(emissions, tags, mask=crf_mask, reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            # Prediction: Use attention_mask
            mask = attention_mask.bool() if attention_mask is not None else None
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}