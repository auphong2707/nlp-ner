import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]  # [batch_size, seq_length, hidden_size]
        temp_sequence_output = sequence_output
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # [batch_size, seq_length, num_labels]

        if labels is not None:
            crf_mask = (labels != 31).bool()  # Convert boolean to 0/1 tensor
            tags = labels.clone()
            tags[labels == 31] = 0    
            loss = -self.crf(emissions, tags, mask=crf_mask, reduction='mean')
            if not self.training:
                # Evaluation mode: compute predictions
                mask = attention_mask.bool() if attention_mask is not None else None
                predictions = self.crf.decode(emissions, mask=mask)
                # Pad predictions to max_length
                batch_size, max_len, _ = emissions.shape
                pred_tensor = torch.full(
                    (batch_size, max_len), 31, dtype=torch.long, device=emissions.device
                )
                for i, pred in enumerate(predictions):
                    if len(pred) > 0:
                        pred_tensor[i, :len(pred)] = torch.tensor(pred, dtype=torch.long, device=emissions.device)
                return {"loss": loss, "pred_tensor": pred_tensor}
            else:
                # Training mode
                return {"loss": loss, "logits": emissions}
        else:
            # Inference mode
            mask = attention_mask.bool() if attention_mask is not None else None
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}