import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import T5Model, T5PreTrainedModel

class T5CRF(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.t5 = T5Model(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)  # Project to tag space
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Only use encoder for token classification
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)

        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            # Training: compute CRF loss
            loss = -self.crf(emissions, labels, mask=mask, reduction='token_mean')
            return {"loss": loss, "logits": emissions}
        else:
            # Inference: decode tags
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}
