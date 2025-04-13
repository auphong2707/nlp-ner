import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config, PreTrainedModel
from torchcrf import CRF

class T5CRF(PreTrainedModel):
    config_class = T5Config  # để hỗ trợ từ HuggingFace API

    def __init__(self, config):
        super().__init__(config)

        self.encoder = T5EncoderModel(config)
        self.hidden2tag = nn.Linear(config.d_model, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        emissions = self.hidden2tag(sequence_output)  # (batch_size, seq_len, num_labels)

        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            labels = labels.long()  # ⚠️ ép kiểu về long trước khi dùng CRF
            # Training mode: compute CRF loss
            loss = -self.crf(emissions, labels, mask=mask, reduction='token_mean')
            return {"loss": loss, "logits": emissions}
        else:
            # Inference mode: decode best path
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}
