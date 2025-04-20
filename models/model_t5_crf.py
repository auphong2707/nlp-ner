from torch import nn
from torchcrf import CRF
from transformers import T5PreTrainedModel, T5ForTokenClassification

class T5CRF(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.t5 = T5ForTokenClassification(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        emissions = outputs.logits  # shape: (batch_size, seq_len, num_labels)

        mask = attention_mask.bool() if attention_mask is not None else None

        # Filter out -100 in labels to avoid index error in CRF
        if labels is not None:
            # Ensure all label values are within [0, num_labels-1]
            # CRF automatically masks with `mask`, so we don't need -100
            labels = labels.clone()
            labels[labels == -100] = 0  # Replace -100 with a valid index, will be masked anyway

            loss = -self.crf(emissions, labels, mask=mask, reduction='token_mean')
            return {"loss": loss, "logits": emissions}
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}
