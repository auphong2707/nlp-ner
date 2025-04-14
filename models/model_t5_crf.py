from torchcrf import CRF
from transformers import T5PreTrainedModel, T5ForTokenClassification

class T5CRF(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # T5 model adapted for token classification
        self.t5 = T5ForTokenClassification(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )

        emissions = outputs.logits  # Shape: (batch_size, seq_len, num_labels)
        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='token_mean')
            return {"loss": loss, "logits": emissions}
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}
