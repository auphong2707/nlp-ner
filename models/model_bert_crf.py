from torchcrf import CRF
from transformers import BertForTokenClassification, BertPreTrainedModel

class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Use the BertForTokenClassification model as the base
        self.bert = BertForTokenClassification(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        emissions = outputs.logits # Shape: (batch_size, sequence_length, num_labels)

        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            # Calculate the negative log likelihood loss
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            # Get the predicted tags
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}