from torchcrf import CRF
from transformers import RobertaForTokenClassification, RobertaPreTrainedModel

class RobertaCRF(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Use the RobertaForTokenClassification model as the base
        self.roberta = RobertaForTokenClassification(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        emissions = outputs.logits

        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            # Calculate the negative log likelihood loss
            tags = labels.clone()
            tags[labels == -100] = 0 
            loss = -self.crf(emissions, tags, mask=mask, reduction='token_mean')            
            return {"loss": loss, "logits": emissions}
        else:
            # Get the predicted tags
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}