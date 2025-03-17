import torch 
import torch.nn as nn
from TorchCRF import CRF
from transformers import BertModel, BertPreTrainedModel, BertConfig

class Bert_CRF(BertPreTrainedModel):
    def __init__(self, bert_model_name, num_labels, hidden_dim=256, dropout=0.3):
        bert_config = BertConfig.from_pretrained(bert_model_name)
        super().__init__(bert_config)

        #Bert model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        
        #linear layer
        self.fc = nn.Linear(self.bert.config.hidden_size,self.num_labels)

        #CRF layer
        self.crf = CRF(self.num_labels)
    
    def forward(self,input_ids, attention_mask, labels = None):
        #lấy đầu ra từ bert
        output = self.bert(input_ids=input_ids, attention_mask = attention_mask)
        sequence_ouput = output.last_hidden_state
        sequence_ouput = self.dropout(sequence_ouput)

        #tính logits qua lớp fully connected
        logits = self.fc(sequence_ouput)

        predictions = self.crf.viterbi_decode(logits, mask=attention_mask.bool())
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            # Nếu loss không phải là scalar, thực hiện giảm xuống 1 giá trị duy nhất
            if loss.dim() > 0:
                loss = loss.mean()
            return {"loss": loss, "logits": logits, "predictions": predictions}
        else:
            return {"logits": logits, "predictions": predictions}





