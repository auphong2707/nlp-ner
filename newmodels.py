import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

class BERT_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config, rnn_dim=128, need_birnn=False):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        
        # BERT Model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        
        # BiLSTM layer (if needed)
        self.need_birnn = need_birnn
        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2  # Because it's bidirectional
        
        # Linear layer for token classification
        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        
        # CRF layer
        self.crf = CRF(config.num_labels, batch_first=True)
        
    def forward(self, input_ids, token_type_ids=None, input_mask=None, labels=None):
        # Getting BERT embeddings
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        
        # Applying BiLSTM if needed
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        
        # Apply dropout and linear layer
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        
        # If labels are provided, compute the loss using CRF
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=input_mask.byte())
            return loss
        
        # Return the predictions from CRF decoding
        return self.crf.decode(emissions, input_mask.byte())

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        # Get predictions from CRF
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        
        # If needed, apply BiLSTM
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        
        return self.crf.decode(emissions, input_mask.byte())
