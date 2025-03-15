import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, bert_model_name, num_labels, hidden_dim=256, dropout=0.3):
        super(BERT_BiLSTM_CRF, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Pass through BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # Shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Fully connected layer
        emissions = self.fc(self.dropout(lstm_output))  # Shape: (batch_size, seq_len, num_labels)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.byte())
            return predictions
