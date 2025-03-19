import torch
from transformers import BertTokenizer, BertModel, PreTrainedModel
from torchcrf import CRF
import torch.nn as nn

# Định nghĩa lại mô hình tương thích với Hugging Face Trainer
class BERT_BiLSTM_CRF(PreTrainedModel):
    def __init__(self, config, num_labels, hidden_dim=256, dropout=0.3):
        super().__init__(config)
        
        # Load pre-trained BERT model
        self.bert = BertModel(config)
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
            return {"loss": loss, "logits": emissions}
        else:
            return {"logits": emissions}