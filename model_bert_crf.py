import torch
import torch.nn as nn
from torchcrf import CRF  # Sử dụng pytorch-crf thay vì torch.CRF
from transformers import BertModel, BertPreTrainedModel, BertConfig

class Bert_CRF(BertPreTrainedModel):
    def __init__(self, bert_model_name, num_labels, hidden_dim=256, dropout=0.3):
        bert_config = BertConfig.from_pretrained(bert_model_name)
        super().__init__(bert_config)

        # Bert model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim  # Không dùng trực tiếp trong code này, nhưng giữ lại nếu bạn muốn thêm layer trung gian
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer để chuyển từ hidden_size của BERT sang num_labels
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)  # batch_first=True để phù hợp với định dạng đầu vào của BERT
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Lấy đầu ra từ BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        sequence_output = self.dropout(sequence_output)

        # Tính logits qua lớp fully connected
        logits = self.fc(sequence_output)  # [batch_size, seq_length, num_labels]

        # Chuyển attention_mask sang kiểu bool (yêu cầu của pytorch-crf)
        mask = attention_mask.bool()
        # return logits

        # Giải mã chuỗi tối ưu bằng Viterbi
        predictions = self.crf.decode(logits, mask=mask)  # Trả về list các chuỗi nhãn tốt nhất

        if labels is not None:
            # Tính negative log-likelihood loss từ CRF
            loss = -self.crf(logits, labels, mask=mask, reduction='mean')  # reduction='mean' để trả về scalar
            return {"loss": loss, "logits": logits, "predictions": predictions}
        else:
            return {"logits": logits, "predictions": predictions}

# # Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo mô hình
    model = Bert_CRF("bert-base-uncased", num_labels=5)

    # Tạo dữ liệu giả lập
    batch_size, seq_length = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    labels = torch.randint(0, 5, (batch_size, seq_length))  # Nhãn giả lập

    # Gọi forward với labels
    output = model(input_ids, attention_mask, labels)
    print("Loss:", output["loss"])
    print("Predictions:", output["predictions"])

    # Gọi forward không có labels
    output = model(input_ids, attention_mask)
    print("Predictions:", output["predictions"])
    print("Shape of logits:", output["logits"].shape)