import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils import set_seed, prepare_dataset
from newmodels import BERT_BiLSTM_CRF
from constants import *
import wandb, huggingface_hub, evaluate
from transformers import TrainingArguments, Trainer

# Set seed for reproducibility
set_seed(SEED)

# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels

# Load dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(TOKENIZER_BERT_BC)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE_BERT_BC, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE_BERT_BC, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE_BERT_BC, shuffle=False, collate_fn=collate_fn)




# Load checkpoint if available
def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None  # Không có checkpoint nếu thư mục chưa tồn tại
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs(EXPERIMENT_RESULTS_BBC_DIR, exist_ok=True)

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_BBC_DIR)
if checkpoint:
    model = BERT_BiLSTM_CRF.from_pretrained(checkpoint)
else:
    model = BERT_BiLSTM_CRF(MODEL_BERT_BC, num_labels=len(ID2LABEL))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setup Training Arguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_BERT_BC,
    report_to="wandb",
    evaluation_strategy='steps',
    save_strategy='steps',
    eval_steps=EVAL_STEPS_BERT_BC,
    save_steps=SAVE_STEPS_BERT_BC,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_BERT_BC,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_BERT_BC,
    num_train_epochs=NUM_TRAIN_EPOCHS_BERT_BC,
    weight_decay=WEIGHT_DECAY_BERT_BC,
    learning_rate=LR_BERT_BC, 
    output_dir=EXPERIMENT_RESULTS_BBC_DIR,
    logging_dir=EXPERIMENT_RESULTS_BBC_DIR + "/logs",
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    seed=SEED,
    max_grad_norm=1.0
)

# Define compute_metrics function
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Chuyển preds về danh sách nếu là numpy array đơn lẻ
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    decoded_labels = []
    decoded_preds = []

    for label_seq, pred_seq in zip(labels, preds):
        if isinstance(label_seq, float) or isinstance(pred_seq, float):
            continue  # Bỏ qua các giá trị không hợp lệ

        current_labels = []
        current_preds = []

        for label, pred in zip(label_seq, pred_seq):
            if label != 31:  # 31 là token padding, cần bỏ qua
                current_labels.append(ID2LABEL.get(label, "O"))
                current_preds.append(ID2LABEL.get(pred, "O"))

        decoded_labels.append(current_labels)
        decoded_preds.append(current_preds)
    
    return metric.compute(predictions=decoded_preds, references=decoded_labels)



# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Training loop (trainer.train() already handles this)
trainer.train()

# Evaluation
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# Save model, tokenizer, and training results
model.save_pretrained(EXPERIMENT_RESULTS_BBC_DIR)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_BBC_DIR)

# Save the training arguments and test results
with open(EXPERIMENT_RESULTS_BBC_DIR + "/training_args.txt", "w") as f:
    f.write(str(training_args))

with open(EXPERIMENT_RESULTS_BBC_DIR + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_BERT_BC_DIR,
    repo_id="auphong2707/nlp-ner",
    repo_type="model",
    private=False
)
