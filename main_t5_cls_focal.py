from utils.constants import *
from utils.functions import set_seed, prepare_dataset_t5
from utils.focal_loss_trainer import FocalLossTrainer

import os, wandb, huggingface_hub, evaluate
import numpy as np
import torch
from transformers import T5ForTokenClassification, AdamW, TrainingArguments, DataCollatorForTokenClassification

# Set seed for reproducibility
set_seed(SEED)

# Login to Weights & Biases and HuggingFace
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Load dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5_CLS_FOCAL)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Define evaluation metric
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=-1)
    decoded_preds, decoded_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        p_seq, l_seq = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                p_seq.append(ID2LABEL[p])
                l_seq.append(ID2LABEL[l])
        decoded_preds.append(p_seq)
        decoded_labels.append(l_seq)
    return metric.compute(predictions=decoded_preds, references=decoded_labels)

def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)

# Load model
os.makedirs(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL, exist_ok=True)

def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL)
if checkpoint:
    model = T5ForTokenClassification.from_pretrained(checkpoint)
else:
    model = T5ForTokenClassification.from_pretrained(MODEL_T5_CLS_FOCAL,
                                                     num_labels=NUM_LABELS,
                                                     ignore_mismatched_sizes=True)

# Optimizer and custom scheduler
optimizer = AdamW(model.parameters(), lr=LR_T5_CLS_FOCAL)

total_steps = len(train_dataset) * NUM_TRAIN_EPOCHS_T5_CLS_FOCAL

class LinearDecayWithMinLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, min_lr, max_steps, last_epoch=-1):
        self.min_lr = min_lr
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        lr_decay = max(0, (1 - step / self.max_steps)) * (self.base_lrs[0] - self.min_lr) + self.min_lr
        return [lr_decay] * len(self.base_lrs)

scheduler = LinearDecayWithMinLR(optimizer, min_lr=1e-6, max_steps=total_steps)

# TrainingArguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_T5_CLS_FOCAL,
    report_to="wandb",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=EVAL_STEPS_T5_CLS_FOCAL,
    save_steps=SAVE_STEPS_T5_CLS_FOCAL,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_T5_CLS_FOCAL,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_T5_CLS_FOCAL,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_T5_CLS_FOCAL,
    learning_rate=LR_T5_CLS_FOCAL,
    weight_decay=WEIGHT_DECAY_T5_CLS_FOCAL,
    num_train_epochs=NUM_TRAIN_EPOCHS_T5_CLS_FOCAL,
    lr_scheduler_type="linear",  # Still required by HF API
    output_dir=EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL,
    logging_dir=EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL + "/logs",
    logging_steps=LOGGING_STEPS_T5_CLS_FOCAL,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    seed=SEED
)

# Trainer
trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    optimizers=(optimizer, scheduler)
    alpha=NER_CLASS_WEIGHTS,   
    gamma=GAMMA,               
    loss_scale=LOSS_SCALE      
)

# Train
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# Evaluate
test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

# Save everything
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL)

with open(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL + "/training_args.txt", "w") as f:
    f.write(str(training_args))

with open(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL + "/test_results.txt", "w") as f:
    f.write(str(test_metrics))

# Upload to Hugging Face Hub
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL,
    repo_id="auphong2707/nlp-ner-t5-focal",
    repo_type="model",
    private=False
)
