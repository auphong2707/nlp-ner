from utils.constants import *
from utils.functions import set_seed, prepare_dataset_t5

import numpy as np
from collections import defaultdict
import wandb, huggingface_hub, os
import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

set_seed(SEED)

# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare dataset and tokenizer for T5 (now returning text-labels for seq2seq)
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5_CLS_CE)

# Data collator cho T5 seq2seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,  # sẽ tự động cập nhật khi trainer khởi tạo
    padding="max_length",
    return_tensors="pt"
)

# Metric setup
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_list = [pred.strip().split() for pred in decoded_preds]
    label_list = [label.strip().split() for label in decoded_labels]

    filtered_preds, filtered_labels = [], []
    for p, l in zip(pred_list, label_list):
        if len(p) == len(l):
            filtered_preds.append(p)
            filtered_labels.append(l)

    if len(filtered_preds) == 0:
        return {"overall_f1": 0.0}

    results = metric.compute(predictions=filtered_preds, references=filtered_labels, zero_division=0)

    wandb.log({
        "eval/overall_f1": results.get("overall_f1", 0.0),
        "eval/overall_precision": results.get("overall_precision", 0.0),
        "eval/overall_recall": results.get("overall_recall", 0.0),
    })

    class_metrics = defaultdict(list)
    for key, val in results.items():
        if "_" in key and key.endswith("_f1"):
            label_type = key.split("_")[0]      
            _, ent_type = label_type.split("-")
            class_metrics[ent_type].append(val)

    for ent_type, f1s in class_metrics.items():
        avg_f1 = np.mean(f1s)
        wandb.log({f"eval/{ent_type}_f1": avg_f1})

    return results  

# Create results directory
os.makedirs(EXPERIMENT_RESULTS_DIR_T5_CLS_CE, exist_ok=True)

# Load or resume model
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5_CLS_CE)

if checkpoint:
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
else:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_T5_CLS_CE)
    model.gradient_checkpointing_enable()

model.to("cuda")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    run_name=EXPERIMENT_NAME_T5_CLS_CE,
    report_to="wandb",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=EVAL_STEPS_T5_CLS_CE,
    save_steps=SAVE_STEPS_T5_CLS_CE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_T5_CLS_CE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_T5_CLS_CE,
    num_train_epochs=NUM_TRAIN_EPOCHS_T5_CLS_CE, 
    weight_decay=WEIGHT_DECAY_T5_CLS_CE,
    learning_rate=LR_T5_CLS_CE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_T5_CLS_CE,
    output_dir=EXPERIMENT_RESULTS_DIR_T5_CLS_CE,
    logging_dir=os.path.join(EXPERIMENT_RESULTS_DIR_T5_CLS_CE, "logs"),
    logging_steps=LOGGING_STEPS_T5_CLS_CE,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    predict_with_generate=True,  
    fp16=True,
    seed=SEED,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# Evaluate
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# Save model and tokenizer
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CLS_CE)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CLS_CE)

# Save training arguments
with open(EXPERIMENT_RESULTS_DIR_T5_CLS_CE + "/training_args.txt", "w") as f:
    f.write(str(training_args))

# Save test results
with open(EXPERIMENT_RESULTS_DIR_T5_CLS_CE + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face Hub
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_DIR_T5_CLS_CE,
    repo_id="auphong2707/nlp-ner",
    repo_type="model",
    private=False
)
