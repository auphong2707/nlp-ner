from utils.constants import *
from utils.functions import set_seed, prepare_dataset_t5
from utils.focal_loss_trainer import FocalLossTrainer

set_seed(SEED)

import numpy as np
import wandb, huggingface_hub, os
import evaluate
from transformers import (
    TrainingArguments, T5ForTokenClassification,
    AutoTokenizer, DataCollatorForTokenClassification
)

# Login
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5_CLS_FOCAL)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Metrics
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.nan_to_num(logits)
    predictions = np.argmax(logits, axis=-1)

    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    wandb.log({
        "eval/precision": results.get("precision", 0.0),
        "eval/recall": results.get("recall", 0.0),
        "eval/f1": results.get("f1", 0.0),
    })
    return results

# Make result dir
os.makedirs(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL, exist_ok=True)

# Check for resume
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL)

# Load model
if checkpoint:
    model = T5ForTokenClassification.from_pretrained(checkpoint, num_labels=NUM_LABELS)
else:
    model = T5ForTokenClassification.from_pretrained(MODEL_T5_CLS_FOCAL, num_labels=NUM_LABELS)
    model.gradient_checkpointing_enable()

model.to("cuda")

# Training args
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_T5_CLS_FOCAL,
    report_to="wandb",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=EVAL_STEPS_T5_CLS_FOCAL,
    save_steps=SAVE_STEPS_T5_CLS_FOCAL,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_T5_CLS_FOCAL,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_T5_CLS_FOCAL,
    num_train_epochs=NUM_TRAIN_EPOCHS_T5_CLS_FOCAL, 
    weight_decay=WEIGHT_DECAY_T5_CLS_FOCAL,
    learning_rate=LR_T5_CLS_FOCAL, 
    output_dir=EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL,
    logging_dir=EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL + "/logs",
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_T5_CLS_FOCAL,
    logging_steps=LOGGING_STEPS_T5_CLS_FOCAL,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    save_total_limit=2,
    greater_is_better=True,
    fp16=True,
    seed=SEED,
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
    alpha=NER_CLASS_WEIGHTS,
    gamma=GAMMA,
    loss_scale=LOSS_SCALE,
)

# Train
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# Evaluate
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# Save model/tokenizer
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL)

# Save args
with open(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL + "/training_args.txt", "w") as f:
    f.write(str(training_args))

# Save test results
with open(EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL,
    repo_id="auphong2707/nlp-ner-focal",
    repo_type="model",
    private=False
)
