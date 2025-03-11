from constants import *
from utils import set_seed, prepare_dataset_t5
set_seed(SEED)

import wandb, huggingface_hub, os
import evaluate
from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration, AutoTokenizer

# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare dataset and tokenizer for T5
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5)

# Define metric
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Chuyển chuỗi thành danh sách token
    decoded_preds = [pred.strip().split() for pred in decoded_preds]
    decoded_labels = [label.strip().split() for label in decoded_labels]

    # Loại bỏ "PAD" chỉ khi đánh giá (không làm trong quá trình huấn luyện)
    filtered_preds = [[token for token in pred if token != "PAD"] for pred in decoded_preds]
    filtered_labels = [[token for token in label if token != "PAD"] for label in decoded_labels]

    return metric.compute(predictions=filtered_preds, references=filtered_labels)


# Create results directory
os.makedirs(EXPERIMENT_RESULTS_DIR_T5, exist_ok=True)

# Load T5 model
checkpoint = None
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5)

if checkpoint:
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
else:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_T5)
    model.gradient_checkpointing_enable()

# Create Training Arguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_T5,
    report_to="wandb",
    evaluation_strategy='steps',
    save_strategy='steps',
    eval_steps=EVAL_STEPS_T5,
    save_steps=SAVE_STEPS_T5,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_T5,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_T5,
    num_train_epochs=NUM_TRAIN_EPOCHS_T5,
    weight_decay=WEIGHT_DECAY_T5,
    learning_rate=LR_T5, 
    output_dir=EXPERIMENT_RESULTS_DIR_T5,
    logging_dir=EXPERIMENT_RESULTS_DIR_T5 + "/logs",
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    seed=SEED
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
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
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5)

# Save training arguments
with open(EXPERIMENT_RESULTS_DIR_T5 + "/training_args.txt", "w") as f:
    f.write(str(training_args))

# Save test results
with open(EXPERIMENT_RESULTS_DIR_T5 + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_T5_DIR,
    repo_id="auphong2707/nlp-ner",
    repo_type="model",
    private=False
)
