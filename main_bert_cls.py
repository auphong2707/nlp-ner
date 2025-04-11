from utils.constants import *
from utils.functions import set_seed, prepare_dataset
set_seed(SEED)

from utils.focal_loss_trainer import FocalLossTrainer

import wandb, huggingface_hub, os
import evaluate
from transformers import TrainingArguments, Trainer, BertForTokenClassification, AutoTokenizer

# [PREPARING DATASET AND FUNCTIONS]
# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare the dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(TOKENIZER_BERT_CLS)

# Define compute_metrics function
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    print(f"preds.shape: {preds.shape}, labels.shape: {labels.shape}")

    decoded_labels = []
    decoded_preds = []
    for label_seq, pred_seq in zip(labels, preds):
        current_labels = []
        current_preds = []
        for label, pred in zip(label_seq, pred_seq):
            # Filter out padding tokens (commonly set to -100)
            if label != -100:
                # Convert numerical IDs to string labels using your mapping
                current_labels.append(ID2LABEL[label])
                current_preds.append(ID2LABEL[pred])
        decoded_labels.append(current_labels)
        decoded_preds.append(current_preds)
    
    return metric.compute(predictions=decoded_preds, references=decoded_labels)
    

# [SETTING UP MODEL AND TRAINING ARGUMENTS]
# Create experiment results directory
os.makedirs(EXPERIMENT_RESULTS_DIR_BERT_CLS, exist_ok=True)

# Load model
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_BERT_CLS)
if checkpoint:
    model = BertForTokenClassification.from_pretrained(checkpoint)
else:
    model = BertForTokenClassification.from_pretrained(MODEL_BERT_CLS,
                                                       num_labels=NUM_LABELS,
                                                       ignore_mismatched_sizes=True)

# Create training arguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_BERT_CLS,
    report_to="wandb",
    eval_strategy='steps',
    save_strategy='steps',
    eval_steps=EVAL_STEPS_BERT_CLS,
    save_steps=SAVE_STEPS_BERT_CLS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_BERT_CLS,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_BERT_CLS,
    num_train_epochs=NUM_TRAIN_EPOCHS_BERT_CLS,
    weight_decay=WEIGHT_DECAY_BERT_CLS,
    learning_rate=LR_BERT_CLS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_BERT_CLS,
    output_dir=EXPERIMENT_RESULTS_DIR_BERT_CLS,
    logging_dir=EXPERIMENT_RESULTS_DIR_BERT_CLS + "/logs",
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    seed=SEED
)

def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)

# Create Trainer
trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
    alpha=NER_CLASS_WEIGHTS,
    gamma=GAMMA,
    loss_scale=LOSS_SCALE,
)

# [TRAINING]
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# [EVALUATING]
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# [SAVING THINGS]
# Save the model and tokenizer
model.save_pretrained(EXPERIMENT_RESULTS_DIR_BERT_CLS)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_BERT_CLS)

# Save the training arguments
with open(EXPERIMENT_RESULTS_DIR_BERT_CLS + "/training_args.txt", "w") as f:
    f.write(str(training_args))

# Save the test results
with open(EXPERIMENT_RESULTS_DIR_BERT_CLS + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_DIR_BERT_CLS,
    repo_id="auphong2707/nlp-ner",
    repo_type="model",
    private=False
)
