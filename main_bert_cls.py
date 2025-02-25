from constants import *
from utils import set_seed, prepare_dataset
set_seed(SEED)

import wandb, huggingface_hub, os
import evaluate
from transformers import TrainingArguments, Trainer, BertForTokenClassification, AutoTokenizer

# [PREPARING DATASET AND FUNCTIONS]
# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare the dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(TOKENIZER_BERT)

# Define compute_metrics function
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions.shape)
    
    true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
    pred_labels = [[ID2LABEL[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    
    results = metric.compute(predictions=pred_labels, references=true_labels)
    
    # Extract entity-level scores
    precision = results["overall_precision"]
    recall = results["overall_recall"]
    f1 = results["overall_f1"]
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# [SETTING UP MODEL AND TRAINING ARGUMENTS]
# Set experiment name
EXPERIMENT_NAME = "bert-ner-experiment"
EXPERIMENT_RESULTS_DIR = RESULTS_BERT_DIR + EXPERIMENT_NAME
os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)

# Load model
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR)
if checkpoint:
    model = BertForTokenClassification.from_pretrained(checkpoint)
else:
    model = BertForTokenClassification.from_pretrained(MODEL_BERT_NER,
                                                       num_labels=len(ID2LABEL),
                                                       ignore_mismatched_sizes=True)

# Create training arguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME,
    report_to="wandb",
    evaluation_strategy='steps',
    save_strategy='steps',
    eval_steps=30,
    save_steps=30,
    learning_rate=LR_BERT,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_BERT,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_BERT,
    num_train_epochs=NUM_TRAIN_EPOCHS_BERT,
    weight_decay=WEIGHT_DECAY_BERT,
    output_dir=EXPERIMENT_RESULTS_DIR,
    logging_dir=EXPERIMENT_RESULTS_DIR + "/logs",
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    seed=SEED
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=TOKENIZER_BERT,
    preprocess_logits_for_metrics=(lambda logits, labels: (logits[0].argmax(dim=-1), labels)),
    compute_metrics=compute_metrics,
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
model.save_pretrained(EXPERIMENT_RESULTS_DIR)
TOKENIZER_BERT.save_pretrained(EXPERIMENT_RESULTS_DIR)

# Save the training arguments
with open(EXPERIMENT_RESULTS_DIR + "/training_args.txt", "w") as f:
    f.write(str(training_args))

# Save the test results
with open(EXPERIMENT_RESULTS_DIR + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_BERT_DIR,
    repo_id="auphong2707/bert-ner",
    repo_type="model",
    private=True
)
