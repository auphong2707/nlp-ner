from constants import *
from utils import set_seed, prepare_dataset
from newmodels import BERT_BiLSTM_CRF
import wandb, huggingface_hub, os
import evaluate
from transformers import TrainingArguments, Trainer, AutoTokenizer
from torch.nn.utils import clip_grad_norm_

# Set random seed for reproducibility
set_seed(SEED)

# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare the dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(TOKENIZER_BERT_BC)

# Define compute_metrics function
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_labels = []
    decoded_preds = []
    for label_seq, pred_seq in zip(labels, preds):
        current_labels = []
        current_preds = []
        for label, pred in zip(label_seq, pred_seq):
            if label != 31:  # 31 is for padding
                current_labels.append(ID2LABEL[label])
                current_preds.append(ID2LABEL[pred])
        decoded_labels.append(current_labels)
        decoded_preds.append(current_preds)
    
    return metric.compute(predictions=decoded_preds, references=decoded_labels)

# Setup results directory
os.makedirs(EXPERIMENT_RESULTS_BBC_DIR, exist_ok=True)

# Load model
model = BERT_BiLSTM_CRF.from_pretrained(MODEL_BERT_BC, num_labels=len(ID2LABEL), ignore_mismatched_sizes=True)

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
    fp16=False,
    seed=SEED,
    max_grad_norm=1.0
)

# Define the preprocess function for logits
def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
