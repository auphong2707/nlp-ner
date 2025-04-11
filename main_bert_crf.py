from utils.constants import *
from utils.functions import set_seed, prepare_dataset
# Set seed for reproducibility
set_seed(SEED)

import torch

import wandb, huggingface_hub, os, platform
import evaluate
from transformers import TrainingArguments, Trainer, BertConfig

from models.model_bert_crf import BertCRF

"[PREPARING DATASET AND FUNCTIONS]"
# Login to wandb & Hugging Face
if platform.system() != "Windows":
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Load dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(TOKENIZER_BERT_CRF)

# Define compute_metrics function
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    decoded_labels = []
    decoded_preds = []
    
    for label_seq, pred_seq in zip(labels, preds):
        current_labels = []
        current_preds = []
        for label, pred in zip(label_seq, pred_seq):
            if label != -100:
                current_labels.append(ID2LABEL[label])
                current_preds.append(ID2LABEL[pred])
        # if current_labels and current_preds:
        decoded_labels.append(current_labels)
        decoded_preds.append(current_preds)

    # if not decoded_preds or not decoded_labels:
    #     print("Warning: No valid predictions or labels found!")
    #     e
    #     return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return metric.compute(predictions=decoded_preds, references=decoded_labels)
    
"[SETTING UP MODEL AND TRAINING ARGUMENTS]"
# Create results directory if not exists
os.makedirs(EXPERIMENT_RESULTS_DIR_BERT_CRF, exist_ok=True)

# Load checkpoint if available
def get_last_checkpoint(ouput_dir):
    if not os.path.exists(ouput_dir):
        return None # No checkpoint if direxctory does not exist
    checkpoints = [d for d in os.listdir(ouput_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(ouput_dir,last_checkpoint)
    return None

config = BertConfig.from_pretrained(MODEL_BERT_CRF)
config.num_labels = num_labels=NUM_LABELS   # MODEL_BERT_CRF should be a pretrained model name like "bert-base-uncased"
checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_BERT_CRF)
if checkpoint:
    model = BertCRF.from_pretrained(checkpoint,config=config)
else:
    model = BertCRF(config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.compile(model)  # Compile the model for better performance

# Setup Training Arguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_BERT_CRF,
    report_to="wandb",
    eval_strategy='steps',
    save_strategy='steps',
    eval_steps=EVAL_STEPS_BERT_CRF,
    save_steps=SAVE_STEPS_BERT_CRF,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_BERT_CRF,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_BERT_CRF,
    num_train_epochs=NUM_TRAIN_EPOCHS_BERT_CRF,
    weight_decay=WEIGHT_DECAY_BERT_CRF,
    learning_rate=LR_BERT_CRF,
    output_dir=EXPERIMENT_RESULTS_DIR_BERT_CRF,
    logging_dir=EXPERIMENT_RESULTS_DIR_BERT_CRF + "/logs",
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    seed=SEED
)

# Create Trainer instance
def preprocess_logits_for_metrics(preds, labels):
    return preds.argmax(dim=-1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

"[TRAINING]"
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

"[EVALUATING]"
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test",)

"[SAVING THINGS]"
# save model, tokenizer, and training results
model.save_pretrained(EXPERIMENT_RESULTS_DIR_BERT_CRF)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_BERT_CRF)

# Save training arguments and test results
with open(EXPERIMENT_RESULTS_DIR_BERT_CRF+"/training_args.txt", "w") as f:
    f.write(str(training_args))

with open(EXPERIMENT_RESULTS_DIR_BERT_CRF+"/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to Hugging Face
if platform.system() != "Windows":
    api = huggingface_hub.HfApi()
    api.upload_large_folder(
        folder_path=RESULTS_DIR_BERT_CRF,
        repo_id="auphong2707/nlp-ner",
        repo_type="model",
        private=False
    )