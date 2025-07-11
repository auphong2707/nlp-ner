from utils.constants import *
from utils.functions import set_seed, prepare_dataset_t5  # NOTE: use T5-specific data prep

# Set seed for reproducibility
set_seed(SEED)

import torch
import wandb, huggingface_hub, os, platform
import evaluate
from transformers import TrainingArguments, T5Config

from models.model_t5_crf import T5CRF  # Your custom model

"[PREPARING DATASET AND FUNCTIONS]"
if platform.system() != "Windows":
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Load dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5_CRF)

metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    decoded_labels, decoded_preds = [], []
    for label_seq, pred_seq in zip(labels, preds):
        current_labels, current_preds = [], []
        for label, pred in zip(label_seq, pred_seq):
            if label != -100:
                current_labels.append(ID2LABEL[label])
                current_preds.append(ID2LABEL[pred])
        decoded_labels.append(current_labels)
        decoded_preds.append(current_preds)

    return metric.compute(predictions=decoded_preds, references=decoded_labels)

"[SETTING UP MODEL AND TRAINING ARGUMENTS]"
os.makedirs(EXPERIMENT_RESULTS_DIR_T5_CRF, exist_ok=True)

def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

config = T5Config.from_pretrained(MODEL_T5_CRF)
config.num_labels = NUM_LABELS
checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5_CRF)

if checkpoint:
    model = T5CRF.from_pretrained(checkpoint, config=config)
else:
    model = T5CRF(config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.compile(model)

training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_T5_CRF,
    report_to="wandb",
    eval_strategy='steps',
    save_strategy='steps',
    eval_steps=EVAL_STEPS_T5_CRF,
    save_steps=SAVE_STEPS_T5_CRF,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_T5_CRF,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_T5_CRF,
    num_train_epochs=NUM_TRAIN_EPOCHS_T5_CRF,
    weight_decay=WEIGHT_DECAY_T5_CRF,
    learning_rate=LR_T5_CRF,
    output_dir=EXPERIMENT_RESULTS_DIR_T5_CRF,
    logging_dir=EXPERIMENT_RESULTS_DIR_T5_CRF + "/logs",
    logging_steps=LOGGING_STEPS_T5_CRF,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    seed=SEED
)

from transformers import Trainer

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
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

"[SAVING THINGS]"
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CRF)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5_CRF)

with open(EXPERIMENT_RESULTS_DIR_T5_CRF + "/training_args.txt", "w") as f:
    f.write(str(training_args))

with open(EXPERIMENT_RESULTS_DIR_T5_CRF + "/test_results.txt", "w") as f:
    f.write(str(test_results))

if platform.system() != "Windows":
    api = huggingface_hub.HfApi()
    api.upload_large_folder(
        folder_path=RESULTS_DIR_T5_CRF,
        repo_id="auphong2707/nlp-ner",
        repo_type="model",
        private=False
    )
