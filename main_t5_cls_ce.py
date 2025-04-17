from utils.constants import *
from utils.functions import set_seed, prepare_dataset_t5
set_seed(SEED)

import numpy as np
import wandb, huggingface_hub, os
import evaluate
from transformers import TrainingArguments, Trainer, T5ForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification

# Login to wandb & Hugging Face
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare dataset and tokenizer for T5
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_t5(TOKENIZER_T5_CLS_CE)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Define metric
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.nan_to_num(logits)
    predictions = np.argmax(logits, axis=-1)

    # Kiểm tra nếu toàn bộ labels là -100 (không có nhãn hợp lệ)
    if all(all(l == -100 for l in label) for label in labels):
        print("⚠️ Toàn bộ labels là -100 trong batch hiện tại — bỏ qua batch này.")
        return {
            "overall_f1": 0.0,
            "overall_precision": 0.0,
            "overall_recall": 0.0,
        }

    # Chuyển prediction và labels sang dạng tên tag
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]

    # Tính toán metric bằng thư viện seqeval
    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)

    # Nếu F1 tụt về 0, in ra để kiểm tra
    if results.get("overall_f1", 0.0) < 0.01:
        print("⚠️ F1-score gần 0. Kiểm tra một vài prediction:")
        print("Pred:", true_predictions[:1])
        print("Label:", true_labels[:1])

    # Trả về tất cả metric, nhưng loại bỏ 3 key không mong muốn
    filtered_results = {
        k: v for k, v in results.items()
        if k not in {"f1", "precision", "recall"}
    }

    # Giữ lại các metric tổng thể để dùng trong eval
    filtered_results["overall_f1"] = results.get("overall_f1", 0.0)
    filtered_results["overall_precision"] = results.get("overall_precision", 0.0)
    filtered_results["overall_recall"] = results.get("overall_recall", 0.0)

    return filtered_results

# Create results directory
os.makedirs(EXPERIMENT_RESULTS_DIR_T5_CLS_CE, exist_ok=True)

# Load T5 model
checkpoint = None
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5_CLS_CE)

if checkpoint:
    model = T5ForTokenClassification.from_pretrained(checkpoint, num_labels=NUM_LABELS)
else:
    model = T5ForTokenClassification.from_pretrained(MODEL_T5_CLS_CE, num_labels=NUM_LABELS)
    model.gradient_checkpointing_enable()

model.to("cuda")

# Create Training Arguments
training_args = TrainingArguments(
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
    logging_dir=EXPERIMENT_RESULTS_DIR_T5_CLS_CE + "/logs",
    logging_steps=LOGGING_STEPS_T5_CLS_CE,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    save_total_limit=2,
    greater_is_better=True,
    fp16=True,
    seed=SEED,
)

# Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
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

# Upload to Hugging Face
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_DIR_T5_CLS_CE,
    repo_id="auphong2707/nlp-ner",
    repo_type="model",
    private=False
)
