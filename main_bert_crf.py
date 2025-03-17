import os

import numpy as np
import torch
from transformers import AutoTokenizer
import wandb, huggingface_hub, evaluate
from transformers import TrainingArguments, Trainer

from model_bert_crf import Bert_CRF
from constants import *
from utils import set_seed, prepare_dataset
# Set seed for reproducibility
torch.set_num_threads(4) 
if __name__ == '__main__':
    set_seed(SEED)


    "[PREPARING DATASET AND FUNCTIONS]"
    # Login to wandb & Hugging Face
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

    # Load dataset and tokenizer
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(TOKENIZER_BERT_CRF)

    # Define compute_metrics function
    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        print(f"preds.shape, labels.shape{preds.shape, labels.shape}")
        # Convert preds and labels to lists if numpy arrays
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        if isinstance(labels,np.ndarray):
            labels = labels.tolist()

        decoded_labels = []
        decoded_preds = []

        for label_seq, pred_seq in zip(labels, preds):
            # if not isinstance(label_seq, list) or not isinstance(pred_seq, list):
            #     continue
            
            current_labels = []
            current_preds = []

            for label, pred in zip(label_seq, pred_seq):
                # if isinstance(label, (float, np.float32)) or isinstance(pred, (float,np.float32)):
                #     continue
                if label!=31:
                    current_labels.append(ID2LABEL.get(label,"0"))
                    current_preds.append(ID2LABEL.get(pred,"0"))
            
            if current_labels and current_preds:
                decoded_labels.append(current_labels)
                decoded_preds.append(current_preds)

        print("Final decoded_preds sample:", decoded_preds[:5])
        print("Final decoded_labels sample:", decoded_labels[:5])

        if not decoded_preds or not decoded_labels:
            print("Warning: No valid predictions or labels found!")
            return {"eval_precision": 0.0, "eval_recall": 0.0, "eval_f1": 0.0}
        
        return metric.compute(predictions=decoded_preds,references=decoded_labels)


    "[SETTING UP MODEL AND TRAINING ARGUMENTS]"
    # Create results directory if not exists
    os.makedirs(EXPERIMENT_RESULTS_BERT_CRF_DIR, exist_ok=True)

    # Load checkpoint if available
    def get_last_checkpoint(ouput_dir):
        if not os.path.exists(ouput_dir):
            return None # No checkpoint if directory does not exist
        checkpoints = [d for d in os.listdir(ouput_dir) if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            return os.path.join(ouput_dir,last_checkpoint)
        return None

    checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_BERT_CRF_DIR)
    if checkpoint:
        model = Bert_CRF.from_pretrained(checkpoint)
    else:
        model = Bert_CRF(MODEL_BERT_CRF,num_labels=len(ID2LABEL),
                        #  ignore_mismatched_sizes=True,
                        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
        output_dir=EXPERIMENT_RESULTS_BERT_CRF_DIR,
        logging_dir=EXPERIMENT_RESULTS_BERT_CRF_DIR + "/logs",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end="eval_overall_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        seed=SEED,
        max_grad_norm=1.0,
        # gradient_accumulation_steps=2,  # Giúp tăng batch size ảo mà không tiêu tốn thêm RAM GPU
        # optim="adamw_torch",  # Dùng AdamW tối ưu hơn
        dataloader_num_workers=4,  # Giúp load dữ liệu nhanh hơn
        dataloader_pin_memory=True,  # Đẩy tensor vào pinned memory giúp CPU -> GPU nhanh hơn
    )

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

    "[TRAINING]"
    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    "[EVALUATING]"
    # Evaluation
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

    "[SAVING THINGS]"
    # save model, tokenizer, and training results
    model.save_pretrained(EXPERIMENT_RESULTS_BERT_CRF_DIR)
    tokenizer.save_pretrained(EXPERIMENT_RESULTS_BERT_CRF_DIR)

    # Save training arguments and test results
    with open(EXPERIMENT_RESULTS_BERT_CRF_DIR+"/training_args.txt", "w") as f:
        f.write(str(training_args))

    with open(EXPERIMENT_RESULTS_BERT_CRF_DIR+"/test_results.txt", "w") as f:
        f.write(str(test_results))

    # Upload to Hugging Face
    api = huggingface_hub.HfApi()
    api.upload_large_folder(
        folder_path=RESULTS_BERT_CRF_DIR,
        repo_id="auphong2707/nlp-ner",
        repo_type="model",
        private=False
    )