import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import TaskType
from transformers import Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv # 微調：增加 dotenv 支援

# 初始化環境
load_dotenv()

# Check PyTorch and CUDA availability
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Setting - guarantee CPU run on the CPU / handle GPU / CUDA checkup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- running device: {DEVICE} ---")

# --- 1. Configuration Area ---
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"
OUTPUT_DIR = "./results_lora_ft"
# ---------------------------------

def load_and_preprocess_data(tokenizer):
    print("--- 1. Loading and preprocessing data ---")
    dataset = load_dataset(DATASET_NAME)

    # 保持你原始的數量設定
    train_data = dataset["train"].shuffle(seed=42).select(range(10000))
    eval_data = dataset["test"].shuffle(seed=42).select(range(1000))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_train_data = train_data.map(tokenize_function, batched=True)
    tokenized_eval_data = eval_data.map(tokenize_function, batched=True)

    tokenized_train_data = tokenized_train_data.remove_columns(['text'])
    tokenized_eval_data = tokenized_eval_data.remove_columns(["text"])
    tokenized_train_data = tokenized_train_data.rename_column("label", "labels")
    tokenized_eval_data = tokenized_eval_data.rename_column("label", "labels")

    print(f"training data length: {len(tokenized_train_data)}, evaluated data length:{len(tokenized_eval_data)}")
    return tokenized_train_data, tokenized_eval_data

def setup_lora_model(model_name: str):
    print("--- 2. Model loading and LoRA configuration ---")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        device_map="auto" if DEVICE == "cuda" else None
    )

    # 保持你原始的 LoRA 設定 (r=4)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}

def train_lora_model(model, train_data, eval_data, tokenizer):
    print("--- 3. Setup training arguments and start training ---")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_log_dir = os.path.join(OUTPUT_DIR, f"logs/{current_time}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        logging_dir=unique_log_dir,
        overwrite_output_dir=True,
        logging_steps=10,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit" if DEVICE == "cuda" else "adamw_torch",
        learning_rate=2e-5,
        fp16=True if DEVICE == "cuda" else False,
        report_to="tensorboard" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("\n--- Starting LoRA fine-tuning ---")
    trainer.train()

    # 存檔路徑確保正確：./results_lora_ft/final_lora_adapter
    final_output_dir = os.path.join(OUTPUT_DIR, "final_lora_adapter")
    trainer.model.save_pretrained(final_output_dir)
    print(f"✅ LoRA adapter weights saved in: {final_output_dir}")

    eval_results = trainer.evaluate()
    print(f"\n--- Final evaluation results ---\n{eval_results}")

    # 微調：手動釋放記憶體避免爆掉
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data, eval_data = load_and_preprocess_data(tokenizer)
    model = setup_lora_model(MODEL_NAME)
    train_lora_model(model, train_data, eval_data, tokenizer)

if __name__ == "__main__":
    main()