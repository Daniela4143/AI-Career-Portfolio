import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# --- 1. 定義模型和路徑 ---
MODEL_NAME = "distilbert-base-uncased"
LORA_ADAPTER_PATH = "./results_lora_ft/final_lora_adapter" # 載入你訓練好的 LoRA 權重
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IMDb 情感分類的標籤（0=負面，1=正面)
ID_TO_LABEL = {0: "Negative 😔", 1: "Positive 😊"}

# --- 2. 載入模型和 Tokenizer ---
print("--- 1. Loading Base Model and Tokenizer ---")
# 載入原始的 DistilBERT 模型
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2, # 這是二元分類
    id2label=ID_TO_LABEL
).to(DEVICE)

# 載入 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 3. 載入 LoRA Adapter ---
print("--- 2. Loading and attaching LoRA Adapter ---")
# 載入 PeftModel（LoRA 模型）
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH).to(DEVICE)

# 將模型設為評估模式
model.eval()

# --- 4. 定義推理函式 ---
def predict_sentiment(text):
    """
    輸入文本，模型預測情感
    """
    print(f"\n[Input Text]: {text}")
    
    # 對文本進行 Tokenization
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True
    ).to(DEVICE)
    
    # 進行推理（關閉梯度計算）
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 取得 logits (輸出分數)
    logits = outputs.logits
    
    # 取得預測的類別 ID
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    # 轉換成人類可讀的標籤
    predicted_label = model.config.id2label[predicted_class_id]
    
    # 輸出結果
    print(f"-> [Prediction]: {predicted_label}")

# --- 5. 運行推理測試 ---
if __name__ == "__main__":
    print(f"--- Running Inference on Device: {DEVICE} ---")

    # 測試用例 1: 正面評論
    predict_sentiment("This movie was absolutely spectacular, a masterpiece of storytelling and visual effects.")
    
    # 測試用例 2: 負面評論
    predict_sentiment("The plot was confusing, the acting was wooden, and the ending was a complete disappointment.")
    
    # 測試用例 3: 模糊評論
    predict_sentiment("It's an okay film, I guess, not the best thing I've ever seen but definitely not the worst.")

    print("\n--- Inference Finished ---")