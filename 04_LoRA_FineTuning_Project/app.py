import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# --- 設定 ---
MODEL_NAME = "distilbert-base-uncased"
LORA_PATH = "./results_lora_ft/final_lora_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ID_TO_LABEL = {0: "Negative 😔", 1: "Positive 😊"}

app = FastAPI(title="嵐嵐的情感分析 API")

# --- 載入模型 (全域) ---
print("--- Loading Model and Adapter... ---")
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# 定義輸入格式
class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    inputs = tokenizer(review.text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    pred_id = torch.argmax(logits, dim=-1).item()
    label = ID_TO_LABEL[pred_id]
    
    return {"text": review.text, "sentiment": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)