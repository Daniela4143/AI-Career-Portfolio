import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# 設定網頁標題
st.set_page_config(page_title="嵐嵐的 AI 影評分析室", page_icon="🎬")

@st.cache_resource # 讓模型只載入一次，避免重複佔用記憶體
def load_model():
    MODEL_NAME = "distilbert-base-uncased"
    LORA_PATH = "./results_lora_ft/final_lora_adapter"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer, DEVICE

# 初始化
st.title("🎬 嵐嵐的 AI 影評分析室")
st.write("輸入一段電影評論，讓我的 LoRA 微調模型幫你分析它的情感！")

model, tokenizer, DEVICE = load_model()
user_input = st.text_area("請輸入英文影評：", placeholder="Type something like 'What a fantastic movie!'")

if st.button("開始分析"):
    if user_input.strip():
        # 推理邏輯
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred_id].item() * 100

        # 顯示結果
        label_map = {0: ("Negative 😔", "red"), 1: ("Positive 😊", "green")}
        label_text, color = label_map[pred_id]
        
        st.subheader(f"預測結果：:{color}[{label_text}]")
        st.info(f"信心指數：{conf:.2f}%")
    else:
        st.warning("請記得輸入文字喔！")