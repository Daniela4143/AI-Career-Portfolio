# 🚀 AI Career Portfolio

這是我在 **AI 與數據科學領域** 的實作專案集，涵蓋從基礎數據分析、傳統機器學習到大語言模型（LLM）微調與 RAG 系統開發。

> 💡 **成果展示：** 詳細的技術文件、代碼實作與運行截圖，請點擊下方各專案資料夾查看。

---

## 🛠️ 技術棧 (Tech Stack)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-FFD21E?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google-gemini&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
---

## 📂 專案列表 (Projects)

### 📈 Project 1: Stock Analysis EDA
* **核心技術：** `Python`, `Pandas`, `Matplotlib`, `Seaborn`
* **專案目標：** 實作自動化數據清洗管線，並透過技術指標視覺化（如 MA 移動平均、KDJ 指標）挖掘市場潛在趨勢。
* **關鍵成果：** 建立了一套可重複使用的金融數據探索性分析（EDA）框架。
* [👉 查看詳情](./01_Stock_Analysis_EDA/)

### 🤖 Project 2: Linear Regression ML
* **核心技術：** `Scikit-Learn`, `NumPy`, `Feature Engineering`
* **專案目標：** 實作端對端的機器學習流程，包含特徵工程處理與房價趨勢預測，並建立模型評估指標（MSE/R2）。
* **關鍵成果：** 理解並實作了迴歸模型在實際業務場景中的預測邏輯。
* [👉 查看詳情](./02_Linear_Regression_ML/)

### 💬 Project 3: RAG Chatbot System (V3)
* **技術亮點：** `LangChain`, `ChromaDB`, `Gemini API`, `Docker`
* **功能特色：** * **知識溯源**：支援 PDF 文本檢索並標註來源頁碼。
    * **對話記憶**：內建 `ConversationBufferMemory` 實現連貫的多輪對話。
    * **容器化部署**：透過 Docker 實現一鍵部署，具備高移植性與環境隔離特點。
* [👉 查看詳情](./03_RAG_Chatbot_System/v3_docker_cloud/)

### 🧪 Project 4: LoRA Fine-Tuning
* **技術亮點：** `PEFT (LoRA)`, `HuggingFace Transformers`, `PyTorch`
* **功能特色：** * **參數高效微調**：實作 LoRA 演算法，僅需訓練 **0.98%** 的參數即完成模型訓練。
    * **情感分析專家**：將 `DistilBERT` 成功微調至 IMDb 任務，達成 **86.3%** 的卓越準確率。
    * **硬體優化**：展示了如何在有限運算資源下訓練大型語言模型。
* [👉 查看詳情](./04_LoRA_FineTuning_Project/)

### 🛡️ Project 5: InsureCheck-AI (產險智能審閱)
* **技術亮點：** `Llama 3`, `SpaCy (Transformer NER)`, `ChromaDB`, `Regex`
* **功能特色：** * **合規去識別化**：整合 NLP 實體辨識與正則表達式，自動遮蔽判決書中的敏感個資（當事人姓名、地址等）。
    * **結構化數據提取**：利用 LLM JSON Mode 將非結構化法學文本精準轉化為理賠特徵（判賠金額、核定理由）。
    * **風險預警引擎**：透過 RAG 檢索相似歷史案例，自動計算申請金額與市場行情之偏離比例，識別異常高額賠付風險。
* [👉 查看詳情](./05_InsuranceJudgmentAI/)

### 🧠 Project 6: Modular Self-RAG Agent (多引擎檢索增強生成系統)
* **技術亮點：** `Python (OOP/ABC)`, `ChromaDB`, `Ollama (Llama 3)`, `Gemini 2.5 Flash`, `Self-RAG`
* **功能特色：** * **模組化策略模式**：實作 `ABC (Abstract Base Class)` 定義 LLM 介面，支援 Ollama 與 Gemini 雙引擎熱插拔切換。
    * **Self-RAG 驗證機制**：內建三階段可靠性過濾：
        * **檢索過濾 (Retrieval Grader)**：透過 Vector Distance 門檻攔截不相關問題。
        * **幻覺檢測 (Faithfulness Check)**：利用 LLM 作為審查員進行事實查核，確保回答內容忠於參考文本。
    * **模型能力評測**：對比 Llama 3 與 Gemini 在邏輯推論上的表現差異，實作「事實查核員」閉環邏輯。
* [👉 查看詳情](./06_Module_Self_RAG_Agent/)

---
