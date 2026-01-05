# 🚀 AI Career Portfolio

這是我在 **AI 與數據科學領域** 的實作專案集，涵蓋從基礎數據分析、傳統機器學習到尖端的大語言模型（LLM）微調與 RAG 系統開發。

> 💡 **成果展示：** 詳細的技術文件、代碼實作與運行截圖，請點擊下方各專案資料夾查看。

---

## 🛠️ 技術棧 (Tech Stack)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-FFD21E?style=for-the-badge)

---

## 📂 專案列表 (Projects)

### 📈 Project 1: Stock Analysis EDA
* **核心技術：** `Python`, `Pandas`, `Matplotlib`, `Seaborn`
* **專案目標：** 實作自動化數據清洗管線，並透過技術指標視覺化（如 MA 移動平均、KDJ 指標）深度挖掘市場潛在趨勢。
* **關鍵成果：** 建立了一套可重複使用的金融數據探索性分析（EDA）框架。
* [👉 查看詳情](./01_Stock_Analysis/)

### 🤖 Project 2: Linear Regression ML
* **核心技術：** `Scikit-Learn`, `NumPy`, `Feature Engineering`
* **專案目標：** 實作端對端的機器學習流程，包含特徵工程處理與房價趨勢預測，並建立嚴格的模型評估指標（MSE/R2）。
* **關鍵成果：** 理解並實作了迴歸模型在實際業務場景中的預測邏輯。
* [👉 查看詳情](./02_Linear_Regression/)

### 💬 Project 3: RAG Chatbot System (V3)
* **技術亮點：** `LangChain`, `ChromaDB`, `Gemini API`, `Docker`
* **功能特色：** * **知識溯源**：支援 PDF 文本檢索並標註來源頁碼。
    * **對話記憶**：內建 `ConversationBufferMemory` 實現連貫的多輪對話。
    * **容器化部署**：透過 Docker 實現一鍵部署，具備高移植性與環境隔離特點。
* [👉 查看詳情](./03_RAG_Chatbot/)

### 🧪 Project 4: LoRA Fine-Tuning
* **技術亮點：** `PEFT (LoRA)`, `HuggingFace Transformers`, `PyTorch`
* **功能特色：** * **參數高效微調**：實作 LoRA 演算法，僅需訓練 **0.98%** 的參數即完成模型訓練。
    * **情感分析專家**：將 `DistilBERT` 成功微調至 IMDb 任務，達成 **86.3%** 的卓越準確率。
    * **硬體優化**：展示了如何在有限運算資源下訓練大型語言模型。
* [👉 查看詳情](./04_LoRA_FineTuning/)

---
