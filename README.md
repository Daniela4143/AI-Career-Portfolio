# AI Career Portfolio

這個作品集收錄六個以 Python 建置的資料科學與生成式 AI 專案，涵蓋金融資料探索、監督式機器學習、RAG 文件問答、LoRA 微調、保險判決分析，以及具自我查核流程的 RAG Agent。各專案可獨立閱讀、安裝與執行；詳細設計、成果截圖與模組細節請參閱各資料夾中的 README。

## 專案總覽

| 專案 | 簡介 | 主要技術 |
| --- | --- | --- |
| [01 Stock Analysis EDA](./01_Stock_Analysis_EDA/) | 從 Yahoo Finance 取得 SPY 歷史股價，計算日報酬、移動平均與日內區間，並視覺化趨勢及分布。 | Python、Pandas、yfinance、Matplotlib、Seaborn |
| [02 Linear Regression ML](./02_Linear_Regression_ML/) | 以合成資料建立線性迴歸流程，示範資料切分、特徵標準化、模型評估與殘差分析。 | Python、NumPy、Pandas、scikit-learn、Matplotlib |
| [03 RAG Chatbot System](./03_RAG_Chatbot_System/) | 逐步演進的 PDF 知識問答系統，提供 CLI、Streamlit 對話介面與 Docker 化版本，支援對話脈絡與來源檢索。 | LangChain、ChromaDB、Hugging Face、Gemini API、Streamlit、Docker |
| [04 LoRA Fine-Tuning](./04_LoRA_FineTuning_Project/) | 使用 LoRA 微調 DistilBERT 進行 IMDb 電影評論情感分類，並提供推論、網頁介面及 API 入口。 | PyTorch、Transformers、PEFT、Datasets、FastAPI、Streamlit |
| [05 Insurance Judgment AI](./05_InsuranceJudgmentAI/) | 對判決文字進行個資遮蔽、結構化資訊擷取，並透過向量檢索比對相似案例及評估金額偏離。 | spaCy、Ollama、Llama 3、ChromaDB、Sentence Transformers |
| [06 Modular Self-RAG Agent](./06_Modular_Self_RAG_Agent/) | 以可替換的 LLM 引擎進行知識檢索、回答生成、相關性篩選與忠實度查核。 | Python、ChromaDB、Ollama、Gemini API、Hugging Face Datasets |

## 技術棧

- **語言與資料處理：** Python、Pandas、NumPy
- **機器學習與 NLP：** scikit-learn、PyTorch、Hugging Face Transformers、PEFT、spaCy
- **生成式 AI 與檢索：** Gemini API、Ollama / Llama 3、LangChain、ChromaDB、Sentence Transformers
- **視覺化與介面：** Matplotlib、Seaborn、Streamlit、FastAPI
- **部署：** Docker

本儲存庫沒有共用的根目錄依賴設定；請依欲執行的專案安裝該資料夾中的 `requirements.txt` 或 `requirement.txt`。建議使用 Python 3.11，並為不同專案建立獨立虛擬環境，避免套件版本互相影響。

## 環境安裝

以下以 Windows PowerShell 為例。先進入要執行的專案資料夾，再建立並啟用虛擬環境：

```powershell
cd 01_Stock_Analysis_EDA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安裝該專案的依賴（以第一個專案為例）：

```powershell
pip install -r requirements.txt
```

其他作業系統請使用相應的虛擬環境啟用方式；每個專案都應在自己的資料夾與虛擬環境中安裝依賴。

### 外部服務與額外準備

- **RAG Chatbot（v1、v2、v3）：** 需要有效的 `GEMINI_API_KEY`。在各版本程式所在資料夾建立 `.env`，內容如下：
  ```env
  GEMINI_API_KEY=請填入你的_Gemini_API_Key
  ```
  v3 提供 `.env.example` 可作為範本。v1、v2 的 `requirements.txt` 未明列 PDF 讀取及 Hugging Face 嵌入套件，安裝後請補裝 `pip install pypdf sentence-transformers`。
- **LoRA 微調：** 首次訓練會下載 IMDb 資料集及 DistilBERT 模型；訓練時間與硬體資源視環境而異。推論介面需要已產生的 `results_lora_ft/final_lora_adapter`。
- **Insurance Judgment AI：** 安裝 spaCy 中文 Transformer 模型並準備 Ollama：
  ```powershell
  python -m spacy download zh_core_web_trf
  ollama pull llama3
  ```
  專案預設讀取 `data/raw_documents/text1.txt`，執行前請將待分析的純文字判決內容放在該路徑。
- **Modular Self-RAG Agent：** 預設使用 Gemini。建立 `.env` 並設定 `GEMINI_API_KEY`（可參考 `.env.example`）；程式也提供 Ollama 引擎實作。

## 使用方式

各資料夾是獨立專案。完成相應依賴安裝與外部服務設定後，在該專案目錄執行：

### 01｜股票資料探索

```powershell
python stock_eda.py
```

程式會連線 Yahoo Finance 下載 SPY 資料；股票代號與日期範圍可在 `stock_eda.py` 的 `main()` 中調整。

### 02｜線性迴歸

```powershell
python ml_regression.py
```

程式會產生合成迴歸資料、訓練模型、列印 RMSE 與 R²，並顯示預測及殘差圖。

### 03｜RAG Chatbot

進入欲執行的版本資料夾，安裝該版本的依賴並設定 Gemini API Key：

```powershell
# CLI 版本
cd 03_RAG_Chatbot_System\v1_basic_cli
python rag_chatbot.py

# Streamlit 網頁版本（v2 或 v3）
cd ..\v2_streamlit_web
streamlit run app.py
```

Docker 版本請在 `v3_docker_cloud` 目錄準備 `.env`，再建置並啟動容器：

```powershell
# 先切換到 v3_docker_cloud 目錄
cd ..\v3_docker_cloud
docker build -t rag-chatbot .
docker run --rm -p 8080:8080 --env-file .env rag-chatbot
```

### 04｜LoRA 微調與推論

在專案目錄安裝依賴後，可選擇訓練模型或啟動已訓練 Adapter 的介面：

```powershell
python lora_finetune.py
streamlit run web_gui.py
```

另可執行 `python lora_inference.py` 進行命令列推論，或執行 `python app.py` 啟動 FastAPI。推論程式需要 `results_lora_ft/final_lora_adapter` 已存在。

### 05｜保險判決分析

確認 Ollama 正在執行、已安裝 `llama3`，且 `data/raw_documents/text1.txt` 已備妥後：

```powershell
python insure_check.py
```

### 06｜Modular Self-RAG Agent

安裝 `requirement.txt` 並設定 Gemini API Key 後：

```powershell
python agent.py
```

程式會載入 SciQ 資料集，建立範例知識庫並執行檢索、生成與回答查核。

## 目錄結構

```text
AI-Career-Portfolio/
├── README.md
├── 01_Stock_Analysis_EDA/
│   ├── README.md
│   ├── stock_eda.py
│   ├── requirements.txt
│   └── images/
├── 02_Linear_Regression_ML/
│   ├── README.md
│   ├── ml_regression.py
│   ├── requirements.txt
│   └── images/
├── 03_RAG_Chatbot_System/
│   ├── v1_basic_cli/       # README、CLI 程式、requirements、images
│   ├── v2_streamlit_web/   # README、Streamlit app、requirements、images
│   └── v3_docker_cloud/    # README、Dockerfile、Streamlit app、requirements、data
├── 04_LoRA_FineTuning_Project/
│   ├── README.md
│   ├── lora_finetune.py
│   ├── lora_inference.py
│   ├── app.py              # FastAPI 推論 API
│   ├── web_gui.py          # Streamlit 推論介面
│   ├── requirements.txt
│   ├── images/
│   └── results_lora_ft/    # 訓練輸出與 LoRA Adapter
├── 05_InsuranceJudgmentAI/
│   ├── README.md
│   ├── insure_check.py
│   ├── requirement.txt
│   ├── data/
│   │   ├── raw_documents/
│   │   └── processed_documents/
│   ├── chroma_db/
│   └── images/
└── 06_Modular_Self_RAG_Agent/
    ├── README.md
    ├── agent.py
    ├── requirement.txt
    ├── .env.example
    └── images/
```

`images/` 存放專案展示圖；`data/` 存放文件資料；`chroma_db/` 與 `results_lora_ft/` 為向量資料庫及模型訓練產物。部分資料、資料庫或模型輸出可能需依各專案說明自行準備或執行程式產生。

## 注意事項

- API 金鑰請放在本機 `.env` 或環境變數中，勿提交至版本控制。
- RAG 與微調專案首次執行可能下載模型、資料集或建立向量索引，需預留網路、磁碟與記憶體空間。
- 股票資料下載仰賴網路連線及 Yahoo Finance 資料服務。
- 各專案的使用細節與展示成果請參閱其資料夾內的 README。
