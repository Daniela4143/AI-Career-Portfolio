## 🖥️ 運行展示 (Demo)

### 1. 系統初始化
系統會檢查 `.env` 環境變數並加載持久化的 Chroma 向量資料庫。
![System Init](systemInitialization.png)

### 2. 知識庫問答與溯源
AI 根據 BERT 論文內容詳細回答 Masked LM 的運作機制，並自動列出知識來源頁碼。
![RAG QA](coreRAGDemo.png)

### 3. 邊界測試 (防幻覺)
當問題超出 PDF 知識庫範圍時，AI 會依據 Prompt 規範拒絕回答，避免產生錯誤資訊。
![Edge Case](AntiHallucinationTest.png)