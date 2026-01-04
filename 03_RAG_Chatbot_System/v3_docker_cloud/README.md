功能展示 (Demo)
1. 專業介面與系統配置 (UI/UX Upgrade)
(images/docker_UIUpgrade.png)

2. 深度對話記憶測試 (Contextual Memory Demo)
驗證系統對上下文的理解能力，而不僅僅是單次檢索。

操作流：

問：BERT 論文中提到的 MLM 是什麼？
(images/ContextualMemoryDemo1.png)
追問：那這種方式有什麼缺點嗎？ (省略了主語 MLM)
(images/ContextualMemoryDemo2.png)

3. 多維度 RAG 效能驗證
針對專業論文內容進行精確度與邊界測試。

精確度測試：詢問 SQuAD v1.1 的 F1 分數，AI 成功從論文摘要提取出 93.2。
(images/docker3.png)
(images/docker4.png)
邊界測試 (Edge Case)：詢問 Docker 安裝方法，AI 誠實回答，證明有效防止幻覺。
(images/docker5.png)

🐳 開發者日誌：DevOps 與部署 (Deployment)
A. Docker 映像檔構建
(images/docker1.png)

B. 容器化運行環境
(images/docker2.png)