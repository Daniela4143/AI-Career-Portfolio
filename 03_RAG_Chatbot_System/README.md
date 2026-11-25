# RAG 文件問答系統

本資料夾收錄三個逐步演進的 PDF 知識問答版本。依需求選擇 CLI、網頁介面或 Docker 部署版本：

| 版本 | 簡介 | 專案說明 |
| --- | --- | --- |
| **v1｜命令列版** | 以 Gemini、ChromaDB 和本機嵌入模型建立 PDF 問答流程，支援來源頁碼。 | [查看 v1 說明](./v1_basic_cli/README.md) |
| **v2｜Streamlit 網頁版** | 提供聊天網頁介面，加入多輪對話記憶與來源頁碼。 | [查看 v2 說明](./v2_streamlit_web/README.md) |
| **v3｜Docker 部署版** | 延伸 Streamlit 網頁版，提供 Docker 建置與容器執行設定。 | [查看 v3 說明](./v3_docker_cloud/README.md) |

三個版本皆以 PDF 文件作為知識來源；各版本的安裝方式、設定需求與執行步驟，請參閱對應的專案說明。