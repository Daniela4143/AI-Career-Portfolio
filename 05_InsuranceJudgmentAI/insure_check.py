import re
import spacy
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import json
import ollama
from pathlib import Path
import os

INPUT_DIR = "data/raw_documents"
OUTPUT_DIR = "data/processed_documents"

# 定義結構化 Prompt
ANALYSIS_PROMPT = """
你是一位資深的保險理賠專家與法律助理。請從下方提供的「去識別化」裁判書文本中，提取關於「精神慰撫金」的資訊。

請遵循以下邏輯：
1. 金額提取：請計算法院最終核定的「精神慰撫金總額」。
2. 理由總結：請用繁體中文簡述法院核定此金額的具體因素（如：過失程度、受害人年齡、家屬痛苦程度、雙方經濟地位）。
3. 格式：務必僅回傳 JSON 格式。

{{
    "amount": 數字,
    "reason_summary": "中文理由總結",
    "case_id": "裁判字號"
}}

文本內容：
{text}
"""

class InsureDataProcessor:
    def __init__(self, whitelist=None):
        try:
            self.nlp = spacy.load("zh_core_web_trf")
        except:
            print("請先執行 python -m spacy download zh_core_web_trf")

        self.role_pattern = r"(上\s*訴\s*人|被\s*上\s*訴\s*人|訴訟代理人|法定代理人|原\s*告|被\s*告)"
        self.whitelist = whitelist if whitelist else {"律師", "法官", "書記官", "上一人", "共同"}
        # 用於切分裁判書大段落的標記
        self.section_markers = {
            "main_judgment": r"\n\s*主\s*文\s*\n",
            "facts_and_reasons": r"\n\s*事實及理由\s*\n",
            "conclusion_start": r"\n\s*中\s*華\s*民\s*國.*\n"
        }

    def split_sections(self, text):
        """將裁判書切分為：Header, 主文, 事實及理由, Footer"""
        sections = {}
        
        # 尋找分隔點
        main_match = re.search(self.section_markers["main_judgment"], text)
        facts_match = re.search(self.section_markers["facts_and_reasons"], text)
        remain_match = re.search(self.section_markers["conclusion_start"], text)
        
        if main_match and facts_match:
            sections["header"] = text[:main_match.start()]
            sections["judgment"] = text[main_match.end():facts_match.start()]
            sections["content"] = text[facts_match.end():remain_match.start()]
            sections["footer"] = text[remain_match.end():]
        else:
            sections["full_text"] = text
        return sections
    
    def get_metadata(self, text):
        """提取裁判書元數據"""
        case_id = self.extract_case_id(text)
        return {
            "case_id": case_id,
            "source": "法院判決書"
        }

    def extract_case_id(self, text):
        pattern = r"(\d+\s*年\s*[\u4e00-\u9fa5]+\s*字第\s*\d+\s*號)"
        match = re.search(pattern, text)
        return match.group(1).replace(" ", "") if match else "Unknown_ID"
    
    def clean_text(self, text):
        """基礎文本標準化"""
        return text.replace("　", " ").strip()
    
    def extract_target_names(self, text):
        """
        從裁判書抬頭提取關鍵人物姓名，對標金融數據去識別化合規要求。
        """
        text = self.clean_text(text)
        target_names = set()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for i, line in enumerate(lines):
            match = re.search(self.role_pattern, line)
            if match:
                role_text = match.group(0)
                # 獲取角色關鍵字後的內容
                raw_candidate = re.sub(f".*{role_text}", "", line).strip()
                
                # 如果該行沒名字，看下一行（處理格式縮排問題）
                if len(raw_candidate) < 2 and i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    if not re.search(self.role_pattern, next_line):
                        raw_candidate = next_line
                
                # 執行深度清洗
                final_name = self._refine_name(raw_candidate)
                if final_name:
                    target_names.add(final_name)
        
        return list(target_names)
    
    def _refine_name(self, name):
        for word in self.whitelist:
            name = name.replace(word, "")
        name = re.sub(r"[^\u4e00-\u9fa5]", "", name)
        if 2 <= len(name) <= 4:
            return name
        return None
       
    def nlp_refine_names(self, text):
        """
        利用 NLP 偵測文中潛藏的人名（例如：林何月娥、陳建瑋）。
        """
        doc = self.nlp(text)
        extra_names = set()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                # 過濾掉太短或在白名單內的詞
                if len(name) >= 2 and name not in self.whitelist:
                    extra_names.add(name)
        return extra_names

    def mask_text(self, text, target_names):
        """
        去識別化遮蔽：
        1. 合併名單並去重。
        2. 過濾掉長度不足或屬於子字串的雜訊。
        3. 使用正則一次性替換，避免重複遮蔽。
        """
        # 取得 NLP 偵測名單
        nlp_names = self.nlp_refine_names(text)
        
        # 合併並清理：只保留長度 >= 2 的名稱
        combined_names = set(n for n in (set(target_names) | nlp_names) if len(n) >= 2)
        
        # 過濾子字串，避免重複遮蔽
        sorted_names = sorted(list(combined_names), key=len, reverse=True)
        final_list = []
        for name in sorted_names:
            if not any(name in existing and name != existing for existing in final_list):
                final_list.append(name)
        
        masked_text = text
        # 由長到短替換
        for name in final_list:
            # 姓名 -> 姓 + 〇 (+ 〇)
            mask = name[0] + "〇" * (len(name) - 1)
            masked_text = masked_text.replace(name, mask)
            
        return masked_text, final_list


class InsureAnalysisEngine:
    def __init__(self, db_path="./chroma_db"):
        # 1. 初始化 Embedding 模型 
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-zh-v1.5"
        )
        
        # 2. 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="insure_cases",
            embedding_function=self.embedding_model
        )

    def extract_features(self, masked_text):
        """
        模擬呼叫 Llama 3 進行結構化提取。
        實際操作時可串接 Ollama API 或 Groq。
        """
        response = ollama.generate(
            model='llama3', 
            prompt=ANALYSIS_PROMPT.format(text=masked_text),
            format='json',
            options={"temperature": 0} # 設定為 0 增加解析穩定性
        )
        
        try:
            data = json.loads(response['response'])
            return data
        except Exception as e:
            print(f"解析 JSON 出錯: {e}")
            return None

    def upsert_to_vector_db(self, case_data, masked_text):
        """
        將特徵與文本存入 ChromaDB
        """
        self.collection.add(
            documents=[masked_text], # 存入去識別化後的文本供未來 RAG 檢索
            metadatas=[{
                "case_id": case_data["case_id"],
                "final_amount": case_data["amount"],
                "reason": case_data["reason_summary"]
            }],
            ids=[case_data["case_id"]]
        )
        print(f"✅ 案件 {case_data['case_id']} 已成功索引至向量資料庫。")
    
    def query_similar_cases(self, current_case_text, current_amount, n_results=3):
        """
        [RAG 檢索與風險預警]
        1. 根據目前案情，檢索相似判例
        2. 計算目前申請金額是否偏離歷史市場行情
        """
        # 查詢向量資料庫
        results = self.collection.query(
            query_texts=[current_case_text],
            n_results=n_results
        )

        # 提取歷史金額進行風險評估
        history_amounts = [m["final_amount"] for m in results["metadatas"][0]]
        avg_amount = sum(history_amounts) / len(history_amounts) if history_amounts else 0
        
        # 風險計算：(目前金額 - 歷史平均) / 歷史平均
        risk_score = (current_amount - avg_amount) / avg_amount if avg_amount > 0 else 0
        is_high_risk = risk_score > 0.3  # 偏離 30% 即標記高風險

        return {
            "similar_cases": results["metadatas"][0],
            "market_average": avg_amount,
            "deviation_ratio": f"{risk_score:.2%}",
            "risk_alert": "🔴 高風險 - 請求金額顯著高於歷史判例" if is_high_risk else "🟢 正常 - 符合市場行情"
        }

def load_text_file(file_path):
    """
    檔案讀取
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # 1. 檔案讀取 
    try:
        file_name = "text1.txt" 
        raw_text = load_text_file(os.path.join(INPUT_DIR, file_name))
        print(f"成功讀取檔案: {file_name}")
    except Exception as e:
        print(f"檔案讀取失敗: {e}")
        return
    
    # 2. 初始化處理器
    processor = InsureDataProcessor()

    # 3. 數據管線：切分 -> 提取名單 -> 去識別化
    sections = processor.split_sections(raw_text)
    header_names = processor.extract_target_names(sections.get("header", ""))
    metadata = processor.get_metadata(sections.get("header", ""))
    case_id = metadata["case_id"]

    content = sections.get("content", "")
    safe_content, detected_all = processor.mask_text(content, header_names)
    print(safe_content)

    analyzer = InsureAnalysisEngine()

    # 4. 分析引擎：LLM 結構化提取
    features = analyzer.extract_features(safe_content)
    
    if features:
        features["case_id"] = case_id 
        print(f"LLM 提取結果: {json.dumps(features, ensure_ascii=False, indent=2)}")
        print(features)
        
        # 5. 存入向量資料庫
        analyzer.upsert_to_vector_db(features, safe_content)

        # 6. 風險預警模擬
        new_claim_amount = 2000000
        new_claim_context = "行人走在斑馬線上遭小貨車撞擊致死，家屬極度痛苦。"
        risk_report = analyzer.query_similar_cases(new_claim_context, new_claim_amount)

        print("\n=== 相似案例風險評估報告 ===")
        print(f"目前請求金額: {new_claim_amount}")
        print(f"歷史相似案平均判賠: {risk_report['market_average']}")
        print(f"偏離比例: {risk_report['deviation_ratio']}")
        print(f"系統預警: {risk_report['risk_alert']}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    main()