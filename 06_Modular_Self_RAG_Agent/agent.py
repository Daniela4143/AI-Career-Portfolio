from datasets import load_dataset
import chromadb
import ollama
from google import genai
from google.genai import types
from abc import ABC, abstractmethod
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# self RAG: json format
DATASET_NAME = "allenai/sciq"
dataset = load_dataset(DATASET_NAME)

class LLMEngine(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OllamaEngine(LLMEngine):
    def __init__(self, model_id='llama3'):
        self.model_id = model_id

    def generate(self, prompt):
        response = ollama.generate(
            model=self.model_id,
            prompt=prompt,
            format='json',          # set format
            options={
                "temperature": 0    # set deterministic response
            }        
        )
        return response["response"]

class GeminiEngine(LLMEngine):
    def __init__(self, model_id = 'gemini-2.5-flash'):
        self.gen_client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = model_id
        self.config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0
        )

    def generate(self, prompt):
        response = self.gen_client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=self.config
        )

        return response.text
    
class RAGAgent:
    def __init__(self, collection_name="sciq", llm_engine: LLMEngine = None):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(collection_name)
        self.llm = llm_engine

    def build_knowledge_base(self, data, number):
        self.collection.add(
            documents=[data[i]["support"] for i in range(number)],
            metadatas=[{"question": data[i]["question"],
                        "answer": data[i]["correct_answer"]}
                        for i in range(number)],
            ids=[str(i) for i in range(number)]
        )

    def retrieve(self, query_text):
        # Retrieve for RAG
        results = self.collection.query(
            query_texts=[query_text],
            n_results=1
        )
        return results

    def build_prompt(self, question, context):
        # Augmentation for RAG with few-shot example
        example_support = dataset["train"][0]["support"]
        example_question = dataset["train"][0]["question"]
        example_answer = dataset["train"][0]["correct_answer"]
        prompt = f"""
        妳是一位專業的科學助理。請根據[參考文本]回答[問題]。妳必須使用「繁體中文」回答。

        ### 範例格式開始 ###
        [參考文本]: {example_support}
        [問題]: {example_question}
        輸出:
        {{
        "analysis": "文本提到中溫生物常用於起司與優格製作。",
        "answer": "{example_answer}"
        }}
        ### 範例格式結束 ###

        [參考文本]: {context}
        [問題]: {question}

        請嚴格遵守 JSON 格式輸出，且回答必須為繁體中文：
        """
        return prompt
    
    def generate(self, question, context):
        prompt = self.build_prompt(question, context)

        # Generation for RAG
        return self.llm.generate(prompt)
    
    def validate_faithfulness(self, context, tested_answer):
        v_prompt = f"""
            妳是一位嚴謹的「事實查核員」。
            妳唯一的任務是：檢查「回答」中的資訊，是否能在「參考文本」中找到支持。
            
            [參考文本]: {context}
            [回答]: {tested_answer}
            
            判斷標準：
            1. 只要「回答」的內容在「參考文本」中有提到，就是 true。
            2. 不要管「回答」是否完整，只要它沒「亂編」文本以外的事實就是 true。
            3. 必須使用「繁體中文」說明原因。

            只需回答 JSON：
            {{
            "is_faithful": true/false,
            "reason": "說明支持或不支持的證據"
            }}
        """
        return self.llm.generate(v_prompt)

    def validate_relevant(self, distances, context, tested_answer):
        # Is the Context Relevant?
        if distances > 1.5:
            answer = "找不到相關知識"
        else:
            # Is the Answer Faithful?
            validation_str = self.validate_faithfulness(context, tested_answer)
            validation_json = json.loads(validation_str)    # transform to python json
            print(validation_json)

            answer = tested_answer
            if validation_json["is_faithful"]:
                answer = tested_answer
            else:
                answer = "回答未通過事實查核"
        return answer


# engine = OllamaEngine(model_id='llama3')
engine = GeminiEngine(model_id='gemini-2.5-flash')
rag = RAGAgent(collection_name="sciq", llm_engine=engine)

rag.build_knowledge_base(dataset["train"], 5)

# query_question = "What eject in Kilauea in hawaii?"
# query_question = "你是誰?"
query_question = "Does flower eject in Kilauea in hawaii?"
context = rag.retrieve(query_question)  # context: most relevant support

distances = context["distances"][0][0]
print(distances)
tested_answer = rag.generate(query_question, context["documents"][0][0])

answer = rag.validate_relevant(distances, context["documents"][0][0], tested_answer)

print(answer)

