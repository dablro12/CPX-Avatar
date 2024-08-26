from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import numpy as np 

def load_pdf_by_page(path: str):
    # PyMuPDFLoader로 PDF 파일 로드
    loader = PyMuPDFLoader(file_path=path)
    documents = loader.load()
    
    # 각 페이지를 개별 문서로 처리
    page_docs = []
    for doc in documents:
        # 각 페이지의 텍스트를 하나의 chunk로 처리
        page_doc = {
            "page_content": doc.page_content,  # 문서의 페이지 내용
            "metadata": {
                "page_number": doc.metadata.get('page_number', len(page_docs) + 1)  # 페이지 번호 추가
            }
        }
        page_docs.append(page_doc)
    
    return page_docs


class classify_model:
    def __init__(self, pdf_path:str):
        self.page_chunks = load_pdf_by_page(path = pdf_path)
        self.chunk_sizes = [len(chunk['page_content']) for chunk in self.page_chunks]
        self.optimal_chunk_size = self.optimal_chunk_size(chunk_size = 2000, chunk_overlap =500)
        self.embedding_model = BedrockEmbeddings(
            credentials_profile_name='default',
            region_name='us-east-1',
            model_id = 'cohere.embed-multilingual-v3'
        )
        self.vector_store = self.vector_store_loader(load = True, save = False)
        self.classify_retriever = self.vector_store.as_retriever(
            k = 5,
            model_id = 'cosine'
        )
        self.classify_prompt = self.get_prompt()
        self.classify_model = BedrockChat(
            credentials_profile_name='default',
            region_name='us-east-1',
            model_id='anthropic.claude-3-haiku-20240307-v1:0',
            model_kwargs={
            "temperature": 0.1,
            "top_p": 0.9}
        )
        
        self.classify_chain = (
            {
                "context" : RunnablePassthrough(),
                "question" : RunnablePassthrough(),
            }
            | self.classify_prompt 
            | self.classify_model
            | StrOutputParser()
        )
    
    def run(self, question):
        classify_input = {
            "context": " ".join([doc.page_content for doc in self.classify_retriever.invoke(question)]),
            "question": question
        }
        
        inference_class = self.classify_chain.invoke(classify_input)
        return inference_class
        
    
        
    def get_prompt(self):
        classification_prompt = PromptTemplate.from_template(
            """You are an assistant specialized in classifying questions into the correct category from CPX_first_stage_class.
            Based on the following question and the retrieved context, select the most appropriate class from CPX_first_stage_class.
            Make sure your final answer is one of the classes listed in CPX_first_stage_class.
            Answer in Korean.

            #CPX_first_stage_class:
            [
                "O" : "Onset 증상의 발생 시점",
                "L" : "Location 증상의 발생 위치",
                "D" : "Duration 증상의 지속 시간",
                "Co" : "Course 증상의 경과 과정",
                "Ex" : "Experience 이전 증상 발생 여부",
                "C" : "Character 증상의 특성（색, 양, 세기, 냄새, 강도、뻗침 등）",
                "A" : "Associated Sx. 계통적 문진",
                "F" : "Factor 악화/완화인자",
                "E" : "Event 이전 건강 검진 / 입원 여부",
                "외" : "외상력 외상 발생 여부",
                "과" : "과거력 고혈압, 당뇨, 고지혈증, 만성 간질환, 결핵, 기저 질환",
                "약" : "약물 투약력 현재 투여 중인 약물",
                "사" : "사회력 술, 담배, 커피, 운동, 식습관, 직업 등",
                "가" : "가족력 가족 중 유사 증상 여부",
                "여" : "여성력 생리력 문진"
            ]
            
            #Context:
            {context}
            
            #Question:
            {question}
            
            #Chain of Thought:
            1. 첫 번째로, 질문의 주요 주제를 분석합니다. 질문에서 어떤 핵심 개념이 드러나고 있나요?
            2. 두 번째로, 문맥에서 관련된 정보들을 검토합니다. 문맥이 어떤 정보를 제공하고 있나요?
            3. 세 번째로, 주제와 문맥을 기반으로 가장 적합한 클래스를 결정합니다.

            #Selected Class: Provide only the class name (e.g., "O", "L", "D", etc.)"""
        )

        return classification_prompt
        
    def optimal_chunk_size(self, chunk_size = 2000, chunk_overlap =500):
        optimal_chunk_size = np.median(self.chunk_sizes)
        
        if optimal_chunk_size > chunk_size:
            optimal_chunk_size = chunk_size
        
        return optimal_chunk_size
    
    def splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = int(self.optimal_chunk_size),
            chunk_overlap= 500
        )
        all_docs = [chunk['page_content'] for chunk in self.page_chunks]
        all_docs = [Document(page_content=content, metadata={"page_number": i + 1}) for i, content in enumerate(all_docs)]
        
        split_docs = text_splitter.split(all_docs)
        return split_docs
    
    def vector_store_loader(self, load = False, save = True):
        if save:
            vector_store = FAISS(
                documents = self.splitter(),
                embedding = self.embedding_model
            )
            vector_store.save_local(
                '../db/q_classifier_faiss'
            )
        
        if load:
            vector_store = FAISS.load_local(
                '../db/q_classifier_faiss',
                self.embedding_model,
                allow_dangerous_deserialization=True  # 위험한 역직렬화 허용
            )
        
        return vector_store
    
        