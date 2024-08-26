import pandas as pd 
import sys, os 
import numpy 
import numpy as np
from langchain.document_loaders import PyMuPDFLoader
# 페이지별로 청크를 나누어서 저장
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate
# from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

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


class PageAwareTextSplitter:
    def __init__(self, chunk_size=2000, chunk_overlap=500):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split(self, page_docs):
        split_docs = []
        for page in page_docs:
            # 페이지 텍스트가 너무 길면 청크로 나눔
            if len(page['page_content']) > self.chunk_size:
                # RecursiveCharacterTextSplitter 사용
                sub_docs = self.splitter.split_text(page['page_content'])
                for i, sub_doc in enumerate(sub_docs):
                    split_docs.append(Document(
                        page_content=sub_doc,
                        metadata={
                            "page_number": page['metadata']['page_number'],
                            "chunk_number": i + 1
                        }
                    ))
            else:
                # 페이지를 그대로 유지
                split_docs.append(Document(
                    page_content=page['page_content'],
                    metadata=page['metadata']
                ))
        return split_docs


def remove_outliers(chunk_sizes):
    # 1사분위수(Q1)와 3사분위수(Q3) 계산
    Q1 = np.percentile(chunk_sizes, 25)
    Q3 = np.percentile(chunk_sizes, 75)
    
    # IQR 계산
    IQR = Q3 - Q1
    
    # 이상치 임계값 설정
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 이상치가 아닌 값들만 필터링
    filtered_chunk_sizes = [size for size in chunk_sizes if lower_bound <= size <= upper_bound]
    
    return filtered_chunk_sizes, lower_bound, upper_bound

class feedback_model:
    def __init__(self, pdf_path = '../data/cpx_part.pdf'):
        self.page_chunks = load_pdf_by_page(path = pdf_path) # PDF를 페이지별로 로드
        self.chunk_sizes = [len(chunk['page_content']) for chunk in self.page_chunks]
        self.filtered_chunk_sizes, self.lower_bound, self.upper_bound = remove_outliers(self.chunk_sizes)
        self.filtered_page_chunks = [self.page_chunks[i] for i, size in enumerate(self.chunk_sizes) if self.lower_bound <= size <= self.upper_bound]
        # 필터링된 청크사이즈를 페이지별로 나누어서 text_splitter에 저장 
        self.page_text_splitter = PageAwareTextSplitter()
        self.split_docs = self.page_text_splitter.split(self.filtered_page_chunks)
        self.vector_store = self.get_vector_store(load = False, save = True)
        self.feedback_retriever = self.vector_store.as_retriever(
            k = 5,
            model_id = 'cosine'
        )
        self.feedback_prompt = self.get_feedback_prompt()
        self.feedback_chain = self.get_feedback_chain(
            model_name = 'anthropic.claude-3-5-sonnet-20240620-v1:0',
            temperature = 0.0
        )
    
    def run(self, data:dict):
        feedback_input = {
            "context": " ".join([doc.page_content for doc in self.feedback_retriever.invoke(data)]),
            "data": data
        }
        
        feedback_output = self.feedback_chain.invoke(feedback_input)
        return feedback_output
        
    
    def get_feedback_chain(self, model_name='anthropic.claude-3-5-sonnet-20240620-v1:0', temperature=0.1):
        # RAG 기반 
        feedback_response__model = ChatBedrock(
            credentials_profile_name='default',
            region_name='us-east-1',
            model_id=model_name,
            model_kwargs={
                "temperature": temperature,
                "top_p": 0.9
            }
        )
        # RAG 기반 체인 생성
        feedback_chain = (
            {
                "data" : RunnablePassthrough(),
                "context" : RunnablePassthrough(),
            }
            | self.feedback_prompt 
            | feedback_response__model
            | StrOutputParser()
        )
        
        return feedback_chain  # 체인을 반환해야 합니다
    
    def get_vector_store(self, load = True, save = True):
        embedding_model = BedrockEmbeddings(
            credentials_profile_name='default',
            region_name='us-east-1',
            model_id = 'amazon.titan-embed-text-v2:0'
        )
        if save:
            vector_store = FAISS.from_documents(
                documents = self.split_docs,
                embedding = embedding_model,
            )

            # 다음에는 이 벡터 저장소를 캐싱해서 가져오기 위해 벡터 저장소를 저장해놓기 
            vector_store.save_local('../db/q_feedback_faiss')
        if load:# Load 
            vector_store = FAISS.load_local(
                '../db/q_feedback_faiss',
                embedding_model,
                allow_dangerous_deserialization=True  # 위험한 역직렬화 허용
            )
        return vector_store
    
    
    def get_feedback_prompt(self):
        # 프롬프트 근거 
        # 1. 환자 역할: 질문에 대해 환자처럼 대답하도록 강조했습니다.
        # 2. 한국어: 답변은 한국어로 작성되며, 특정 단어를 반드시 포함해야 한다고 명시했습니다.
        # 3. 예시 기반: 제공된 예시(example)와 유사한 스타일로 답변을 작성하도록 했습니다.
        # 4. Chain of Thought (COT): 왜 그러한 답변을 했는지에 대한 분석을 요구했습니다.
        # 5. 질문의 의도: CPX_first_stage_class와 question_class에 맞춰 정확하게 답변을 작성해야 한다고 설명했습니다.
        # 6. 한 문장으로 제한: 답변이 반드시 한 문장으로 작성되어야 한다고 명시했습니다.
        feedback_prompt = PromptTemplate.from_template(
            """You are an assistant specialized in providing feedback based on performance scores and detailed analysis.
            Based on the following data, generate constructive feedback for the performance, focusing on strengths and areas for improvement.
            Ensure your feedback is comprehensive and written in Korean.
            
            #Data:
            {data}

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
            
            # Feedback Process:
            1. 총점을 분석하고, 전체적인 성과에 대한 피드백을 제공합니다.
            2. 병력 청취, 진단, 교육, 약물 처방 점수의 상세 점수를 바탕으로 각 부분의 강점과 개선이 필요한 부분을 설명합니다.
            3. 수행 여부에 대한 피드백을 바탕으로, 누락된 항목이나 부족한 부분에 대한 구체적인 피드백을 제시합니다.
            4. 마지막으로, 향후 개선 방안을 제안합니다.

            # Feedback: Write a comprehensive and constructive feedback in Korean based on the provided data.
            """
    )
        return feedback_prompt

            # #Chain of Thought:
            # 1. 첫 번째로, 질문의 주요 주제를 분석합니다. 질문에서 어떤 핵심 개념이 드러나고 있나요?
            # 2. 두 번째로, 문맥에서 관련된 정보들을 검토합니다. 문맥이 어떤 정보를 제공하고 있나요?
            # 3. 세 번째로, 주제와 문맥을 기반으로 주어진 키워드를 포함하여 가장 적합한 답변을 한 문장으로 만듭니다. 필수로 제공된 핵심 단어({required_voca})를 포함하세요.
            # 4. 마지막으로, 답변은 한 문장으로 제한되어야 하며, 예시와 유사한 스타일로 작성되어야 합니다.
            
    