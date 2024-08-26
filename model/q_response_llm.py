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

class generate_model:
    def __init__(self, query_table_path = '../data/cpx_part.pdf'):
        self.page_chunks = load_pdf_by_page(path = query_table_path) # PDF를 페이지별로 로드
        self.chunk_sizes = [len(chunk['page_content']) for chunk in self.page_chunks]
        self.filtered_chunk_sizes, self.lower_bound, self.upper_bound = remove_outliers(self.chunk_sizes)
        self.filtered_page_chunks = [self.page_chunks[i] for i, size in enumerate(self.chunk_sizes) if self.lower_bound <= size <= self.upper_bound]
        # 필터링된 청크사이즈를 페이지별로 나누어서 text_splitter에 저장 
        self.page_text_splitter = PageAwareTextSplitter()
        self.split_docs = self.page_text_splitter.split(self.filtered_page_chunks)
        self.vector_store = self.get_vector_store(load = True, save = False)
        self.response_retriever = self.vector_store.as_retriever(
            k = 5,
            model_id = 'cosine'
        )
        self.response_prompt = self.get_response_prompt()
        self.response_chain = self.get_response_chain(
            model_name = 'anthropic.claude-3-haiku-20240307-v1:0',
            temperature = 0.0
        )
    
    def run(self,
            question:str,
            patient_query:str,
            question_class:str = "L",
            required_voca:str =  "3일 전과 어제는 명치쪽에서 오늘 오른쪽 윗배로 이동",
            example:str = "배가 아픈게 3일 전과 어제는 명치쪽에서 오늘 오른쪽 아랫배로 이동해서 너무 아파요"):
        
        response_input = {
            "question_class": question_class, #여기에도 클래스 들어가야함
            "patient_query": patient_query,
            "required_voca" :required_voca, #여기에 데이터 들어가야함 
            # "context": " ".join([doc.page_content for doc in self.response_retriever.invoke(question)]), #사전 정보를 넣기
            "example" : example, #예시는 선생님들에게 부탁드리기
            "question": question #질문하기
        }
        inference_response = self.response_chain.invoke(response_input)
        return inference_response
        
    
    def get_response_chain(self, model_name='anthropic.claude-3-5-sonnet-20240620-v1:0', temperature=0.1):
        # RAG 기반 
        response_model = ChatBedrock(
            credentials_profile_name='default',
            region_name='us-east-1',
            model_id=model_name,
            model_kwargs={
                "temperature": temperature,
                "top_p": 0.9
            }
        )
        # RAG 기반 체인 생성
        response_chain = (
            {
                "question_class": RunnablePassthrough(),
                "patient_query" : RunnablePassthrough(),
                "required_voca": RunnablePassthrough(),
                # "context": RunnablePassthrough(),  # 이 부분이 올바르게 작동하는지 확인해야 합니다.
                "example": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | self.response_prompt  # 프롬프트와 연결
            | response_model  # 모델과 연결
            | StrOutputParser()  # 출력 파싱
        )
        
        return response_chain  # 체인을 반환해야 합니다
    
    def get_vector_store(self, load = True, save = False):
        embedding_model = BedrockEmbeddings(
            credentials_profile_name='default',
            region_name='us-east-1',
            model_id = 'cohere.embed-multilingual-v3'
        )
        if save:
            vector_store = FAISS.from_documents(
                documents = split_docs,
                embedding = embedding_model,
            )

            # 다음에는 이 벡터 저장소를 캐싱해서 가져오기 위해 벡터 저장소를 저장해놓기 
            vector_store.save_local('../db/q_response_faiss')
        if load:# Load 
            vector_store = FAISS.load_local(
                '../db/q_response_faiss',
                embedding_model,
                allow_dangerous_deserialization=True  # 위험한 역직렬화 허용
            )
        return vector_store
    
    
    def get_response_prompt(self):
        # 프롬프트 근거 
        # 1. 환자 역할: 질문에 대해 환자처럼 대답하도록 강조했습니다.
        # 2. 한국어: 답변은 한국어로 작성되며, 특정 단어를 반드시 포함해야 한다고 명시했습니다.
        # 3. 예시 기반: 제공된 예시(example)와 유사한 스타일로 답변을 작성하도록 했습니다.
        # 4. Chain of Thought (COT): 왜 그러한 답변을 했는지에 대한 분석을 요구했습니다.
        # 5. 질문의 의도: CPX_first_stage_class와 question_class에 맞춰 정확하게 답변을 작성해야 한다고 설명했습니다.
        # 6. 한 문장으로 제한: 답변이 반드시 한 문장으로 작성되어야 한다고 명시했습니다.
        response_prompt = PromptTemplate.from_template(
        """You are an assistant that simulates patient responses based on specific questions.
        Your task is to answer the question in Korean as if you were the patient, ensuring to include specific keywords given to you.
        Additionally, you should generate the response in a style similar to the example provided.

        The question's intent will be categorized using the CPX_first_stage_class, and the key tag (question_class) will be provided.
        You must ensure that your final response is only one sentence long and includes the required vocabulary provided.

        # CPX_first_stage_class:
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
        
        # patient_query:
        {patient_query}
        
        # CPX first_stage_class-key tag:
        {question_class}
        
        # Required Vocabulary:
        {required_voca}

        # Question:
        {question}
        
        # Example:
        {example}

        # Response (질문에 대해 환자가 특정단어를 이용해 반응하는 문장을 한 문장으로 작성):
        """
    )
        return response_prompt

            # #Chain of Thought:
            # 1. 첫 번째로, 질문의 주요 주제를 분석합니다. 질문에서 어떤 핵심 개념이 드러나고 있나요?
            # 2. 두 번째로, 문맥에서 관련된 정보들을 검토합니다. 문맥이 어떤 정보를 제공하고 있나요?
            # 3. 세 번째로, 주제와 문맥을 기반으로 주어진 키워드를 포함하여 가장 적합한 답변을 한 문장으로 만듭니다. 필수로 제공된 핵심 단어({required_voca})를 포함하세요.
            # 4. 마지막으로, 답변은 한 문장으로 제한되어야 하며, 예시와 유사한 스타일로 작성되어야 합니다.
            
    