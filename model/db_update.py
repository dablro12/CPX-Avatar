gwfweimport sys, os
sys.path.append('../')
import pandas as pd 
import numpy as np 
from langchain.document_loaders import PyMuPDFLoader
from utils.loader import load_pdf_by_page, PageAwareTextSplitter
from utils.transformer import remove_outliers
from langchain.embeddings import BedrockEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS

# 캐시 경로를 절대 경로로 설정
cache_dir = os.path.abspath('../cache/')

def get_docs(pdf_path):
    # PDF를 페이지별로 로드
    page_chunks = load_pdf_by_page(path=pdf_path)
    chunk_sizes = [len(chunk['page_content']) for chunk in page_chunks]
    # 이상치 제거
    filtered_chunk_sizes, lower_bound, upper_bound = remove_outliers(chunk_sizes)
    filtered_page_chunks = [page_chunks[i] for i, size in enumerate(chunk_sizes) if lower_bound <= size <= upper_bound]

    page_text_splitter = PageAwareTextSplitter()
    split_docs = page_text_splitter.split(filtered_page_chunks)

    return split_docs

def get_embedding_model(docs:list, model_name='amazon.titan-embed-text-v2:0', load_vector_store_path=None, namespace='default_namespace', save=False):
    # 임베딩 모델 초기화
    embedding_model = BedrockEmbeddings(
        credentials_profile_name='default',
        region_name='us-east-1',
        model_id=model_name
    )

    # 캐시 파일 저장소 초기화
    store = LocalFileStore(root_path=cache_dir)
    # 캐시백된 임베딩 모델 생성
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding_model, store, namespace=namespace
    )

    if save:
        # 새 문서로부터 벡터 저장소 생성 및 저장
        vector_store = FAISS.from_documents(
            documents=docs,
            embedding=cached_embedder
        )
        vector_store.save_local(load_vector_store_path)
    else:
        # 기존 벡터 저장소 로드
        vector_store = FAISS.load_local(
            load_vector_store_path,
            cached_embedder,
            allow_dangerous_deserialization=True  # 위험한 역직렬화 허용
        )
        # 새로운 문서를 기존 벡터 저장소에 추가
        vector_store.add_documents(docs) #FAISS는 docs을 추가하면 자동으로 벡터도 업데이트
        # 업데이트된 벡터 저장소를 저장
        vector_store.save_local(load_vector_store_path)

    return vector_store

def main(pdf_path: str, load_vector_store_path: str, namespace: str, first_save=False):
    split_docs = get_docs(pdf_path)
    vector_store = get_embedding_model(
        docs=split_docs,
        model_name='amazon.titan-embed-text-v2:0',
        load_vector_store_path=load_vector_store_path,
        namespace=namespace,
        save=first_save  # 필요시 True로 설정 #load 
    )

    return vector_store

if __name__ == '__main__':
    vector_store_path = '../db/cpx_faiss'
    namespace = 'cpx_all_embeddings'
    paths_li = [
        '../data/cpx_part.pdf',
    ]
    for pdf_path in paths_li:
        main(
            pdf_path=pdf_path,
            load_vector_store_path=vector_store_path,
            namespace=namespace,
            first_save=True
        )
