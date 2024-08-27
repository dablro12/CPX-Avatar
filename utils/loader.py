from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

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
