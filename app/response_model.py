from langchain.chains import LLMChain, SimpleChain, RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import SimpleRetriever
from langchain_aws import ChatBedrock
from langchain_core.chains import TransformChain, SequentialChain

# AWS Bedrock에서 Llama3 모델 설정
response_model = ChatBedrock(
    credentials_profile_name='default',
    region_name='us-east-1',
    model_id='meta.llama3-70b-instruct-v1:0',
    model_kwargs={
        "temperature": 0.1,
        "top_p": 0.9
    }
)

# 검색 시스템 설정 RAG시스템 설정 
retriever = SimpleRetriever.from_texts(
    texts=["이 문서들은 검색에 사용됩니다.", "여기에 여러 외부 데이터베이스나 자료들을 연결할 수 있습니다."],
    embedding_function=BedrockEmbeddings(credentials_profile_name='default')
)

# CoT 템플릿 구성
cot_template = PromptTemplate(
    input_variables=["question", "retrieved_docs"],
    template="""
    질문: {question}

    검색된 문서들:
    {retrieved_docs}

    다음과 같은 단계로 문제를 해결하세요:
    1. 질문을 이해하고 필요한 정보를 식별하세요.
    2. 검색된 문서에서 필요한 정보를 추출하세요.
    3. 논리적으로 추론하세요.
    4. 최종 결론을 도출하여 답변을 작성하세요.
    """
)

# CoT 체인 생성
cot_chain = LLMChain(
    prompt=cot_template,
    llm=response_model,
    output_parser=StrOutputParser()
)

# RAG 체인 생성
rag_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=cot_chain
)

# 최종 체인 구성
complete_chain = SimpleChain(
    first_chain=rag_chain,
    second_chain=cot_chain
)

# 질문 예시
question = "인공지능의 주요 윤리적 문제는 무엇인가요?"

# 시스템 실행 및 답변 생성
response = complete_chain.run(question)
print(response)
