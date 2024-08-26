import sys, os
sys.path.append('../')

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse,StreamingResponse
import pandas as pd
import numpy as np  
from model import q_class_llm, q_response_llm, q_feedback_llm
app = FastAPI()

################################################################
# 초기 변수 불러놓기 
stage_dfs = pd.read_csv('../data/cpx_stage.csv')
question_dfs = pd.read_csv('../data/cpx_question_label.csv')
supplements_medication = pd.read_csv('../data/diseaseDB.csv')

# 첫번째 열을 칼럼 이름으로 지정
stage_dfs.columns = stage_dfs.iloc[0]
stage_dfs = stage_dfs[1:]
question_dfs.columns = question_dfs.iloc[0]
question_dfs = question_dfs[1:]

stage_dfs = stage_dfs[stage_dfs['카테고리'] == '급성복통']
# 카테고리 
category = stage_dfs['카테고리']
# 성별	나이	혈압	맥박	호흡	체온	주소를 떼오기 
pre_info = stage_dfs.iloc[:, 2:10] # 의대생에게 줄 정보 
# O L D Co Ex CAFE 
first_stages = stage_dfs.iloc[:, 10:26] # 각 칼럼 순서에 따라 답변에 대한 핵심단어가 있음 
# classification CPX Class 리스트 가져오기
CPX_first_stage_class = first_stages.columns
# 신체 진찰 
second_stages = stage_dfs.iloc[:, 26:32]  # 신체 진찰에 정답지가 존재 
# 진단 및 진단 교육 안내
third_stages_dignosis = stage_dfs.iloc[:, 32:35] # 진단에 대한 정답지가 존재 
third_stages_education = stage_dfs.iloc[:, 35:38] # 검사 및 치료 권유 정답지가 존재 
third_stages_medication = stage_dfs.iloc[:, 38:] # 약물 처방에 대한 정답지가 존재
################################################################
# LLM Initialize
question_classify_llm = q_class_llm.classify_model(pdf_path = '../data/cpx_total.pdf')
question_response_llm = q_response_llm.generate_model(query_table_path = '../data/cpx_part.pdf')
###############################################################
# 전역 변수로 random_index를 선언 (초기 값 None)
random_index = None
question_set_li = first_stages.columns
cnt_cpx_question_set = set(question_set_li)  # get question response에서 같은 클래스안에 있는 단어를 말하지 않기 위한 변수

first_stage_boolean = {
    "O" : False,
    "L" : False,
    "D" : False,
    "Co" : False,
    "Ex" : False,
    "C" : False,
    "A" : False,
    "F" : False,
    "E" : False,
    "외" : False,
    "과" : False,
    "약" : False,
    "사" : False,
    "가" : False,
    "여" : False,
}

dignosis_score, education_score, medication_score = 0, 0, 0
dignosis_feedback, education_feedback, medication_feedback = "", "", ""

@app.get("/get_init_info")
async def get_init_info(trash:str):
    """
    pre_info 데이터프레임에서 랜덤 인덱스에 해당하는 데이터를 반환하는 API 엔드포인트.
    """
    global random_index
    global question_set_li
    global cnt_cpx_question_set_li
    global first_stage_boolean
    global dignosis_score
    global education_score
    global medication_score

    trash = trash 
    # random_index는 함수 내에서 호출될 때마다 새로 생성되도록 합니다.
    random_index = np.random.randint(0, len(pre_info))
    
    # pre_info에서 random_index에 해당되는 행 데이터를 가져옴
    random_data = pre_info.iloc[random_index]
    
    # 우진이 second 필요한 데이터 주기
    set_dict, required_data = get_second_required(second_stages = second_stages, index = random_index)
    
    # 처음에는 random_index 값을 dict에 포함해 반환 
    return_dict = {
        # 'pid' : random_index,
        'second_all_set' : set_dict,
        'required_data' : required_data
    }
    
    # # return dict에 random_data.do_dict()를 추가
    return_dict.update(random_data.to_dict())
    
    # 결과를 JSON 형식으로 반환
    return JSONResponse(content=return_dict)


@app.get("/get_index")
async def get_index(trash:str):
    global random_index
    """
    random_index를 반환하는 함수
    """
    trash = trash 
    return dict(pid = random_index)

# 첫번쨰로 질문을 인풋으로 받아서 답변을 보내는 API 엔드포인트
@app.get("/get_question_response")
async def get_question_class(question: str):
    """
    첫 번째 단계 진단에 대한 정답지를 반환하는 API 엔드포인트
    """
    global random_index
    global question_set_li
    global cnt_cpx_question_set_li

    if random_index is None:
        query = first_stages.iloc[0, :]
    else:
        query = first_stages.iloc[random_index, :]

    # query의 컬럼이 키고 question이 값인 딕셔너리 생성
    query_dict = query.to_dict()
    for key in query_dict.keys():
        if pd.isnull(query_dict[key]):
            query_dict[key] = 'NULL'
            
    ################### 질문 분류 ########################
    # # 사용자가 보낸 질문에 대해 클래스를 예측
    infer_class = question_classify_llm.run(question)
    
    # 만약 llm response가 CPX_first_stage_class에 포함되어 있지 않다면 다시 response
    while infer_class not in CPX_first_stage_class:
        infer_class = question_classify_llm.run(question)
        
    first_stage_boolean[infer_class] = True # 해당 클래스를 수행했다는 의미로 True로 변경
    
    ######################3 Response 준비 및 받기 ########################
    # set변수인 cnt_cpx_question_set에서 infer_class인 부분 제거 및 query에서도 infer_class인 칼럼에 해당되는 값을 제거
    # cnt_cpx_question_set = cnt_cpx_question_set - set([infer_class])
    # NULL인 부분 제거
    query_dict = {key: value for key, value in query_dict.items() if value != 'NULL'}
    
    # response 받기 
    response = question_response_llm.run(
        question = question,
        patient_query= query_dict, 
        question_class = infer_class,
        required_voca = query_dict[infer_class],
        example = """
        "어제부터 배가 아파서 아주 혼났어요~"
        "기침이 너무 심해서 일상생활이 안돼요."
        "가슴이 돌로 누르는 듯이 아핐어요."
        "어제 밤에는 속이 안 좋았는데 오늘 아침이 되니까 오른쪽 아랫배가 아프네요."
        "3일 전부터 미열이 났고 계속 기침이 나네요. 콜록콜록"
        "답답하고 숨이 안쉬어져요."
        "밤마다 좀 땀이 나고 으슬으슬해요. 콜록"
        "2주 전에 감기가 걸린 이후 콜록 기침이 나네요. 보세요 말하는 중간에도 콜록콜록 기침이 나요."
        "담배를 지금 하루에 두 갑씩 한 10년 폈거든... 내가 폐암일까?"
        "우리 아버지도 폐암으로 돌아가셔서 걱정이 많이 돼요."
        "저는 밤에 야식을 먹는 걸 좋아해요. 게다가 먹고 바로 누워서 잔답니다~"
        "고혈압약 먹는데 그게 기침하고 관련이 있나요?"
        "내가 40년 전만 하더라도 아주 운동을 잘했는데 작년부터는 계단을 오르면 숨이 차더니 이제는 친구들하고 걸어다니기만해도 힘들어..."
        "저번주에 등산을 갔는데 아이고 심장아파서 죽는건가 했어요. 근데 좀 쉬니까 괜찮더라고요."
        "평상시에도 움직이면 좀 심장이 아프긴했는데 1주일전부터는 쉴때도 좀 아픈 것 같아요."
        "숨도 차고 가슴도 어디 맞은 듯이 아파서 구급차 타고 왔어요."
        "어제 술을 먹고 잤는데 아니 자다가 가슴이 아파서 벌떡!일어났다니까!!!"
        "점점 통증이 등으로 퍼져가요."
        "요즘에 너무 피곤하더니만 가슴에 뭔가 났네요. 빨갛게? "
        "제가 세상에서 제일 좋아하는 음식이 불닭볶음면입니다."
        "저는 밥을 안 먹으면 그렇게 속이 쓰리더라고요. 왜 그러나요?"
        "저는 매일 술을 2병씩 마셔요. 악 악 악 명치가 너무 아파요.전에도 이러더니만..."
        "제가 2일 전에 베트남에서 한 달 여행을 갔다가 왔는데 그게 제 복통과 관련이 있을까욧 ㅎㅎ?"
        "제가 어렸을 때부터 변에서 가끔 피가 나오면서 복통이 있었어요."
        "소변을 누는데 아프네요... 얼른 치료해주세요 선생님."
        "제가 임신을 했을수도 있긴하죠... 마지막 생리가 벌써 6주전이네"
        "산부인과적으로 수술 받은건 없어요."
        "제가 아주 약국이죠. 고혈압약, 당뇨약, 이상지질형증약 먹어요."
        "저는 약을 안먹는 튼튼한 사람입니다. 흐흐"
        "제 어머니는 당뇨가 있고, 제 아버지는 대장암이 있습니다."
        "저는 지금까지 큰 병 걸린 적이 없습니다."
        "제가 아주 종합병원이에요. 부정맥도 있고 당뇨도 있고 고혈압도 있어요."
        "앞으로 숙이면 통증이 좀 덜한데 누우면 아주 힘들어서 바로 못누워요."
        "봄만 되면 기침이 나요."
        """
        )
    
    query_dict[infer_class] = 'NULL'
    
    # data를 딕셔너리로 형태로 반환 
    return response
    # return return_dict


def check_word_in_df_scoring(df, dignosis):
        score = 0
        for word in df:
            if word in dignosis:
                score += 1
        return score

@app.get("/get_third_dignosis")
async def get_third_dignosis(dignosis: str):
    """
    1번만 진행하기  
    진단에 대해 특정 단어가 있는지 확인하고,우 없는 경우를 scoring
    """
    global random_index
    global dignosis_score
    global dignosis_feedback
    if random_index is None:
        random_index = 0
    
    # 진단에 대한 정답지 가져옴
    dignosis_df = third_stages_dignosis.iloc[random_index, :]
    
    dignosis_score = check_word_in_df_scoring(df = dignosis_df, dignosis = dignosis)
    
    dignosis_answer = dignosis_df[dignosis_df.notnull()]
    # feedback 
    if dignosis_score == 3:
        dignosis_feedback = "진단에 대한 답변이 정확합니다."
    else:
        dignosis_feedback = """
            진단에 대한 답변이 부정확합니다.
            사용자분은 환자에게 다음과 같은 진단 계획을 구상하였습니다. {dignosis}
            정답은 {dignosis_answer}압니다.
        """
    
    return_dict = JSONResponse(content={"dignosis_score": dignosis_score})
    return return_dict


@app.get("/get_third_education") #교육 - 검사 
async def get_third_education(education: str):
    """
    1번만 진행하기  
    환자에게 치료나 검사 목적의 교육에 대해 특정 단어가 있는지 확인하고, 있는 경우 없는 경우를 scoring
    """
    global random_index
    global education_score
    global education_feedback
    if random_index is None:
        random_index = 0
    
    # 진단에 대한 정답지 가져옴
    education_df = third_stages_education.iloc[random_index, :]
    
    education_score = check_word_in_df_scoring(df = education_df, dignosis=education)
    education_answer = education_df[education_df.notnull()]
    
    if education_score == 3:
        education_feedback = "교육에 대한 답변이 정확합니다."
    else:
        education_feedback = """
            교육에 대한 답변이 부정확합니다.
            사용자분은 환자에게 다음과 같은 교육 계획을 구상하였습니다. : {education}
            정답은 {education_answer}압니다.
        """
    
    return_dict = {"education_score": education_score}
    return return_dict


@app.get("/get_third_medication")
async def get_third_medication(medication: str):
    """
    1번만 진행하기  
    환자에게 치료나 검사 목적의 치료에 대해 특정 단어가 있는지 확인하고, 있는 경우 없는 경우를 scoring
    """
    global random_index
    global medication_score
    global medication_feedback

    if random_index is None:
        random_index = 0
    
    # 진단에 대한 정답지 가져옴
    medication_df = third_stages_medication.iloc[random_index, :]

    # 만약 medication_df에 NULL이 3개라면 다 점수를 0점처리
    if medication_df.isnull().sum() == 3:
        return JSONResponse(content={"medication_score": 0})
    
    medication_score = check_word_in_df_scoring(df = medication_df, dignosis=medication)
    medication_answer = medication_df[medication_df.notnull()]
    
    # 데이터 프레임을 가져와서 특정 핵심 단어를 가지고 옴 
    if medication_score == 3:
        medication_feedback = "교육에 대한 답변이 정확합니다."
    else:
        if medication_answer.empty:  # 수정된 부분
            medication_feedback = f"""
            사용자분은 환자에게 다음과 같은 치료 계획을 구상하였습니다. : {medication}
            """
        else:    
            medication_feedback = f"""
                교육에 대한 답변이 부정확합니다.
                사용자분은 환자에게 다음과 같은 치료 계획을 구상하였습니다. : {medication}
                정답은 {medication_answer}압니다.
            """

    result_dict = {"medication_score": medication_score}
    
    return JSONResponse(content=result_dict)

@app.get("/get_all_feedback")
async def get_all_feedback(feedback: str = 'want'):
    first_score = sum([1 for value in first_stage_boolean.values() if value])
    if feedback == 'want':
        question_feedback_llm = q_feedback_llm.feedback_model(pdf_path='../data/cpx_part.pdf')
        human_return_dict = {
            "total_score": f"맞춘 점수/총 점수 : {first_score + dignosis_score + education_score + medication_score}/24",
            "detail_score": f"병력 청취 점수 : {first_score}, 진단 점수 : {dignosis_score}, 교육 점수 : {education_score}, 약물 처방 점수 : {medication_score}",
            "first_feedback": first_stage_boolean,
            "third_feedback": {
                "dignosis_score": dignosis_feedback,
                "education_score": education_feedback,
                "medication_score": medication_feedback
            }
        }
        
        # question이 문자열이 아니면 문자열로 변환
        if isinstance(human_return_dict, dict):
            human_return_dict = " ".join([f"{k}: {v}" for k, v in human_return_dict.items()])
        
        return_dict = {
            "human_feedback": human_return_dict,
            "llm_feedback": question_feedback_llm.run(human_return_dict)
        }
        return JSONResponse(content=return_dict)
    else:
        return_score = {
            "total_score": f"맞춘 점수/총 점수 : {first_score + dignosis_score + education_score + medication_score}/24"
        }
        return JSONResponse(content=return_score)

def query_collection(columns):
    set_dict = {}
    for column in columns:
        set_li = []
        # 각 쿼리에서 : (-), : (+) 제거 후 set_li에 추가
        for querys in second_stages[column]:
            if pd.notna(querys):
                pre_li = querys.split('\n')
                for pre in pre_li:
                    clean_pre = pre.split(':')[0].strip()  # 단어만 추출하고 공백 제거
                    set_li.append(clean_pre)

        set_li = list(set(set_li))  # 중복 제거
        set_dict[column] = set_li
    
    return set_dict

# +인 애들만 넘겨주기 
def get_second_data(second_stages, index):
    second_stages_query = second_stages.iloc[index, :]
    plus_dict = {}

    for key, value in second_stages_query.items():
        plus_cnt = []
        detail_dict = {}
        try: 
            for value in value.split('\n'):
                if '+' in value:
                    if value.split(':')[1:] == None:
                        pass
                    else:
                        if ', ' in value.split(':')[1:]:
                            detail_dict[value.split(':')[0].strip()] = value.split(':')[1:].split(', ')[-1]
                        else:
                            detail_dict[value.split(':')[0].strip()] = value.split(':')[1:]
                    
            plus_dict[key] = detail_dict
            if detail_dict == {}:
                del plus_dict[key]
            
        except:
            pass
    return plus_dict

def get_second_required(second_stages, index):
    second_stages = second_stages.iloc[:, 1:]
    # PE HEENT, PE Abdomen, PE 하복부 칼럼 데이터만 가져오기
    second_stages = second_stages[['PE HEENT', 'PE Abdomen', 'PE 하복부']]
    
    
    column_li = second_stages.columns
    set_li = []
    set_dict = query_collection(columns=column_li)
    # 빈 리스트 제거
    set_dict = {k: v for k, v in set_dict.items() if v} 
    for key, value_li in set_dict.items():
        for value in value_li:
            if value == '':
                value_li.remove(value)
        
    required_data = get_second_data(second_stages,index)
    return set_dict, required_data
    
    
    




























if __name__ == "__main__":
    import uvicorn
    import sys, os 
    # load_port를 인자로 받기
    load_port = int(sys.argv[1])
    
    
    uvicorn.run(app, host="0.0.0.0", port=load_port)


    # 사용자가 보낸 질문에 대한 대답을 선택 (여기서는 예시로 첫 번째 단계의 랜덤한 답변을 선택)
    # response = data.iloc[random_response_idx]
    
    # 결과를 JSON 형식으로 반환
    # return JSONResponse(content={"question": question, "response": response})

    
    # 질문을 받아야함

# # 신체 진찰에 대한 정답지를 반환하는 API 엔드포인트
# @app.get("/get_second_stage")
