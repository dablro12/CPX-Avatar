#!/bin/bash
# ngrok 백그라운드로 실행 
mkdir log
nohup /snap/bin/ngrok http --domain=warm-newly-stag.ngrok-free.app 8081 &> /home/eiden/eiden/LLM/CPX-Avatar/log/ngrok.log & # ngrok 백그라운드에서 실행, 로그를 ngrok.log에 저장
nohup /home/eiden/anaconda3/envs/goomon/bin/python3.10 /home/eiden/eiden/LLM/CPX-Avatar/app/server.py 8081 &> /home/eiden/eiden/LLM/CPX-Avatar/log/api_server.log & # API 백그라운드에서 실행, 로그를 server.log에 저장
#https://warm-newly-stag.ngrok-free.app->이 주소로 접속하면 ngrok으로 연결된 서버로 접속 가능
#https://warm-newly-stag.ngrok-free.app/{apiname}?{parameter=value} # 이런 형식으로 사용 가능 