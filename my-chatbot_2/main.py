# 필요한 외부 라이브러리 임포트
import os                     # 환경 변수 접근용 모듈
import json                   # JSON 파일 입출력용 모듈
from dotenv import load_dotenv, find_dotenv  # .env 파일에서 환경변수 로드
from openai import OpenAI     # OpenAI API 사용을 위한 라이브러리
import tiktoken               # 토큰 수 계산용 라이브러리

# .env 파일을 찾아서 환경 변수를 로드
load_dotenv(find_dotenv())

# .env에서 API 키와 시스템 메시지 불러오기
API_KEY = os.environ["API_KEY"]
SYSTEM_MESSAGE = os.environ["SYSTEM_MESSAGE"]

# OpenAI API를 대체 제공하는 Together API의 엔드포인트 설정
BASE_URL = "https://api.together.xyz"

# 기본 사용할 LLM 모델 이름 설정
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

# 대화 기록 저장 파일 이름
FILENAME = "message_history.json"

# 입력 토큰 제한 수 설정
INPUT_TOKEN_LIMIT = 2048

# OpenAI 클라이언트 객체 생성
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 일반적인 채팅 응답 생성 함수 (streaming 없이)
def chat_completion(messages, model=DEFAULT_MODEL, temperature=0.1, **kwargs):
    response = client.chat.completions.create(
        model=model,                # 사용할 모델
        messages=messages,          # 대화 메시지 리스트
        temperature=temperature,    # 창의성 조절 파라미터
        stream=False,               # 스트리밍 응답 비활성화
        **kwargs                    # 추가 파라미터
    )
    return response.choices[0].message.content  # 응답 메시지 반환

# 스트리밍 방식으로 응답 생성하는 함수
def chat_completion_stream(messages, model=DEFAULT_MODEL, temperature=0.1, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        **kwargs
    )

    response_content = ""  # 전체 응답 저장 변수

    for chunk in response:  # 응답 스트림을 한 덩어리씩 읽음
        chunk_content = chunk.choices[0].delta.content
        if chunk_content is not None:
            print(chunk_content, end="")  # 출력 스트리밍
            response_content += chunk_content  # 전체 응답에 누적

    print()
    return response_content  # 전체 응답 문자열 반환

# 주어진 텍스트의 토큰 수를 계산하는 함수
def count_tokens(text, model):
    encoding = tiktoken.get_encoding("cl100k_base")  # 기본 인코딩 방식 선택
    tokens = encoding.encode(text)  # 텍스트를 토큰으로 인코딩
    return len(tokens)  # 토큰 개수 반환

# 전체 메시지의 총 토큰 수 계산 함수
def count_total_tokens(messages, model):
    total = 0
    for message in messages:
        total += count_tokens(message["content"], model)  # 각 메시지의 토큰 수 합산
    return total

# 토큰 수가 제한을 초과하지 않도록 오래된 메시지를 삭제하는 함수
def enforce_token_limit(messages, token_limit, model=DEFAULT_MODEL):
    while count_total_tokens(messages, model) > token_limit:
        if len(messages) > 1:
            messages.pop(1)  # 가장 오래된 사용자 메시지 제거 (system은 유지)
        else:
            break  # 메시지가 하나뿐이면 종료

# 객체를 JSON 파일로 저장하는 함수
def save_to_json_file(obj, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4, ensure_ascii=False)  # JSON 파일로 저장 (인코딩 포함)

# JSON 파일에서 객체를 불러오는 함수
def load_from_json_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)  # JSON 파일 로드
    except Exception as e:
        print(f"{filename} 파일을 읽는 중 오류 발생: {e}")
        return None  # 오류 발생 시 None 반환

# 전체 챗봇 실행 함수
def chatbot():
    # 기존 대화 메시지 불러오기, 없으면 system 메시지로 초기화
    messages = load_from_json_file(FILENAME)
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
            
    # 사용자에게 챗봇 시작 안내 출력
    print("Chatbot: 안녕하세요! 무엇을 도와드릴까요? (종료하려면 'quit' 또는 'exit'을 입력하세요.)")
            
    while True:
        user_input = input("You: ")  # 사용자 입력 받기
        if user_input.lower() in ['quit', 'exit']:
            break  # 종료 명령 시 반복 종료
            
        messages.append({"role": "user", "content": user_input})  # 사용자 메시지 추가
            
        total_tokens = count_total_tokens(messages, DEFAULT_MODEL)  # 현재 전체 토큰 수 계산
        print(f"[현재 토큰 수: {total_tokens} / {INPUT_TOKEN_LIMIT}]")
            
        enforce_token_limit(messages, INPUT_TOKEN_LIMIT)  # 토큰 수 제한 초과 시 오래된 메시지 삭제
            
        print("Chatbot: ", end="")
        response = chat_completion_stream(messages)  # 챗봇 응답 출력 (스트리밍 방식)
        print()
            
        messages.append({"role": "assistant", "content": response})  # 챗봇 응답 메시지 저장
            
        save_to_json_file(messages, FILENAME)  # 전체 메시지 기록 저장
            
# 메인 함수 실행
chatbot()

            