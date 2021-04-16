# PredatorPrey 
## 현재 실행가능한 MARL모델 : QMIX, COMA
Unity로 멀티 에이전트 강화학습(MARL) 수행하기!

ML-Agent를 이용하여 유니티 기반의 MARL 프레임웍을 제공합니다.

Unity 기반의 MARL 환경 제작이 필요하신분께 큰 도움이 되기를 바랍니다.~

<img src="https://user-images.githubusercontent.com/17878413/114650020-760ada80-9d1c-11eb-8c5d-3a84173f7c9b.png" width="40%"></img>


# 환경 설정 방법 및 요구 사양
필요한 패키지들을 설치하기 위하여 아래대로 명령을 입력하세요. 자동으로 설치 됩니다.
 ```shell
pip install -r requirements.txt
```
요구 사양
 ```shell
python 3.6 
Unity 3D 2020.2.2f1
Unity ML-Agent : ml-agents-release_12
```

# 실행 방법 및 Unity Project
PredatorPrey 환경은 main.py를 실행 하시면 됩니다.

Unity 관련 코드 및 자료는 Unity_PredatorPrey에 있습니다. 

PredatorPrey게임 실행 파일은 envs/PredatorPrey_Game/PredatorPrey.exe 입니다.

# 학습된 모델 사용
config/default.yaml 에서 "checkpoint_path" 값을 아래와 같이 세팅 하시면 됩니다.
 ```shell
checkpoint_path: "learning_results/QMIX" 
```


