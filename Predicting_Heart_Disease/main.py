import pandas as pd
import os

# 현재 실행 중인 main.py의 디렉토리 위치를 가져옵니다.
base_path = os.path.dirname(os.path.abspath(__file__))
# 데이터 파일의 전체 경로를 생성합니다.
train_file_path = os.path.join('..', 'data', 'playground-series-s6e2', 'train.csv')
test_file_path = os.path.join('..', 'data', 'playground-series-s6e2', 'test.csv')

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

# 2. 데이터 크기 및 상위 5개 행 확인
print(train.shape)
print(train.head())

# 3. 심장병(Target) 비중 확인
print(train['Heart Disease'].value_counts(normalize=True))