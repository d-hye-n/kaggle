import pandas as pd

# 1. 데이터 불러오기 (경로는 본인 PC에 맞게 수정)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. 데이터 크기 및 상위 5개 행 확인
print(train.shape)
print(train.head())

# 3. 심장병(Target) 비중 확인
print(train['target'].value_counts(normalize=True))