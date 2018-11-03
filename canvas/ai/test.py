import pandas as pd

train = pd.read_csv('train/train.csv')
test = pd.read_csv('test/test.csv')

print(train.shape)
print(test.shape)