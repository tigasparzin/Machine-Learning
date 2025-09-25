import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')
df = df.drop(columns=['Timestamp'])


print(df.head(5).to_markdown(index=False))