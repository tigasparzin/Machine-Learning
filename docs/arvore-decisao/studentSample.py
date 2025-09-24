import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

print(df.head(5).to_markdown(index=False))