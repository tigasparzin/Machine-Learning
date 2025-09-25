import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import pandas as pd


plt.figure(figsize=(12, 10))

# Preprocess the data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # remocao da coluna Timestamp
    df = df.drop(columns=['Timestamp'])

    # Tratamento de missing values
    ## 'Study Environment' tem 1 valor ausente -> preenchid com a moda (valor mais frequente)
    df['StudyEnv'].fillna(df['StudyEnv'].mode()[0], inplace=True)

    #conversao de variaveis categoricas
    label_encoder = LabelEncoder()
    df['AcademicStage'] = label_encoder.fit_transform(df['AcademicStage'])
    df['StudyEnv'] = label_encoder.fit_transform(df['StudyEnv'])
    df['Strategy'] = label_encoder.fit_transform(df['Strategy'])
    df['BadHabits'] = label_encoder.fit_transform(df['BadHabits'])

    # Selecao de features
    features = [
        'AcademicStage',                       
        'PeerPressure',                       
        'HomePressure',    
        'StudyEnv',                            
        'Strategy',                      
        'BadHabits',
        'AcademicComp' 
    ]

    return df[features]

# Carregar base
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')


X = preprocess(df)

#run Kmeans
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

#Plot
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', s=50)
cent = kmeans.cluster_centers_
plt.scatter(cent[:, 0], cent[:, 1], c='red', marker='*', s=200, label='Centroids')



plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# # Print centroids and inertia
# print("Final centroids:", kmeans.cluster_centers_)
# print("Inertia (WCSS):", kmeans.inertia_)

# # Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
