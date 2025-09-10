# knn_stress.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Carrega base
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

# pre-processamento
# remoção da coluna Timestamp
df = df.drop(columns=['Timestamp'])

# Tratamento de missing values
# 'StudyEnv' tem 1 valor ausente -> preencher com a moda (valor mais frequente)
df['StudyEnv'].fillna(df['StudyEnv'].mode()[0], inplace=True)

# conversão de variáveis categóricas (apenas estas colunas)
label_encoder = LabelEncoder()
df['AcademicStage'] = label_encoder.fit_transform(df['AcademicStage'])
df['StudyEnv']      = label_encoder.fit_transform(df['StudyEnv'])
df['Strategy']      = label_encoder.fit_transform(df['Strategy'])
df['BadHabits']     = label_encoder.fit_transform(df['BadHabits'])

# Definir target
df['Stress'] = df['Stress'].astype(int)

X = df.drop(columns=['Stress'])
y = df['Stress']

# ESCALA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# transforma em 2 dimensoes
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# Treino kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# vizualizar fronteria
plt.figure(figsize=(12, 10))
h = 0.02

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)

sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1],
    hue=y.astype(str), style=y.astype(str),
    s=60, edgecolor='black', linewidth=0.5, palette="deep"
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KNN Decision Boundary com PCA (k=3)")
plt.tight_layout()

# export SVG
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

