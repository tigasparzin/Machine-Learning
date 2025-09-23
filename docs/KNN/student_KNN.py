import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns

# Carregar os dados
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

# Pré-processamento dos dados
# Codificar variáveis categóricas
label_encoders = {}
categorical_cols = ['AcademicStage', 'StudyEnv', 'Strategy', 'BadHabits']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Separar features e target
X = df.drop(['Timestamp', 'Stress'], axis=1)
y = df['Stress']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA para reduzir a 2 componentes para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Treinar KNN com os componentes principais
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_pca, y_train)

# Prever no conjunto de treino para visualização
predictions = knn.predict(X_pca)
accuracy = accuracy_score(y_train, predictions)
print(f"Acurácia: {accuracy:.2f}")

# Criar figura para visualização
plt.figure(figsize=(12, 10))

# Visualizar decision boundary
h = 0.02  # Step size in mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette="deep", s=100)
plt.xlabel("Primeiro Componente Principal")
plt.ylabel("Segundo Componente Principal")
plt.title(f"KNN Decision Boundary com PCA (k=5)\nAcurácia: {accuracy:.2f}")

# Gerar SVG como string
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())