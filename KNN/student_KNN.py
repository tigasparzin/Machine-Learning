import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

plt.figure(figsize=(12, 10))

# Carregar os dados
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

# Pré-processamento dos dados: codificar variáveis categóricas
label_encoders = {}
categorical_cols = ['AcademicStage', 'StudyEnv', 'Strategy', 'BadHabits']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Separar features e target
X = df.drop(columns=['Timestamp', 'Stress'])
y = df['Stress'].astype(int)

# Split correto (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treinar KNN com TODAS as features para métrica
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Visualização (2D) da fronteira de decisão
# Seleciona duas colunas numéricas automaticamente
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

feat1, feat2 = num_cols[0], num_cols[1]

X2 = X[[feat1, feat2]]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y
)

knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X2_train, y2_train)

# Grade para contorno
h = 0.05  # passo da malha
x_min, x_max = X2[feat1].min() - 1, X2[feat1].max() + 1
y_min, y_max = X2[feat2].min() - 1, X2[feat2].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot da fronteira
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)

# Scatter dos dados (teste) para não poluir
sns.scatterplot(
    x=X2_test[feat1], y=X2_test[feat2],
    hue=y2_test.astype(str), style=y2_test.astype(str),
    s=80, edgecolor="k", linewidth=0.5
)

plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title("KNN Decision Boundary (k=3) — Visualização em 2D")
plt.legend(title="Stress", loc="best")

# Exportar SVG para string (como você fez)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches="tight")
svg_text = buffer.getvalue()
print(svg_text)
