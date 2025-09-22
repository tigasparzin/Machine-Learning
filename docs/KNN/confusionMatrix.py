import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import cycle

plt.figure(figsize=(12, 10))

def standardization(df):
    # normalizações nas variáveis numéricas (1–5)
    df['Z-Peers']  = df['PeersPressure'].apply(lambda x: (x - df['PeersPressure'].mean())/df['PeersPressure'].std())
    df['N-Peers']  = df['PeersPressure'].apply(lambda x: (x - df['PeersPressure'].min())/(df['PeersPressure'].max()-df['PeersPressure'].min()))
    df['Z-Family'] = df['FamilyPressure'].apply(lambda x: (x - df['FamilyPressure'].mean())/df['FamilyPressure'].std())
    df['N-Family'] = df['FamilyPressure'].apply(lambda x: (x - df['FamilyPressure'].min())/(df['FamilyPressure'].max()-df['FamilyPressure'].min()))
    df['Z-Comp']   = df['Competition'].apply(lambda x: (x - df['Competition'].mean())/df['Competition'].std())
    df['N-Comp']   = df['Competition'].apply(lambda x: (x - df['Competition'].min())/(df['Competition'].max()-df['Competition'].min()))
    # escolha de features (mantendo 4 colunas como no seu exemplo de X)
    features = ['N-Peers', 'N-Family', 'N-Comp', 'AcademicStage', 'StudyEnv', 'Strategy', 'BadHabits', 'Stress']
    return df[features]

def preprocess(df):
    # na
    df['StudyEnv'].fillna(df['StudyEnv'].mode()[0], inplace=True)
    # encodar categóricas
    le = LabelEncoder()
    df['AcademicStage'] = le.fit_transform(df['AcademicStage'].astype(str))
    df['StudyEnv']      = le.fit_transform(df['StudyEnv'].astype(str))
    df['Strategy']      = le.fit_transform(df['Strategy'].astype(str))
    df['BadHabits']     = le.fit_transform(df['BadHabits'].astype(str))
    # selecionar colunas base (numéricas + categóricas + alvo)
    cols = ['PeersPressure','FamilyPressure','Competition','AcademicStage','StudyEnv','Strategy','BadHabits','Stress']
    return df[cols]

# carregar dataset
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

# preprocess + standardização
d = preprocess(df.copy())
d = standardization(d)

# definir X (4 colunas como no seu padrão) e y
X = d[['N-Peers', 'N-Family', 'N-Comp', 'AcademicStage']]
y = d['Stress'].astype(int)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# knn
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# permutation importance
r = permutation_importance(knn, X_test, y_test, n_repeats=30, random_state=42, scoring='accuracy')
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': r.importances_mean, 'Std': r.importances_std})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("<br>Feature Importances (Permutation):")
print(feature_importance.to_html(index=False))

# classification report + confusion matrix
report_dict = classification_report(y_test, predictions, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
cm = confusion_matrix(y_test, predictions)
labels = knn.classes_
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("<h3>Relatório de Classificação:</h3>")
print(report_df.to_html(classes="table table-bordered table-striped", border=0))

# scaler + pca 2D
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# split visualização
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

# treinar knn no espaço PCA
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# fronteira
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y.astype(str), style=y.astype(str), s=100)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KNN Decision Boundary (k=3) — PCA 2D")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
