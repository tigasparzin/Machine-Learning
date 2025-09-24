import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
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

# Variáveis de entrada e alvo
X = preprocess(df)
y = df['Stress']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Visualize decision boundary
h = 0.02  # Step size in mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, palette="deep", s=100)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Decision Boundary (k=3)")

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())