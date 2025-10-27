import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# carregar os dados
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

# Preprocess the data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # remocao da coluna Timestamp
    df = df.drop(columns=['Timestamp'])

    # Tratamento de missing values
    ## 'Study Environment' tem 1 valor ausente -> preenchid com a moda (valor mais frequente)
    df['StudyEnv'] = df['StudyEnv'].fillna(df['StudyEnv'].mode().iloc[0])
    
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


d = preprocess(df.copy())

X = d[['AcademicStage','PeerPressure',                       
        'HomePressure',    
        'StudyEnv',                            
        'Strategy',                      
        'BadHabits',
        'AcademicComp']]
# alvo binário: baixo (<3) e alto (>=3)
y = (df['Stress'] >= 3).map({False: 1, True: 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# gerar matriz de confusão
labels = np.sort(np.unique(np.concatenate([y_test, predictions])))
cm = confusion_matrix(y_test, predictions, labels=labels)

# plotar matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d", colorbar=False)

ax.set_title("Matriz de Confusão - KNN")
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")

# exportar para SVG
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches="tight")
svg_text = buffer.getvalue()
print(svg_text)

plt.close(fig)