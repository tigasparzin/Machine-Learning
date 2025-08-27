import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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


plt.figure(figsize=(12, 10))

# Carregar base
df = pd.read_csv('./data/StressExp.csv')

# Variáveis de entrada e alvo
X = preprocess(df)
y = df['Stress']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())

