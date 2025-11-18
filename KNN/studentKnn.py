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


# Carregar base
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

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


r = permutation_importance(
    knn,                  
    X_test,               
    y_test,               
    n_repeats=30,         
    random_state=42,
    scoring='accuracy'    
)


feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': r.importances_mean,
    'Std': r.importances_std
})

report_dict = classification_report(y_test, predictions, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

cm = confusion_matrix(y_test, predictions)
labels = knn.classes_
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# ordenar e mostrar (HTML igual ao seu exemplo)
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("<br>Feature Importances (Permutation):")
print(feature_importance.to_html(index=False))

# print("<h3>Relatório de Classificação:</h3>")
# print(report_df.to_html(classes="table table-bordered table-striped", border=0))

# Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduzir para 2 dimensões (apenas para visualização)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Treinar KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)


# Visualize decision boundary
h = 0.02  # Step size in mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, style=y, palette="deep", s=100)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Decision Boundary (k=3)")

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())