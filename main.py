import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

# Leer el archivo y manejar valores nulos
def read_data(file):
    data = pd.read_csv(file)
    data = data.drop(columns=['Cabin'])
    print("PRE: Valores Nulos detectados en df:")
    print("Shape PRE", data.shape)
    print(data.isna().sum())
    print("POST: Valores Nulos detectados en df:")
    print(data.isna().sum())
    print("Shape POST", data.shape)
    return data

# Mapear variable categórica 'Sex' a valores numéricos
def map_sex(data):
    sex_mapping = {'male': 0, 'female': 1}  
    data['SexNumer'] = data['Sex'].map(sex_mapping)
    return data

# Realizar codificación one-hot de la variable 'Embarked'
def one_hot_encoding(data):
    encoder = OneHotEncoder(sparse=False)
    embarked_encoded = encoder.fit_transform(data[['Embarked']])
    embarked_categories = encoder.categories_[0]
    embarked_df = pd.DataFrame(embarked_encoded, columns=[f'Embarked_{category}' for category in embarked_categories], index=data.index)
    data = pd.concat([data, embarked_df], axis=1)
    return data

# Extraer valores numéricos del boleto y crear una nueva columna 'TicketNum'
def extract_ticket_number(data):
    ticket_num = data['Ticket'].apply(lambda x: re.findall(r'\d+', x))
    ticket_num = ticket_num.apply(lambda x: int(x[0]) if len(x) > 0 else None)
    data['TicketNum'] = ticket_num
    return data

def optimize_model_params(model, params, X_train, y_train):
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def create_nn_model(X_train_scaled):
    model_nn = Sequential()
    model_nn.add(Dense(128, input_shape=(X_train_scaled.shape[1],), activation="relu"))
    model_nn.add(Dense(64, activation="relu"))
    model_nn.add(Dense(32, activation="relu"))
    model_nn.add(Dense(1, activation="sigmoid"))
    model_nn.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model_nn

def main():
    file = 'titanic.csv'

    data = read_data(file)
    print("SHAPE:-------",len(data))
    data = map_sex(data)
    data = one_hot_encoding(data)
    data = extract_ticket_number(data)
    selected_features = data.select_dtypes(include=['float64', 'int64'])

    print("DataFrame con codificación one-hot de 'Embarked' y columna 'TicketNum':")
    print(data.head(10))

    correlation_matrix_selected = selected_features.corr()
   
    plt.figure(figsize=(10, 12))
    heatmap = sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30, fontname='Arial', fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontname='Arial', fontsize=10, rotation=30)
    plt.title('Matriz de Correlación (Means)')
    plt.savefig('Matriz de Correlación (Means).jpg')
    plt.show()
   
    X = selected_features.drop(columns=['Survived'])
    y = selected_features['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Entrenamiento y evaluación de Decision Tree con validación cruzada
    model_tree = DecisionTreeClassifier()
    tree_scores = cross_val_score(model_tree, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print("Exactitud del modelo de árbol de decisiones con validación cruzada:", np.mean(tree_scores))

    # Entrenamiento y evaluación de Regresión Logística con validación cruzada
    model_lr = LogisticRegression()
    lr_scores = cross_val_score(model_lr, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print("Exactitud del modelo de regresión logística con validación cruzada:", np.mean(lr_scores))

    # Entrenamiento y evaluación de Adaboost con validación cruzada
    model_ada = AdaBoostClassifier()
    ada_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
    best_ada_model, best_ada_params = optimize_model_params(model_ada, ada_params, X_train_scaled, y_train)
    ada_scores = cross_val_score(best_ada_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print("Exactitud del modelo Adaboost con validación cruzada:", np.mean(ada_scores))

    # Entrenamiento y evaluación de Red Neuronal con validación cruzada
    nn_model = KerasClassifier(build_fn=create_nn_model, epochs=10, batch_size=10, verbose=0, X_train_scaled=X_train_scaled)
    nn_scores = cross_val_score(nn_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print("Exactitud del modelo de Red Neuronal con validación cruzada:", np.mean(nn_scores))

    # Ajustar modelos finales

    # Ajustar modelo de Decision Tree
    model_tree.fit(X_train_scaled, y_train)

    # Ajustar modelo de Regresión Logística
    model_lr.fit(X_train_scaled, y_train)

    # Ajustar modelo de Adaboost
    best_ada_model.fit(X_train_scaled, y_train)

    # Ajustar modelo de Red Neuronal
    nn_model.fit(X_train_scaled, y_train)

    # Calcular y graficar la curva ROC para Decision Tree, Regresión Logística, Adaboost y Red Neuronal
    plt.figure(figsize=(8, 6))

    # Curva ROC para la Red Neuronal
    y_pred_nn_prob = nn_model.predict_proba(X_test_scaled)[:, 1]
    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_nn_prob)
    auc_nn = roc_auc_score(y_test, y_pred_nn_prob)
    sns.lineplot(x=fpr_nn, y=tpr_nn, label=f'Neural Network (AUC = {auc_nn:.3f})')

    # Curva ROC para Regresión Logística
    y_pred_lr_prob = model_lr.predict_proba(X_test_scaled)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_prob)
    auc_lr = roc_auc_score(y_test, y_pred_lr_prob)
    sns.lineplot(x=fpr_lr, y=tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})')

    # Curva ROC para Decision Tree
    y_pred_tree_prob = model_tree.predict_proba(X_test_scaled)[:, 1]
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_pred_tree_prob)
    auc_tree = roc_auc_score(y_test, y_pred_tree_prob)
    sns.lineplot(x=fpr_tree, y=tpr_tree, label=f'Decision Tree (AUC = {auc_tree:.3f})')

    # Curva ROC para Adaboost
    y_pred_ada_prob = best_ada_model.predict_proba(X_test_scaled)[:, 1]
    fpr_ada, tpr_ada, _ = roc_curve(y_test, y_pred_ada_prob)
    auc_ada = roc_auc_score(y_test, y_pred_ada_prob)
    sns.lineplot(x=fpr_ada, y=tpr_ada, label=f'Adaboost (AUC = {auc_ada:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('ROC Curve_All.jpg')
    plt.show()

if __name__ == "__main__":
    main()
