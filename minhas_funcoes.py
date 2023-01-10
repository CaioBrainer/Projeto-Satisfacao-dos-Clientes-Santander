# Nesse arquivo estão as funções personalizadas que criei para tratamento dos dados e treinamento do modelo
# bibliotecas necessárias para as funções funcionarem


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA


def plot_multiple_CM(modelo, X_treino, X_teste, y_treino, y_teste):
    # realiza as previsões:
    previsoes_treino = modelo.predict(X_treino)
    previsoes_teste = modelo.predict(X_teste)

    # calcula a matriz de confusão:
    CM_treino = confusion_matrix(y_treino, previsoes_treino, normalize='true')  # normalize='pred'
    CM_teste = confusion_matrix(y_teste, previsoes_teste, normalize='true')  # normalize='pred'

    # projeta a figura:
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].set_title("TREINO")
    ax[1].set_title("TESTE")

    ConfusionMatrixDisplay(
        confusion_matrix=CM_treino,
        display_labels=['Satisfeito', 'Insatisfeito']).plot(ax=ax[0], colorbar=False, cmap='magma')

    ConfusionMatrixDisplay(
        confusion_matrix=CM_teste,
        display_labels=['Satisfeito', 'Insatisfeito']).plot(ax=ax[1], colorbar=False, cmap='magma')

    plt.subplots_adjust(wspace=0.40, hspace=0.1);


def sobreamostragem(modelo, X_treino, X_teste, y_treino, y_teste):
    # syntetic minority oversample tecnique
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X_treino, y_treino)

    modelo.fit(X_smote, y_smote)

    smote_pred = modelo.predict(X_smote)
    cm = confusion_matrix(y_smote, smote_pred, normalize='true')

    plot_multiple_CM(modelo, X_smote, X_teste, y_smote, y_teste);


def area_sob_curva(modelo, X, y):
    y_score = modelo.predict_proba(X)[:, 1]
    falsos_positivos_taxa, verdadeiros_positivos_taxa, threshold1 = roc_curve(y, y_score)

    plt.subplots(figsize=(8, 8))
    plt.title('Área sob a curva')
    plt.plot(falsos_positivos_taxa, verdadeiros_positivos_taxa)
    plt.plot([0, 1], ls="--", color='red')
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('Razão de verdadeiros positivos')
    plt.xlabel('Razão de falsos positivos')
    plt.show()


def treino_pca(modelo, x_treino, x_teste, y_treino, y_teste, taxa_de_variancia=0.99):
    # Aplicando o PCA
    pca_var = PCA(n_components=taxa_de_variancia)
    pca_treino = pca_var.fit_transform(x_treino)
    pca_teste = pca_var.transform(x_teste)
    X_treino_pca = pd.DataFrame(pca_treino)
    X_teste_pca = pd.DataFrame(pca_teste)

    # Treinando o modelo
    modelo.fit(X_treino_pca, np.ravel(y_treino));

    # Plotando a matrix de confusão
    plot_multiple_CM(modelo, X_treino_pca, X_teste_pca, y_treino, y_teste)


def diagrama_variancia(x, taxa_varianca):
    # Iremos utilizar um número de variáveis que explique 99% da variância de nosso dataset
    pca = PCA(n_components=taxa_varianca)
    pca.fit_transform(x)

    # extraindo as variâncias...
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    # De forma gráfica...
    plt.subplots(figsize=(7, 7))
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.9, align='center', label='Variância explicada individual')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Variância explicada cumulativa')
    plt.ylabel('Taxa de variância explicada')
    plt.xlabel('Componentes principais')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
