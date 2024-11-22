import pandas as pd # visualização e tratamento dos dados
from sklearn.preprocessing import LabelEncoder # Para substituir a variável de Classificação por numérica em uma coluna apenas (0,1,2...)

from sklearn.model_selection import train_test_split, cross_val_score # separar dados, treinamento e teste -- validação cruzada 

from sklearn.svm import SVC # SVM
from sklearn.linear_model import LinearRegression # Regressão Linear 
from sklearn.ensemble import RandomForestClassifier # Random Forest

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error # Testar os dados

from joblib import dump, load # Para salvar e carregar os modelos

import os  # Biblioteca para manipular arquivos e diretórios

# Visualização de dados
import matplotlib.pyplot as plt  # Biblioteca para criação de gráficos
import seaborn as sns  # Biblioteca para visualizações estatísticas
import plotly.express as px  # Biblioteca para gráficos interativos

#----------------------------------------------------------------------------------------------------------------------------------------

class Modelo():             
    def __init__(self):
        self.dicionario = {}     

    def CarregarDataset(self, path):
        colunas = ['Sepala_Comprimento', 'Sepala_Largura', 'Petala_Comprimento', 'Petala_Largura', 'Especies']
        self.df = pd.read_csv(path, names=colunas)

        # Criar pasta para salvar os gráficos
        if not os.path.exists('graficos'):
            os.makedirs('graficos')

        # Histograma interativo separado por espécie // Gráfico para visualizar a distribuição dos dados por espécie
        fig = px.histogram(self.df, x='Sepala_Comprimento', color='Especies', barmode='overlay', title='Histograma Interativo de Comprimento da Sépala')
        fig.write_image('graficos/histograma_sepala_comprimento.png')
        fig.show()

        fig = px.histogram(self.df, x='Sepala_Largura', color='Especies', barmode='overlay', title='Histograma Interativo de Largura da Sépala')
        fig.write_image('graficos/histograma_sepala_largura.png')
        fig.show()

        fig = px.histogram(self.df, x='Petala_Comprimento', color='Especies', barmode='overlay', title='Histograma Interativo de Comprimento da Pétala')
        fig.write_image('graficos/histograma_petala_comprimento.png')
        fig.show()

        fig = px.histogram(self.df, x='Petala_Largura', color='Especies', barmode='overlay', title='Histograma Interativo de Largura da Pétala')
        fig.write_image('graficos/histograma_petala_largura.png')
        fig.show()

    def TratamentoDeDados(self):
        self.df.head()
        self.df.isnull().sum()
        self.df.info()
        le = LabelEncoder()
        self.df['Especie_Tipo'] = le.fit_transform(self.df['Especies'])

        # Gráfico de dispersão interativo // Gráfico para visualizar a relação entre as variáveis
        fig = px.scatter_matrix(self.df, dimensions=['Sepala_Comprimento', 'Sepala_Largura', 'Petala_Comprimento', 'Petala_Largura'], color='Especies', title='Gráfico de Dispersão Interativo')
        fig.write_image('graficos/grafico_dispersao.png')
        fig.show()

        # Mapa de calor das correlações // Gráfico para visualizar as correlações entre as variáveis
        plt.figure(figsize=(10, 6))
        numeric_df = self.df.drop(columns=['Especies'])  # Remover a coluna 'Especies' para evitar o erro
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Mapa de Calor Para Visualizar Correlações entre Variáveis')
        plt.savefig('graficos/mapa_de_calor.png')
        plt.show()

    def Treinamento(self):
        self.x = self.df.drop(['Especie_Tipo', 'Especies'], axis=1)  # Dados das plantas, Informações para treinar o modelo (80%)
        self.y = self.df['Especie_Tipo']  # Classificatória, O que quer ser previsto, o que o modelo vai descobrir (20%)
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)  # Divisão dos dados para treinamento e teste 

        # Treinamento dos modelos
        self.rd = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rd.fit(self.x_train, self.y_train)

        self.lr = LinearRegression()
        self.lr.fit(self.x_train, self.y_train)

        self.svm = SVC()
        self.svm.fit(self.x_train, self.y_train)

        # Dicionário
        self.dicionario = {'Random Forest': self.rd, 'Linear Regression': self.lr, 'SVM': self.svm}

        # Criar pasta para salvar os modelos
        if not os.path.exists('modelos'):
            os.makedirs('modelos')

        # Salvar os modelos treinados
        dump(self.rd, 'modelos/random_forest.joblib')
        dump(self.lr, 'modelos/linear_regression.joblib')
        dump(self.svm, 'modelos/svm.joblib')

        # Validação cruzada do treinamento 
        for nome_modelo, modelo in self.dicionario.items():
            if nome_modelo == 'Linear Regression':
                scores = cross_val_score(modelo, self.x_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
                scores = -scores
                print(f"O erro quadrático médio na validação cruzada para {nome_modelo}: {scores}")
                print(f"A média do erro quadrático médio na validação cruzada para {nome_modelo}: {scores.mean()}")
            else:
                scores = cross_val_score(modelo, self.x_train, self.y_train, cv=5, scoring='accuracy')
                print(f"A acurácia da validação cruzada para {nome_modelo}: {scores}")
                print(f"A média da acurácia da validação cruzada para {nome_modelo}: {scores.mean()}")

    def Teste(self):
        # Prever o conjunto de teste -- ou seja, ele tenta prever a espécie da flor (y) a partir das informações e dados das flores (x)

        # Teste do Random Forest
        y_pred_rd = self.rd.predict(self.x_test)  # O modelo treinado é usado para prever os 20% de X que foram separados para teste
        acuracia_rd = accuracy_score(self.y_test, y_pred_rd)  # A acurácia mede se os valores previstos condizem com os valores reais (y_test)
        print(acuracia_rd, "Acurácia (Random Forest)")
        precisao_rd = precision_score(self.y_test, y_pred_rd, average='macro')
        print(precisao_rd, "Precisão (Random Forest)")
        recall_rd = recall_score(self.y_test, y_pred_rd, average='macro')
        print(recall_rd, "Recall (Random Forest)")
        f1_rd = f1_score(self.y_test, y_pred_rd, average='macro')
        print(f1_rd, "F1 (Random Forest)")
        matriz_rd = confusion_matrix(self.y_test, y_pred_rd)
        print(matriz_rd, "Matriz (Random Forest)")

        # Teste do Linear Regression
        y_pred_lr = self.lr.predict(self.x_test)
        mse_lr = mean_squared_error(self.y_test, y_pred_lr)
        print(mse_lr, "Erro Quadrático Médio (Linear Regression)")

        # Teste do SVM
        y_pred_svm = self.svm.predict(self.x_test)
        acuracia_svm = accuracy_score(self.y_test, y_pred_svm)
        print(acuracia_svm, "Acurácia (SVM)")
        precisao_svm = precision_score(self.y_test, y_pred_svm, average='macro')
        print(precisao_svm, "Precisão (SVM)")
        recall_svm = recall_score(self.y_test, y_pred_svm, average='macro')
        print(recall_svm, "Recall (SVM)")
        f1_svm = f1_score(self.y_test, y_pred_svm, average='macro')
        print(f1_svm, "F1 (SVM)")
        matriz_svm = confusion_matrix(self.y_test, y_pred_svm)
        print(matriz_svm, "Matriz (SVM)")

        # Matriz de Confusão
        from sklearn.metrics import ConfusionMatrixDisplay  # Importando para visualizar a matriz de confusão

        def plot_confusion_matrix(model, X_test, y_test, title):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
            disp.plot(cmap='Blues')
            plt.title(title)
            plt.savefig(f'graficos/{title.lower().replace(" ", "_")}.png')
            plt.show()

        plot_confusion_matrix(self.rd, self.x_test, self.y_test, "Matriz de Confusão - Random Forest")
        plot_confusion_matrix(self.svm, self.x_test, self.y_test, "Matriz de Confusão - SVM")

        # Gráfico de barras para comparar a acurácia dos modelos
        modelos = ['Random Forest', 'SVM']
        scores = [acuracia_rd, acuracia_svm]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=modelos, y=scores)
        plt.ylim(0, 1)
        plt.xlabel('Modelos')
        plt.ylabel('Acurácia')
        plt.title('Comparação de Acurácia dos Modelos')
        plt.savefig('graficos/comparacao_acuracia_modelos.png')
        plt.show()

        # Gráfico de barras para mostrar o Erro Quadrático Médio do modelo de Regressão Linear
        modelos_lr = ['Linear Regression']
        scores_lr = [mse_lr]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=modelos_lr, y=scores_lr)
        plt.xlabel('Modelos')
        plt.ylabel('Erro Quadrático Médio')
        plt.title('Erro Quadrático Médio do Modelo de Regressão Linear')
        plt.savefig('graficos/erro_quadratico_medio_linear_regression.png')
        plt.show()

    def Train(self):
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.
        self.TratamentoDeDados()
        self.Treinamento()  # Executa o treinamento do modelo
        self.Teste()  # Executa o teste do modelo

modelo = Modelo()
modelo.Train()
