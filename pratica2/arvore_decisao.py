from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def cria_modelo(x, y, min_samples):
    """
        Retorna o modelo a ser usado.

        X: matriz (ou DataFrame) em que cada linha é um exemplo e cada coluna é uma feature (atributo/caracteristica) do mesmo
        y: para cada posição i, y[i] é a classe alvo (ground truth) do exemplo x[i]
        min_samples: define o mínimo de exemplos necessários para que um nodo da árvore efetue a divisão
    """
    decision_tree_classifier = DecisionTreeClassifier(
        random_state=1, min_samples_split=min_samples)

    return decision_tree_classifier.fit(x, y)

def divide_treino_teste(df, val_proporcao_treino):
    """
        A partir do DataFrame df, faz a divisão entre treino e teste obedecendo a proporção val_proporcao_treino.
    """
    #1. obtenha o treino usando o método sample do DataFrame
    df_treino = df.sample(frac=val_proporcao_treino, random_state=1)

    #2. Para obter o teste, selecione as instancias que estão em df e não estão em df_treino (use o método drop)
    df_teste = df.drop(df_treino.index)

    return df_treino, df_teste



def faz_classificacao(x_treino,y_treino,x_teste,y_teste,min_samples):
    """
        Efetua a classificação, retornando:
            - O vetor y_predicted em que, para cada posição i,
             retorna o resultado previsto do exemplo representado
             por X_teste[i] que a classe alvo seria y_teste[i].
            - A acuracia (proporção de exemplos classificados corretamente)
                dicas:
                * caso tenhamos dois vetores a e b, ao fazer a operção a==b, ele retornará
                um vetor em que o valor  de cada posição i será igual a verdadeiro caso a==b.
                * np.sum soma os valores de um vetor (considerando True=1 e False=0)
    """
    #cria o modelo (use a função previamente criada)
    model_dtree = cria_modelo(x_treino, y_treino, min_samples)

    #realiza a predição (use o método predict do modelo)
    y_predicted = model_dtree.predict(x_teste)

    #calcule a acurácia
    acuracia = sum(y_teste==y_predicted) / len(y_teste)


    return y_predicted,acuracia

def plot_performance_min_samples(X_treino,y_treino,X_teste,y_teste):
    """
        Crie um gráfico em que o eixo x é a variação do parametro min_sample e,
        o eixo y, representará a acurácia.
        Você deverá veriar o min_samples de 0.001 até 0.7 de 0.01 em 0.01 passos.
        Crie duas linhas: representando a acurácia no treino durante a variação do
        min_sample e, a outra, a acuracia do teste com os diversos valores de min_sample.
        Dicas:
            - A função arange do numpy pode ser usada no for (ao invés de range). Pois o range
            permite apenas passos com valores inteiros
            - para obter a acurácia no treino, o teste deverá possuir as mesmas instancias
            do treino
            - Entenda como é feito para plotar o grafico: https://matplotlib.org/users/pyplot_tutorial.html
    """
    arr_ac_treino = []
    arr_ac_teste = []
    arr_min_samples =[]
    for min_samples in np.arange(0.001,0.7,0.01):
        #complete a linha abaixo com a função e parametros corretos para calcular a acurácia no teste
        y_predicted, ac_teste = faz_classificacao(X_treino,y_treino,X_teste,y_teste,min_samples)
        #complete a linha abaixo com a função e parametros corretos para calcular a acurácia no treino
        y_predicted, ac_treino = faz_classificacao(X_treino,y_treino,X_treino,y_treino,min_samples)

        #adiciona a acuracia no treino, no teste e o parametro min_samples
        arr_ac_treino.append(ac_treino)
        arr_ac_teste.append(ac_teste)
        arr_min_samples.append(min_samples)

    #plota o resultado
    plt.plot(arr_min_samples,arr_ac_treino,"b--")
    plt.plot(arr_min_samples,arr_ac_teste,"r-")
