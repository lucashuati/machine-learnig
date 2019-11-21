from abc import abstractmethod
import numpy as np
import math


class Gradiente():
    def __init__(self, arr_dz, arr_dw, db):
        self.arr_dz = arr_dz
        self.arr_dw = arr_dw
        self.db = db

    def __str__(self):
        return "db: "+str(self.db)+" arr_dw: "+str(self.arr_dw)


"""
Atividade 1: Funcoes de ativação - altere a resposta das funcoes lambdas correspondentes
de acordo com a especificação
"""


class FuncaoAtivacao:
    def __init__(self, funcao, dz_funcao, dz_ultima_camada=None):
        self.funcao = funcao
        self.dz_funcao = dz_funcao
        self.dz_ultima_camada = dz_ultima_camada


ALPHA = 0.1
sigmoid = FuncaoAtivacao(
    lambda z: 1 / (1 + np.exp(-z)),
    lambda a, z, y, arr_dz_w_prox: np.multiply(a*(1-a), arr_dz_w_prox),
    lambda a, z, y, arr_dz_w_prox: a - y
)

relu = FuncaoAtivacao(
    lambda z: np.maximum(z, 0),
    lambda a, z, y, arr_dz_w_prox: np.multiply((z >= 0), arr_dz_w_prox)
)

leaky_relu = FuncaoAtivacao(
    lambda z: np.maximum(z, ALPHA * z),
    lambda a, z, y, arr_dz_w_prox: np.multiply(
        (1 * (z >= 0) + ALPHA * (z < 0)), arr_dz_w_prox)
)


tanh = FuncaoAtivacao(
    lambda z: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)),
    lambda a, z, y, arr_dz_w_prox: np.multiply(
        (1 - np.power(np.tanh(z), 2)), arr_dz_w_prox)
)


class Unidade:
    def __init__(self, func_ativacao, dz_func):
        self.b = 0
        self.func_ativacao = func_ativacao
        self.dz_func = dz_func
        self.arr_w = None
        self.arr_z = None
        self.arr_a = None
        self.mat_a_ant = None
        self.gradiente = None

    def __str__(self):
        return "arr_z: "+str(self.arr_z) +\
            "\narr_a:"+str(self.arr_a) +\
            "\ngradiente: "+str(self.gradiente) +\
            "\nmat_a_ant: "+str(object=self.mat_a_ant)

    def z(self, mat_a_ant):
        return self.arr_w.dot(mat_a_ant.T) + self.b

    def forward_propagation(self, mat_a_ant):
        """
        Função que retorna os resultados da função z por instancia usando a matriz mat_a_ant:
        valores de ativações anterior ou entrada (caso seja primeira camada)
        """
        if(self.arr_w is None):
            # np.zeros(qtd_pesos)#np.rand(qtd_pesos)
            self.arr_w = np.random.rand(mat_a_ant.shape[1])*0.01

        self.mat_a_ant = mat_a_ant
        self.arr_z = self.z(self.mat_a_ant)
        self.arr_a = self.func_ativacao(self.arr_z)

        return self.arr_a

    def backward_propagation(self, arr_y, arr_dz_w_prox):
        n_instances = len(arr_y)

        arr_dz = self.dz_func(self.arr_a, self.arr_z, arr_y,
                              arr_dz_w_prox)  # self.dz(arr_da)

        arr_dw = 1/n_instances * arr_dz.dot(self.mat_a_ant)

        db = 1/n_instances * np.sum(arr_dz)

        self.gradiente = Gradiente(arr_dz, arr_dw, db)

        return self.gradiente

    def loss_function(self, arr_y):
        return np.sum(-(arr_y*np.log(self.arr_a)+(1-arr_y)*np.log(1-self.arr_a)))/len(arr_y)

    def atualiza_pesos(self, learning_rate):
        self.arr_w = self.arr_w-learning_rate*self.gradiente.arr_dw
        self.b = self.b-learning_rate*self.gradiente.db


class Camada:
    def __init__(self, qtd_unidades, func_ativacao, func_dz):
        self.arr_unidades = []
        self.mat_a = None
        self.ant_camada = None
        self.prox_camada = None
        self.qtd_un_camada_ant = None

        """
        Atividade 2: Inicialize o vetor de arr_unidades com a unidade correspondete
        """
        for i in range(qtd_unidades):
            self.arr_unidades.append(Unidade(func_ativacao, func_dz))

    def forward_propagation(self, mat_a_ant):
        """
        Atividade 3: Calculo da matriz de ativações mat_a a partir da matriz de ativações da camada anterior mat_a_ant.
        Caso seja a primeira camada, mat_a_ant a matriz de entrada do modelo
        """
        # obtenha a quantidade de unidades na camada anterior  por meio de mat_a_ant
        # ..pense nas dimensões da matriz

        self.qtd_un_camada_ant = np.size(mat_a_ant, 1)
        qtd_un_camada = len(self.arr_unidades)

        # Inicialize com zeros a matriz de ativacao da camada atual
        # ..Novamente, verifique as dimensões da matriz

        self.mat_a = np.zeros((np.size(mat_a_ant, 0), qtd_un_camada + 1))

        # para cada unidade, realiza o forward propagation
        # ...o forward_propagation da unidade retorna um vetor arr_a com as ativações
        # ... por instancia. Você deve armazenar os valores corretamente na matriz mat_a (veja especificação)
        for i, unidade in enumerate(self.arr_unidades):
            self.mat_a[:, i] = unidade.forward_propagation(mat_a_ant)

        return self.mat_a

    # atividade 4: Complete as propriedades mat_w, mat_dz e mat_dz_w
    @property
    def mat_w(self):
        """
        Cria a matriz _mat_w a partir dos vetores de pesos de cada unidade
        Verifique as dimensões da matriz (preencha os None)
        """
        # inicialize com zero a matriz
        # de acordo com as suas dimensoes
        qtd_un_camada = len(self.arr_unidades)
        _mat_w = np.zeros((qtd_un_camada + 1, self.qtd_un_camada_ant))

        # para cada unidade, preencha corretamente os valores da matriz
        # usando o vetor de pesos de cada unidade
        for i, unidade in enumerate(self.arr_unidades):
            _mat_w[i, :] = unidade.arr_w

        return _mat_w

    @property
    def mat_dz(self):
        """
        Cria a matriz _mat_dz a partir dos vetores arr_dz do gradiente de cada unidade
        Verifique as dimensões da matriz (preencha os None)
        """
        # inicialize com zero a matriz
        qtd_un_camada = len(self.arr_unidades)
        _mat_dz = np.zeros((np.size(self.mat_a, 0), qtd_un_camada + 1))

        # para cada unidade, preencha corretamente os valores da matriz
        for i, unidade in enumerate(self.arr_unidades):
            _mat_dz[:, i] = unidade.gradiente.arr_dz

        return _mat_dz

    @property
    def mat_dz_w(self):
        """
        Realiza o calculo do produto entre mat_dz e mat_w
        chame as propriedades correspondentes
        """
        return np.dot(self.mat_dz, self.mat_w)

    def backward_propagation(self, arr_y):
        """
        Atividade 5: Realiza o calculo do backward propagation
        """
        # obtenha o mat_dz_w da proxima camada
        # ..Caso não exista proxima camada, mat_dz_w_prox permanecerá None
        if self.prox_camada:
            mat_dz_w_prox = self.prox_camada.mat_dz_w
        else:
            mat_dz_w_prox = None

        for i, unidade in enumerate(self.arr_unidades):
            # Caso exista mat_dz_w_prox, obtenha o arr_dz_w_prox
            # ..correspondente a esta unidade. Para isso, fique atento a dimensão de mat_dz_w_prox
            if mat_dz_w_prox is None:
                arr_dz_w_prox = None
            else:
                arr_dz_w_prox = mat_dz_w_prox[:, i]

            # chame o backwrd_propagation desta unidade
            unidade.backward_propagation(arr_y, arr_dz_w_prox)

    def atualiza_pesos(self, learning_rate):
        """
        Já está pronto: para cada unidade, atualiza seus pesos
        """
        for unidade in self.arr_unidades:
            unidade.atualiza_pesos(learning_rate)


class RedeNeural:
    def __init__(self, arr_qtd_un_por_camada,
                 arr_func_a_por_camada,
                 num_iteracoes):
        self.arr_camadas = []
        self.arr_qtd_un_por_camada = arr_qtd_un_por_camada
        self.arr_func_a_por_camada = arr_func_a_por_camada
        self.num_iteracoes = num_iteracoes
        self.arr_y = []
        self.mat_x = None

    def config_rede(self, mat_x, arr_y):
        """
        Atividade 6 - Configura a rede: armazena em arr_camada todas as camadas por meio da
        quantidade de unidades por camada e funções de ativação correspondente, inicializa mat_x e arr_y
        """
        self.mat_x = mat_x
        self.arr_y = arr_y

        indice_ultima_camada = len(self.arr_qtd_un_por_camada) - 1
        # para cada camada, ao i
        for camada_l, qtd_unidades in enumerate(self.arr_qtd_un_por_camada):
            # por meio de arr_func_a_por_camada defina a dz_função que será usada
            # ..caso seja a ultima camada, será usada a dz_ultima camada
            if camada_l == indice_ultima_camada:
                dz_funcao = self.arr_func_a_por_camada[camada_l].dz_ultima_camada
            else:
                dz_funcao = self.arr_func_a_por_camada[camada_l].dz_funcao

            # instancie a camda
            obj_camada = Camada(
                qtd_unidades, self.arr_func_a_por_camada[camada_l].funcao, dz_funcao)
            self.arr_camadas.append(obj_camada)

            # armazena a camada anterior
            if(camada_l > 0):
                obj_camada.ant_camada = self.arr_camadas[camada_l-1]

        # para cada camada até a penultima, armazene em camada.prox_camada a camada seguinte
        for l, camada in enumerate(self.arr_camadas):
            if l < len(self.arr_camadas)-1:
                camada.prox_camada = self.arr_camadas[l+1]

    def forward_propagation(self):
        """
        Atividade 7: Execute, para todas as camadas, o método forward_propagation.
        """
        num_camadas = len(self.arr_camadas)
        for l in range(num_camadas):
            if l > 0:
                self.arr_camadas[l].forward_propagation(
                    self.arr_camadas[l-1].mat_a)
            else:
                self.arr_camadas[l].forward_propagation(self.mat_x)

    def backward_propagation(self):
        """
        Atividade 8: Execute, para todas as camadas, o método backward_propagation. Fique atento na ordem de execução
        """
        num_camadas = len(self.arr_camadas)

        for i in range(num_camadas, 0, -1):
            self.arr_camadas[i-1].backward_propagation(self.arr_y)

    def atualiza_pesos(self, learning_rate):
        """
        Já está pronto: chama o atualiza pesos das camadas
        """
        for camada in self.arr_camadas:
            camada.atualiza_pesos(learning_rate)

    def fit(self, mat_x, arr_y, learning_rate=1.1):
        """
        Atividade 9: Realiza self.num_iteracoes iterações (épocas) da rede neural
        """
        # primeiramente, você deve configurar a rede
        self.config_rede(mat_x, arr_y)

        for i in range(self.num_iteracoes):
            # faça aqui a execução desta iteração
            self.forward_propagation()
            self.backward_propagation()
            self.atualiza_pesos(learning_rate)

            if(i % 100 == 0):
                loss = self.loss_function(arr_y)
                print("Iteração: " + str(i) + " Loss: " + str(loss))

    def loss_function(self, arr_y):
        """
        Atividade 10: Calcula a função de perda por meio do vetor de ativações
        """
        # Para calcular o loss_function, voce deverá obter o vetor de
        # ..ativações (arr_a) apropriado. Fique atento com qual camada/unidade você deverá
        # ..obter o arr_a. Preencha os None com o valor apropriado

        indice_ultima_camada = len(self.arr_camadas) - 1
        arr_a = self.arr_camadas[indice_ultima_camada].arr_unidades[0].arr_a

        return np.sum(-np.multiply((1 - arr_y), np.log(1 - arr_a)) - np.multiply(arr_y, np.log(arr_a))) / len(arr_y)

    def predict(self, mat_x):
        """
        Atividade 11: faz a predição, para uma matriz de instancias/atributos mat_x
        """
        # faz a predição, para uma matriz de instancias/atributos mat_x
        # ..para realizar a predição, você deverá fazer o mesmo que na regressão logistica
        # ..porém, deverá ficar atento a qual camada/unidade você deverá obter o vetor de ativações arr_a
        # ..preencha os none com o valor apropriado
        self.mat_x = mat_x
        self.forward_propagation()

        indice_ultima_camada = len(self.arr_camadas) - 1
        arr_a = self.arr_camadas[indice_ultima_camada].arr_unidades[0].arr_a

        return 1 * (arr_a > 0.5)
