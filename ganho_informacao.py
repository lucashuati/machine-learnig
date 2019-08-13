import math


def entropia(df_dados, nom_col_classe):
    """
        Calcula a entropia de acordo com df_dados (DataFrame) e a classe. Use a função math.log com
        o log na base 2. Não esqueça de importar o módulo math.

        df_dados: Dados a serem considerados para o calculo da entropia
        nom_col_classe: nome da coluna (em df_dados) que representa a classe
    """
    # ser_count_col armazena, para cada valor da classe, a sua quantidade
    ser_count_col = df_dados[nom_col_classe].value_counts()
    num_total = len(df_dados)
    entropia = 0

    # Navege em ser_count_col para fazer o calculo da entropia
    for item, count_atr in ser_count_col.iteritems():
        #altere os valores de val_prob e entropia para o calculo correto da mesma
        #prop_instances deverá ser a proporção de instancias de uma determinada classe
        #caso tenha duvida sobre o iteritems e value_counts, consulte o passo a passo do pandas
        val_prob = count_atr / num_total
        entropia += (-val_prob * math.log(val_prob, 2))
    return entropia


def ganho_informacao_condicional(df_dados,val_entropia_y,nom_col_classe,nom_atributo,val_atributo):
    """
    Calcula o GI(Y|nom_atributo=val_atributo), ou seja,
    calcula o ganho de informação do atributo 'nom_atributo'
    quando ele assume o valor 'val_atributo'.
    O valor de Entropia(Y) já foi calculado e está armazenado em val_entropia_y.
    Dica: A entropia condicional pode ser calculada filtrando o DataFrame df_dados.

    df_dados: Dataframe com os dados a serem analisados.
    val_entropia_y: Entropia(Y) (ver slides)
    nom_col_classe: nome da coluna que representa a classe
    nom_atributo: atributo a ser calculado o ganho de informação
    val_atributo: valor do atributo a ser considerado para este calculo
    """
    val_gi = 0
    # em df_dados_filtrado, filtre o df_dados da forma correta. 
    # Lembre que df_dados é um DataFrame
    df_dados_filtrado = df_dados[df_dados[nom_atributo]==val_atributo]
    ent_condicional = entropia(df_dados_filtrado, nom_col_classe)

    #use df_dados_filtrado obter o ganho de informação armazene em val_gi
    val_gi = val_entropia_y - ent_condicional

    #para testes:
    #print("GI({classe}| {atr}={val}) = {val_gi}".format(classe=nom_col_classe,atr=nom_atributo,val=val_atributo,val_gi=val_gi))

    return val_gi


def ganho_informacao(df_dados,nom_col_classe,nom_atributo):
    """
        Calcula GI(Y| nom_atributo), ou seja, o ganho de informação do atributo nom_atributo.

        df_dados: DataFrame com os dados a serem analisados.
        nom_col_classe: nome da coluna que representa a classe
        nom_atributo: atributo a ser calculado o ganho de informação
        val_atributo: valor do atributo a ser considerado para este calculo
    """
    #Muito similar ao codigo da entropia, mas aqui você deverá navegar sobre
    #os possiveis valores do atributo nom_atributo para calcular o infoGain
    val_entropia_y = entropia(df_dados, nom_col_classe)
    ser_count_col = df_dados[nom_atributo].value_counts()
    num_total = len(df_dados)
    res = 0
    for atr, count_atr in ser_count_col.iteritems():  
        res += (count_atr / num_total) * (
            ganho_informacao_condicional(
                df_dados,
                val_entropia_y,
                nom_col_classe,
                nom_atributo,
                atr
        ))
    return res