
import pandas as pd

df = pd.read_excel(r"C:\Users\gusta\OneDrive\Área de Trabalho\ml-4-poneis-main\ml-4-poneis-main\data\drive-download-20250528T163348Z-1-001\dados_frutas.xlsx")
df # printa a tabela 


# %%
# Aplicando as dicas

# df["Arredondada"] == 1  Retorna um valor booleando dependendo se a fruta é ou não arredondada (esta na tabela inicial com valores binários)
# filtroRedonda = df["Arredondada"] == 1
# df[filtroRedonda] Reduzindo o espaço amostral somente as redondas (filtroRedonda = df["Arredondada"] == 1) serão incluidas 

filtroRedonda = df["Arredondada"] == 1
filtroSuculenta = df["Suculenta"] == 1
filtroVermelha = df["Vermelha"] == 1
filtroDoce = df["Doce"] == 1

df[filtroRedonda & filtroSuculenta & filtroSuculenta & filtroDoce & filtroVermelha] # Multiplos filtros juntos para aumentar a probabilidade de acerto 



df[filtroRedonda & filtroSuculenta & filtroSuculenta & filtroDoce &filtroVermelha] # Multiplos filtros juntos para aumentar a probabilidade de acerto 

# Fazendo a árvore de decisões

from sklearn import tree

features = ["Arredondada", "Suculenta", "Vermelha", "Doce"] # caracteristicas

target = "Fruta" # Variavel que quer achar 

x = df[features]  # Temos um data frame somente as características
y = df[target] # Temos um data frame somente das frutas (target)


arvore = tree.DecisionTreeClassifier() # Definindo um objeto do tipo 'DecisionTreeClassifier'
arvore.fit(x,y) # fit = aprenda, aprenda com os meus dados no caso X e Y --> ISSO É O MACHINE LEARNING 

tree.plot_tree(arvore, class_names= arvore.classes_, feature_names= features, filled=True) # usamos o metodo plot tree e passamos o objeto "arvore" --> MOSTRA OQUE APRENDEU 
# imagem da arvore --> na pasta imagens

arvore.predict([[0, 1, 1, 1]]) # --> mostra a um resultado que bate com essas 'features'
# ([["Arredondada", "Suculenta", "Vermelha", "Doce"]])
# trabalhando com os features definidos na mesma ordem
# 1 = true e 0 = false

prob = arvore.predict_proba([[0, 1, 1, 1]])[0] # --> faz uma lista de probabilidade --> UMA probabibilidade para CADA classe
# ou seja uma lista com a probabilidade de cada fruta 

pd.Series(prob, index = arvore.classes_)
# Cria uma "Lista" --> com a probabilidade de cada classe de acordo com nosso ARRAY "prob"
# Resultado da previsão (probabilidades para cada fruta):
# Banana     0.0
# Cereja     0.5
# Limão      0.0
# Maçã       0.5
# Morango    0.0
# Pera       0.0
# Tomate     0.0
