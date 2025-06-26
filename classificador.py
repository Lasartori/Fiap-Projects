import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

avaliacoes = pd.read_csv("b2w.csv")

def treinar_modelo(dados,coluna_texto, coluna_sentimento):
    vetorizar = CountVectorizer(max_features=100) #Limita as 100 palavras mais recorrentes
    bag_of_words = vetorizar.fit_transform(avaliacoes.review_text)
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,avaliacoes.polarity,stratify=avaliacoes.polarity, random_state=42)
    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(teste, classe_teste)

def word_cloud_neg(dados,coluna_texto):
    texto_negativo = dados.query('polarity == 0')
    todas_avaliacoes = [texto for texto in texto_negativo[coluna_texto]]
    todas_palavras = ' '.join(todas_avaliacoes)
    nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110,collocations=False).generate(todas_palavras)
    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def word_cloud_pos(dados,coluna_texto):
    texto_positivo = dados.query('polarity == 1')
    todas_avaliacoes = [texto for texto in texto_positivo[coluna_texto]]
    todas_palavras = ' '.join(todas_avaliacoes)
    nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110,collocations=False).generate(todas_palavras)
    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis('off')
    plt.show()



# Mostrar todas as colunas
# pd.set_option('display.max_columns', None)

avaliacoes.dropna(inplace=True, axis=0)
avaliacoes = avaliacoes.drop(["original_index", "review_text_processed", "review_text_tokenized", "rating", "kfold_polarity", "kfold_rating"], axis=1)
polaridade = avaliacoes['polarity'].value_counts()

#Exemplo da implementação de um Bag of Words
# texto = ["Este produto é muito bom", "Este produto é muito ruim"]
# vetorizar = CountVectorizer()
# bag_of_words = vetorizar.fit_transform(texto)
# matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns = vetorizar.get_feature_names_out())
# print(matriz_esparsa)

# print(treinar_modelo(avaliacoes, "review_text", "polarity"))


todas_avaliacoes = [texto for texto in avaliacoes.review_text]
todas_palavras = ' '.join(todas_avaliacoes)


#gerar a review negativa sobre as avaliações
# word_cloud_neg(avaliacoes,'review_text')

#gerar a review positiva sobre as avaliações
word_cloud_pos(avaliacoes,'review_text')