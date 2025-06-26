import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk import tokenize, ngrams
import seaborn as sns

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

def grafico (dados, coluna_texto, quantidade):
    todas_palavras = ' '.join(texto for texto in dados[coluna_texto])
    token_frase = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)
    #Mostrando as palavras e as suas frequencias
    dataframe_frequencia = pd.DataFrame({'Palavra': list(frequencia.keys()), 'Frequencia':list(frequencia.values())})
    #Mostrando as 10 palavras que mais aparecem e suas frequencias
    dataframe_frequencia = dataframe_frequencia.nlargest(columns= 'Frequencia', n = quantidade)
    #Criar um gráfico usando o seaborn para apresentações
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = dataframe_frequencia, x = 'Palavra', y = 'Frequencia', color='lightblue')
    ax.set(ylabel = 'contagem')
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
#word_cloud_neg(avaliacoes,'review_text')

#gerar a review positiva sobre as avaliações
#word_cloud_pos(avaliacoes,'review_text')

token_espaco = tokenize.WhitespaceTokenizer()
token_dataset = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_dataset)

#Gerar gráfico com a função grafico
# grafico(avaliacoes, 'review_text', 20)

palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')

frase_processada = list()
for avaliacao in avaliacoes.review_text:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(avaliacao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

avaliacoes['texto_sem_stopwords'] = frase_processada


#Stemming em RSLP (Português)
stemmer = nltk.RSLPStemmer()
stemmer.stem('Corredor')
stemmer.stem('Corre')
stemmer.stem('Correria')


frase_processada = list()
for avaliacao in avaliacoes.review_text:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(avaliacao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(stemmer.stem(palavra))
    frase_processada.append(' '.join(nova_frase))
avaliacoes['texto_stemmizado'] = frase_processada

# treinar_modelo(avaliacoes, 'texto_stemmizado', 'polarity')

# word_cloud_neg(avaliacoes,'texto_stemmizado')

#Uso do TF - IDF 
# tfidf = TfidfVectorizer(lowercase=False, max_features=100)
# tfidf_tratados = tfidf.fit_transform(avaliacoes.texto_stemmizado)

# treino,teste,classe_treino,classe_teste = train_test_split(tfidf_tratados, avaliacoes.polarity,stratify=avaliacoes.polarity, random_state=42)
# regressao_logistica = LogisticRegression()
# regressao_logistica.fit(treino, classe_treino)
# acuracia_tfidf = regressao_logistica.score(teste, classe_teste)

#Ngrams
tfidf = TfidfVectorizer(lowercase=False, ngram_range=(1,2))
vetor_tfidf = tfidf.fit_transform(avaliacoes.texto_stemmizado)
treino,teste,classe_treino,classe_teste = train_test_split(vetor_tfidf, avaliacoes.polarity, random_state=42)
regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf = regressao_logistica.score(teste, classe_teste)
# print(acuracia_tfidf)

pesos = pd.DataFrame(regressao_logistica.coef_[0].T, index=tfidf.get_feature_names_out())
maiores_pesos = pesos.nlargest(10,0)
print(maiores_pesos)

