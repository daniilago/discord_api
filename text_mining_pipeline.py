"""
Pipeline de Mineração de Texto — Discord
Etapas: Pré-processamento -> TF-IDF -> LDA (tópicos) -> Sentimento (HuggingFace)
"""
import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from collections import Counter

# NLP
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sentimento
from transformers import pipeline as hf_pipeline

warnings.filterwarnings("ignore")

os.makedirs("text_mining_plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    csvs = glob.glob("data/channel_history/**/*.csv", recursive=True)
    if not csvs:
        raise FileNotFoundError("Nenhum CSV encontrado em data/channel_history/")
    print(f"Carregando: {csvs[0]}")
    df = pd.read_csv(csvs[0])
    print(f"Total de mensagens carregadas: {len(df)}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. FILTRAGEM INICIAL
# ─────────────────────────────────────────────────────────────────────────────

BOTS = {"Loritta#0219", "DomoAI#7882", "Viggle#6074"}
DELETED = {"Deleted User#0000"}

def filtrar_mensagens(df: pd.DataFrame) -> pd.DataFrame:
    antes = len(df)

    # Remove bots e usuários deletados
    df = df[~df['autor'].isin(BOTS | DELETED)].copy()

    # Remove mensagens que são só anexo/imagem (sem texto real)
    df = df[~df['conteudo'].str.startswith('[', na=True)].copy()

    # Remove mensagens muito curtas (< 8 chars após strip)
    df = df[df['conteudo'].str.strip().str.len() >= 8].copy()

    depois = len(df)
    print(f"Após filtragem: {depois} mensagens ({antes - depois} removidas)")
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PRÉ-PROCESSAMENTO TEXTUAL
# ─────────────────────────────────────────────────────────────────────────────

nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])

STOPWORDS_EXTRAS = {
    "kk", "kkk", "kkkk", "kkkkk", "kkkkkk", "kkkkkkk", "kkkkkkkk",
    "rs", "haha", "hehe", "vc", "vcs", "pq", "tb", "tbm", "mt", "mto",
    "nd", "nada", "nao", "sim", "ai", "ae", "la", "aq", "aqui",
    "ta", "tá", "ja", "já", "ne", "né", "eh", "hj", "msm", "mesmo",
    "cara", "mano", "bro", "mlk", "tipo", "gnt", "pra", "pro", "num",
    "nem", "sd", "sla", "slk", "slc", "oq", "q", "ok", "dps", "dp",
    "ir", "ser", "ter", "ter", "fazer", "ficar", "dar", "vir",
}

def limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    # Remove menções @usuario
    texto = re.sub(r'<@!?\d+>', '', texto)
    # Remove menções de cargo
    texto = re.sub(r'<@&\d+>', '', texto)
    # Remove emojis Discord customizados <:nome:id>
    texto = re.sub(r'<a?:[a-zA-Z0-9_]+:\d+>', '', texto)
    # Remove links
    texto = re.sub(r'https?://\S+', '', texto)
    # Remove sequências de K (risadas)
    texto = re.sub(r'\bk{2,}\b', '', texto, flags=re.IGNORECASE)
    # Remove caracteres especiais exceto letras, números e espaços
    texto = re.sub(r'[^\w\s]', ' ', texto)
    # Remove números isolados
    texto = re.sub(r'\b\d+\b', '', texto)
    # Normaliza espaços
    texto = re.sub(r'\s+', ' ', texto).strip().lower()
    return texto

def lematizar(texto: str) -> str:
    doc = nlp(texto)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
        and token.lemma_ not in STOPWORDS_EXTRAS
        and token.text not in STOPWORDS_EXTRAS
    ]
    return " ".join(tokens)

def preprocessar(df: pd.DataFrame) -> pd.DataFrame:
    print("Limpando textos...")
    df['texto_limpo'] = df['conteudo'].apply(limpar_texto)

    print("Lematizando (pode demorar ~1-2 min)...")
    df['texto_processado'] = df['texto_limpo'].apply(lematizar)

    # Remove textos que ficaram vazios após processamento
    df = df[df['texto_processado'].str.strip().str.len() > 3].copy()
    print(f"Após pré-processamento: {len(df)} mensagens com conteúdo válido")
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def calcular_tfidf(df: pd.DataFrame):
    print("\nCalculando TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=5,        # palavra deve aparecer em pelo menos 5 mensagens
        max_df=0.85,     # ignora palavras em mais de 85% das mensagens
        ngram_range=(1, 2),  # unigramas e bigramas
    )
    matriz = vectorizer.fit_transform(df['texto_processado'])
    print(f"Vocabulário TF-IDF: {len(vectorizer.get_feature_names_out())} termos")
    return matriz, vectorizer

def plot_top_tfidf(vectorizer, matriz, n=20):
    media_tfidf = np.asarray(matriz.mean(axis=0)).flatten()
    features = vectorizer.get_feature_names_out()
    top_idx = media_tfidf.argsort()[-n:][::-1]
    top_termos = [features[i] for i in top_idx]
    top_scores = [media_tfidf[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_termos[::-1], top_scores[::-1], color='#4A90D9')
    ax.set_title(f'Top {n} Termos por TF-IDF Médio', fontsize=13, fontweight='bold')
    ax.set_xlabel('TF-IDF Médio')
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.savefig('text_mining_plots/tfidf_top_termos.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico TF-IDF salvo.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODELAGEM DE TÓPICOS (LDA)
# ─────────────────────────────────────────────────────────────────────────────

N_TOPICOS = 6

def modelar_topicos(matriz, vectorizer):
    print(f"\nModelando {N_TOPICOS} tópicos com LDA...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICOS,
        random_state=42,
        max_iter=20,
        learning_method='online',
    )
    lda.fit(matriz)

    features = vectorizer.get_feature_names_out()
    topicos = {}
    print("\n=== TÓPICOS DESCOBERTOS ===")
    for i, comp in enumerate(lda.components_):
        top_words = [features[j] for j in comp.argsort()[-10:][::-1]]
        topicos[i] = top_words
        print(f"Tópico {i+1}: {', '.join(top_words)}")

    return lda, topicos

def plot_topicos(topicos: dict):
    n = len(topicos)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    cores = ['#4A90D9','#E94560','#2ECC71','#F39C12','#9B59B6','#1ABC9C']

    for i, (idx, palavras) in enumerate(topicos.items()):
        ax = axes[i]
        ax.barh(palavras[::-1], range(len(palavras), 0, -1), color=cores[i], alpha=0.85)
        ax.set_title(f'Tópico {idx+1}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel('Relevância relativa')
        ax.xaxis.set_visible(False)

    plt.suptitle('Tópicos Descobertos por LDA', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('text_mining_plots/lda_topicos.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico de tópicos salvo.")

def plot_distribuicao_topicos(df: pd.DataFrame, lda, matriz):
    """Distribuição dos tópicos ao longo do tempo"""
    topico_dominante = lda.transform(matriz).argmax(axis=1)
    df = df.copy()
    df['topico'] = topico_dominante
    df['data'] = pd.to_datetime(df['data'])
    df['mes'] = df['data'].dt.to_period('M')

    pivot = df.groupby(['mes', 'topico']).size().unstack(fill_value=0)
    pivot.index = pivot.index.astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind='area', ax=ax, alpha=0.7,
               colormap='tab10', stacked=True)
    ax.set_title('Distribuição de Tópicos ao Longo do Tempo', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Número de Mensagens')
    ax.legend([f'Tópico {i+1}' for i in range(N_TOPICOS)],
              loc='upper left', fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('text_mining_plots/topicos_tempo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico de tópicos por tempo salvo.")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANÁLISE DE SENTIMENTO (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

MODELO_SENTIMENTO = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

def analisar_sentimento(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nCarregando modelo de sentimento: {MODELO_SENTIMENTO}")
    print("(Primeira execução faz download ~250MB — pode demorar)")
    sentiment_pipe = hf_pipeline(
        "text-classification",
        model=MODELO_SENTIMENTO,
        truncation=True,
        max_length=128,
        batch_size=64,
        device=-1,  # CPU; mude para 0 se tiver GPU
    )

    # Usa o texto limpo (não lematizado) para preservar contexto
    textos = df['texto_limpo'].tolist()

    print(f"Analisando sentimento de {len(textos)} mensagens...")
    resultados = sentiment_pipe(textos)

    df = df.copy()
    df['sentimento'] = [r['label'] for r in resultados]
    df['sentimento_score'] = [r['score'] for r in resultados]

    # Padroniza labels para PT
    mapa = {'positive': 'Positivo', 'negative': 'Negativo', 'neutral': 'Neutro'}
    df['sentimento'] = df['sentimento'].map(mapa).fillna(df['sentimento'])

    print("\n=== DISTRIBUIÇÃO DE SENTIMENTO ===")
    print(df['sentimento'].value_counts())
    return df

def plot_sentimento_geral(df: pd.DataFrame):
    contagem = df['sentimento'].value_counts()
    cores = {'Positivo': '#2ECC71', 'Neutro': '#95A5A6', 'Negativo': '#E94560'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pizza
    axes[0].pie(
        contagem.values,
        labels=contagem.index,
        colors=[cores.get(l, '#999') for l in contagem.index],
        autopct='%1.1f%%',
        startangle=90,
    )
    axes[0].set_title('Distribuição Geral de Sentimento', fontsize=12, fontweight='bold')

    # Barras por usuário (top 8)
    top_users = df['autor'].value_counts().head(8).index
    df_top = df[df['autor'].isin(top_users)]
    pivot = df_top.groupby(['autor', 'sentimento']).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    pivot_pct[['Positivo', 'Neutro', 'Negativo']].plot(
        kind='bar', ax=axes[1], stacked=True,
        color=['#2ECC71', '#95A5A6', '#E94560'],
        alpha=0.85,
    )
    axes[1].set_title('Sentimento por Usuário (Top 8)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('% de mensagens')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig('text_mining_plots/sentimento_geral.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico de sentimento salvo.")

def plot_sentimento_temporal(df: pd.DataFrame):
    df = df.copy()
    df['data'] = pd.to_datetime(df['data'])
    df['mes'] = df['data'].dt.to_period('M')

    pivot = df.groupby(['mes', 'sentimento']).size().unstack(fill_value=0)
    pivot.index = pivot.index.astype(str)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    cores = {'Positivo': '#2ECC71', 'Neutro': '#95A5A6', 'Negativo': '#E94560'}
    for col in ['Positivo', 'Neutro', 'Negativo']:
        if col in pivot_pct.columns:
            ax.plot(pivot_pct.index, pivot_pct[col],
                    marker='o', label=col, color=cores[col], linewidth=2)

    ax.set_title('Evolução do Sentimento ao Longo do Tempo', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mês')
    ax.set_ylabel('% de mensagens')
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('text_mining_plots/sentimento_temporal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico de sentimento temporal salvo.")

def plot_sentimento_vs_engajamento(df: pd.DataFrame):
    """Complemento ao trabalho anterior: sentimento x engajamento"""
    df = df.copy()
    df['engajou'] = (df['reacoes'] > 0).astype(int)

    pivot = df.groupby(['sentimento', 'engajou']).size().unstack(fill_value=0)
    pivot.columns = ['Não Engajou', 'Engajou']
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_pct.plot(kind='bar', ax=ax,
                   color=['#95A5A6', '#4A90D9'], alpha=0.85)
    ax.set_title('Sentimento vs Engajamento', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sentimento')
    ax.set_ylabel('% de mensagens')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.tight_layout()
    plt.savefig('text_mining_plots/sentimento_engajamento.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico sentimento vs engajamento salvo.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ESTATÍSTICAS DO CORPUS
# ─────────────────────────────────────────────────────────────────────────────

def stats_corpus(df: pd.DataFrame):
    print("\n=== ESTATÍSTICAS DO CORPUS ===")
    todos_tokens = " ".join(df['texto_processado']).split()
    freq = Counter(todos_tokens)
    vocab_size = len(freq)
    total_tokens = len(todos_tokens)
    print(f"Total de tokens (após pré-proc): {total_tokens}")
    print(f"Vocabulário único:               {vocab_size}")
    print(f"Média de tokens por mensagem:    {total_tokens/len(df):.1f}")
    print(f"\nTop 20 palavras mais frequentes:")
    for palavra, count in freq.most_common(20):
        print(f"  {palavra:<20} {count}")

    # Plot frequência
    top = freq.most_common(25)
    palavras, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(palavras)[::-1], list(counts)[::-1], color='#4A90D9', alpha=0.85)
    ax.set_title('Top 25 Palavras Mais Frequentes (após pré-processamento)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequência')
    plt.tight_layout()
    plt.savefig('text_mining_plots/frequencia_palavras.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico de frequência salvo.")

def comparar_topicos_engajamento(df: pd.DataFrame, lda, matriz, vectorizer):
    """Compara os tópicos dominantes entre mensagens que engajaram vs não engajaram"""
    df = df.copy()
    df['engajou'] = (df['reacoes'] > 0).astype(int)
    topico_probs = lda.transform(matriz)
    df['topico'] = topico_probs.argmax(axis=1)

    # ── Gráfico 1: distribuição de tópicos por classe ──────────────────────
    pivot = df.groupby(['topico', 'engajou']).size().unstack(fill_value=0)
    pivot.columns = ['Não Engajou', 'Engajou']
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.index = [f'Tópico {i+1}' for i in pivot_pct.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot_pct.plot(kind='bar', ax=axes[0],
                   color=['#95A5A6', '#4A90D9'], alpha=0.85)
    axes[0].set_title('% de Engajamento por Tópico', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('% de mensagens')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())
    axes[0].legend(loc='upper right')

    # ── Gráfico 2: top palavras das mensagens engajadas vs não engajadas ───
    features = vectorizer.get_feature_names_out()

    for idx, (label, classe) in enumerate([(0, 'Não Engajou'), (1, 'Engajou')]):
        subset = df[df['engajou'] == idx]['texto_processado']
        if len(subset) == 0:
            continue
        vec_sub = vectorizer.transform(subset)
        media = np.asarray(vec_sub.mean(axis=0)).flatten()
        top_idx = media.argsort()[-10:][::-1]
        top_words = [features[i] for i in top_idx]
        top_scores = [media[i] for i in top_idx]

        ax = axes[1]
        offset = idx * 0.35
        posicoes = np.arange(len(top_words)) + offset
        cor = '#95A5A6' if idx == 0 else '#4A90D9'
        ax.bar(posicoes, top_scores, width=0.35, label=classe, color=cor, alpha=0.85)

    axes[1].set_title('Top 10 Termos TF-IDF\nEngajou vs Não Engajou', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('TF-IDF Médio')
    axes[1].set_xticks(np.arange(10) + 0.175)

    # Pega os termos da classe engajou para o eixo x
    subset_eng = df[df['engajou'] == 1]['texto_processado']
    if len(subset_eng) > 0:
        vec_eng = vectorizer.transform(subset_eng)
        media_eng = np.asarray(vec_eng.mean(axis=0)).flatten()
        top_eng = [features[i] for i in media_eng.argsort()[-10:][::-1]]
        axes[1].set_xticklabels(top_eng, rotation=30, ha='right', fontsize=8)
    axes[1].legend()

    plt.suptitle('Análise de Tópicos: Mensagens que Engajaram vs Não Engajaram',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('text_mining_plots/topicos_engajamento.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico tópicos vs engajamento salvo.")

    # ── Estatísticas no terminal ───────────────────────────────────────────
    print("\n=== TÓPICOS DAS MENSAGENS QUE ENGAJARAM ===")
    engajadas = df[df['engajou'] == 1]['topico'].value_counts()
    for topico, count in engajadas.items():
        pct = count / len(df[df['engajou'] == 1]) * 100
        print(f"  Tópico {topico+1}: {count} mensagens ({pct:.1f}%)")

if __name__ == "__main__":
    print("=" * 55)
    print("  PIPELINE DE MINERAÇÃO DE TEXTO — DISCORD")
    print("=" * 55)

    # 1. Carrega
    df = load_data()

    # 2. Filtra bots e mensagens inválidas
    df = filtrar_mensagens(df)

    # 3. Pré-processamento
    df = preprocessar(df)

    # 4. Estatísticas do corpus
    stats_corpus(df)

    # 5. TF-IDF
    matriz_tfidf, vectorizer = calcular_tfidf(df)
    plot_top_tfidf(vectorizer, matriz_tfidf)

    # 6. LDA — Modelagem de tópicos
    lda, topicos = modelar_topicos(matriz_tfidf, vectorizer)
    plot_topicos(topicos)
    df = plot_distribuicao_topicos(df, lda, matriz_tfidf)
    
    # 7. Comparação tópicos: engajou vs não engajou
    comparar_topicos_engajamento(df, lda, matriz_tfidf, vectorizer)

    # 8. Sentimento
    df = analisar_sentimento(df)
    plot_sentimento_geral(df)
    plot_sentimento_temporal(df)
    plot_sentimento_vs_engajamento(df)

    # Salva CSV enriquecido
    df.to_csv("data/discord_text_mining.csv", index=False)

    print("\n" + "=" * 55)
    print("  PIPELINE CONCLUÍDO")
    print("=" * 55)
    print("\nArquivos gerados em text_mining_plots/:")
    for f in sorted(os.listdir("text_mining_plots")):
        print(f"  - {f}")
    print("\nCSV enriquecido: data/discord_text_mining.csv")
