import pandas as pd
import re

def count_emojis(text):
    if pd.isna(text):
        return 0
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return len(emoji_pattern.findall(str(text)))

def top_posters(df, n=10):
    return df['autor'].value_counts().head(n)

def top_emoji_users(df, n=10):
    df = df.copy()
    df['emoji_count'] = df['conteudo'].apply(count_emojis)
    total_emojis = df.groupby('autor')['emoji_count'].sum().sort_values(ascending=False).head(n)
    avg_emojis = df.groupby('autor')['emoji_count'].mean().sort_values(ascending=False).head(n)
    return total_emojis, avg_emojis

def longest_messages(df, n=10):
    if 'message_length' in df.columns:
        avg_length = df.groupby('autor')['message_length'].mean().sort_values(ascending=False).head(n)
    else:
        df_temp = df.copy()
        df_temp['message_length'] = df_temp['conteudo'].astype(str).str.len()
        avg_length = df_temp.groupby('autor')['message_length'].mean().sort_values(ascending=False).head(n)
    return avg_length

def most_mentioned_users(df, n=10):
    all_mentions = []
    for mentions_str in df['mentioned_users'].dropna():
        if mentions_str:
            all_mentions.extend(mentions_str.split(','))
    if all_mentions:
        return pd.Series(all_mentions).value_counts().head(n)
    return pd.Series(dtype=int)

def print_all_filters(server_name: str, channel_name: str):
    df = pd.read_csv(f"dados/channel_history/{server_name}/{channel_name}.csv")

    print("=== HISTÓRICO GERAL DO CANAL ===\n")

    print("1. QUEM MAIS FALA (por quantidade de mensagens):")
    print(top_posters(df, n=10))
    print()

    print("2. QUEM MAIS USA EMOJIS (total de emojis):")
    total_emojis, avg_emojis = top_emoji_users(df, n=10)
    print(total_emojis)
    print("\n   (média de emojis por mensagem):")
    print(avg_emojis)
    print()

    print("3. QUEM ESCREVE AS MENSAGENS MAIS LONGAS (média de caracteres):")
    print(longest_messages(df, n=10))
    print()

    print("4. QUEM É MAIS MENCIONADO:")
    if 'mentioned_users' in df.columns:
        mentioned = most_mentioned_users(df, n=10)
        if len(mentioned) > 0:
            print(mentioned)
        else:
            print("(Nenhuma menção encontrada)")
    else:
        print("(Coluna 'mentioned_users' não encontrada - rode o collect novamente)")
    print()

    print("5. RESUMO GERAL:")
    print(f"Total de mensagens: {len(df)}")
    print(f"Usuários únicos: {df['autor'].nunique()}")
    if 'data' in df.columns:
        print(f"Período: {df['data'].min()} até {df['data'].max()}")
    print(f"Mensagens com reações: {(df['reacoes'] > 0).sum()}")
    print(f"Mensagens com anexos: {df['tem_anexo'].sum()}")
    if 'menciona_alguem' in df.columns:
        print(f"Mensagens que mencionam alguém: {df['menciona_alguem'].sum()}")

def analyse_user(username: str, server_name: str, channel_name: str):
    df = pd.read_csv(f"dados/user_history/{server_name}/{channel_name}/{username}.csv")

    df_user = df[df['autor'] == username].copy()

    if df_user.empty:
        print(f"Usuário '{username}' não encontrado no dataset.")
        print("Usuários disponíveis:")
        print(df['autor'].unique())
        return

    print(f"=== ANÁLISE DO USUÁRIO: {username} ===\n")
    print(f"Total de mensagens: {len(df_user)}")
    print(f"Mensagens com reações: {(df_user['reacoes'] > 0).sum()}")
    print(f"Mensagens com anexos: {df_user['tem_anexo'].sum()}")
    print(f"Mensagens que mencionam alguém: {df_user['menciona_alguem'].sum()}")

    df_user['message_length'] = df_user['conteudo'].astype(str).str.len()
    print(f"Tamanho médio das mensagens: {df_user['message_length'].mean():.1f} caracteres")

    df_user['emoji_count'] = df_user['conteudo'].apply(count_emojis)
    print(f"Total de emojis usados: {df_user['emoji_count'].sum()}")
    print(f"Média de emojis por mensagem: {df_user['emoji_count'].mean():.2f}")

    all_mentions = []
    for mentions_str in df_user['mentioned_users'].dropna():
        if mentions_str:
            all_mentions.extend(mentions_str.split(','))
    if all_mentions:
        print(f"\nQuem {username} mais menciona:")
        print(pd.Series(all_mentions).value_counts().head(5))

    df_user['data'] = pd.to_datetime(df_user['data'])
    df_user['hora'] = df_user['data'].dt.hour
    print(f"\nHorários mais ativos (horário de Brasília):")
    print(df_user['hora'].value_counts().head(5))