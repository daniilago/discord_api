import pandas as pd
import re

def count_emojis(text):
    """Count emoji characters in text"""
    if pd.isna(text):
        return 0
    # Simple emoji pattern - matches common emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return len(emoji_pattern.findall(str(text)))

def top_posters(df, n=10):
    """Who sends the most messages"""
    counts = df['autor'].value_counts().head(n)
    return counts

def top_emoji_users(df, n=10):
    """Who uses the most emojis (total and per message)"""
    df['emoji_count'] = df['conteudo'].apply(count_emojis)
    total_emojis = df.groupby('autor')['emoji_count'].sum().sort_values(ascending=False).head(n)
    avg_emojis = df.groupby('autor')['emoji_count'].mean().sort_values(ascending=False).head(n)
    return total_emojis, avg_emojis

def longest_messages(df, n=10):
    """Who writes the longest messages (by average length)"""
    if 'message_length' in df.columns:
        avg_length = df.groupby('autor')['message_length'].mean().sort_values(ascending=False).head(n)
    else:
        # Compute message length from content
        df_temp = df.copy()
        df_temp['message_length'] = df_temp['conteudo'].astype(str).str.len()
        avg_length = df_temp.groupby('autor')['message_length'].mean().sort_values(ascending=False).head(n)
    return avg_length

def most_mentioned_users(df, n=10):
    """Who is mentioned the most in messages"""
    all_mentions = []
    for mentions_str in df['mentioned_users'].dropna():
        if mentions_str:
            all_mentions.extend(mentions_str.split(','))

    if all_mentions:
        mention_counts = pd.Series(all_mentions).value_counts().head(n)
    else:
        mention_counts = pd.Series(dtype=int)
    return mention_counts

def fastest_repliers(df, n=10):
    """Who replies the fastest (average response time in minutes)"""
    # Convert data to datetime - handle various formats
    if df['data'].dtype == 'object':
        df['data'] = pd.to_datetime(df['data'], format='mixed')
    df_sorted = df.sort_values('data')

    # Build a map of message ID to author and timestamp
    msg_map = df_sorted.set_index('id')[['autor', 'data']].to_dict('index')

    # Track reply times
    reply_times = []

    for _, row in df_sorted.iterrows():
        if pd.notna(row['reply_to_id']) and row['reply_to_id'] in msg_map:
            original_msg = msg_map[row['reply_to_id']]
            time_diff = (row['data'] - original_msg['data']).total_seconds() / 60
            reply_times.append({
                'replier': row['autor'],
                'reply_time_minutes': time_diff
            })

    if reply_times:
        reply_df = pd.DataFrame(reply_times)
        avg_reply_time = reply_df.groupby('replier')['reply_time_minutes'].mean().sort_values().head(n)
        reply_count = reply_df['replier'].value_counts().head(n)
        return avg_reply_time, reply_count
    else:
        return pd.Series(dtype=float), pd.Series(dtype=int)

def print_all_filters():
    """Run all filter analyses"""
    df = pd.read_csv("discord_data.csv")

    print("=== DISCORD ACTIVITY FILTERS ===\n")

    # 1. Most Active (by message count)
    print("1. WHO TALKS MOST (by message count):")
    print(top_posters(df, n=10))
    print()

    # 2. Emoji Users
    print("2. WHO USES MORE EMOJIS (total emojis):")
    total_emojis, avg_emojis = top_emoji_users(df, n=10)
    print(total_emojis)
    print("\n   (avg emojis per message):")
    print(avg_emojis)
    print()

    # 3. Longest Messages
    print("3. WHO WRITES THE LONGEST MESSAGES (avg characters):")
    print(longest_messages(df, n=10))
    print()

    # 4. Most Mentioned
    print("4. WHO IS MORE MENTIONED:")
    if 'mentioned_users' in df.columns:
        mentioned = most_mentioned_users(df, n=10)
        if len(mentioned) > 0:
            print(mentioned)
        else:
            print("(No mentions in dataset)")
    else:
        print("(Column 'mentioned_users' missing - re-run collect.py with updated code)")
    print()

    # 5. Reply Speed
    # print("5. WHO REPLIES FASTEST (avg minutes):")
    # if 'reply_to_id' in df.columns:
    #     avg_time, count = fastest_repliers(df, n=10)
    #     if len(avg_time) > 0:
    #         print(avg_time)
    #         print("\n   (number of replies):")
    #         print(count)
    #     else:
    #         print("(No replies in dataset)")
    # else:
    #     print("(Column 'reply_to_id' missing - re-run collect.py with updated code)")
    # print()

    # 6. Overall activity summary
    print("6. ACTIVITY SUMMARY:")
    print(f"Total messages: {len(df)}")
    print(f"Unique authors: {df['autor'].nunique()}")
    if 'data' in df.columns:
        print(f"Time period: {df['data'].min()} to {df['data'].max()}")
    print(f"Messages with reactions: {(df['reacoes'] > 0).sum()}")
    print(f"Messages with attachments: {df['tem_anexo'].sum()}")
    if 'menciona_alguem' in df.columns:
        print(f"Messages that mention someone: {df['menciona_alguem'].sum()}")

