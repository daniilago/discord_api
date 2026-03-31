import pandas as pd

def analise():
    df = pd.read_csv("discord_data.csv")  

    print("=== O QUE HÁ NA BASE ===")
    print(df.describe())
    print(f"\nTotal de mensagens: {len(df)}")
    print(f"Autores únicos: {df['autor'].nunique()}")
    print(f"Período: {df['data'].min()} até {df['data'].max()}")

    print("\n=== MAIS ATIVOS ===")
    print(df['autor'].value_counts().head(10))

    print("\n=== ENGAJAMENTO ===")
    print(f"Mensagens com reações: {(df['reacoes'] > 0).sum()}")
    print(f"Mensagens com anexos: {df['tem_anexo'].sum()}")