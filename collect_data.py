import discord
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TOKEN")
CANAL_ID = int(os.getenv("CANAL_ID"))

def collect():
    import pandas as pd

    mensagens = []
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Bot conectado como {client.user}")
        canal = client.get_channel(CANAL_ID)

        async for msg in canal.history(limit=1000):
            mensagens.append({
                "id": msg.id,
                "autor": str(msg.author),
                "conteudo": msg.content,
                "data": msg.created_at,
                "reacoes": sum(r.count for r in msg.reactions),
                "tem_anexo": len(msg.attachments) > 0,
                "menciona_alguem": len(msg.mentions) > 0,
            })

        df = pd.DataFrame(mensagens)
        df.to_csv("discord_data.csv", index=False)
        print(f"{len(df)} mensagens salvas!")
        await client.close()

    client.run(TOKEN)  