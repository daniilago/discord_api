import discord
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

TOKEN = os.getenv("TOKEN")
CANAL_ID = int(os.getenv("CANAL_ID"))

def collect():
    mensagens = []
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Bot conectado como {client.user}")
        canal = client.get_channel(CANAL_ID)

        async for msg in canal.history(limit=500):
            # Get mentioned usernames
            mentioned_users = [str(user) for user in msg.mentions]

            # Get reply-to message ID if this message is a reply
            reply_to_id = None
            if msg.reference and msg.reference.message_id:
                reply_to_id = msg.reference.message_id

            mensagens.append({
                "id": msg.id,
                "autor": str(msg.author),
                "conteudo": msg.content,
                "data": msg.created_at,
                "reacoes": sum(r.count for r in msg.reactions),
                "tem_anexo": len(msg.attachments) > 0,
                "menciona_alguem": len(msg.mentions) > 0,
                "mentioned_users": ",".join(mentioned_users) if mentioned_users else "",
                "reply_to_id": reply_to_id,
                "message_length": len(msg.content),
            })

        df = pd.DataFrame(mensagens)
        df.to_csv("discord_data.csv", index=False)
        print(f"{len(df)} mensagens salvas!")
        await client.close()

    client.run(TOKEN)  