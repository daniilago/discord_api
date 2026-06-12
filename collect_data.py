import discord
import os
from dotenv import load_dotenv
from datetime import timezone, timedelta
import pandas as pd
from config import root

BRASILIA = timezone(timedelta(hours=-3))

def format_data(dt):
    if dt is None:
        return ""
    return dt.astimezone(BRASILIA).strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(f"{root}/channel_history", exist_ok=True)
os.makedirs(f"{root}/user_history", exist_ok=True)
os.makedirs(f"{root}/server_infos", exist_ok=True)

load_dotenv()

TOKEN = os.getenv("TOKEN")
CANAL_ID = int(os.getenv("CANAL_ID"))

def describe_content(msg):
    if msg.content:
        return msg.content
    if msg.attachments:
        descricoes = []
        for a in msg.attachments:
            if a.content_type and a.content_type.startswith("image/"):
                descricoes.append(f"[IMAGEM: {a.filename}]")
            elif a.content_type and a.content_type.startswith("video/"):
                descricoes.append(f"[VIDEO: {a.filename}]")
            else:
                descricoes.append(f"[ANEXO: {a.filename}]")
        return " ".join(descricoes)
    return ""

def collect(username: str = None):
    mensagens = []
    meta = {}
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Bot conectado como {client.user}")
        guild = client.get_guild(int(os.getenv("SERVER_ID")))
        canal = client.get_channel(CANAL_ID)
        meta['server'] = guild.name
        meta['channel'] = canal.name
        limite_history = None if username else 10000

        async for msg in canal.history(limit=limite_history):
            if username and str(msg.author) != username:
                continue
            if not username and (not msg.content or len(msg.content.strip()) < 8):
                continue

            mentioned_users = [str(user) for user in msg.mentions]
            reply_to_id = msg.reference.message_id if msg.reference and msg.reference.message_id else None

            mensagens.append({
                "id": msg.id,
                "autor": str(msg.author),
                "conteudo": describe_content(msg),
                f"{root}": format_data(msg.created_at),
                "reacoes": sum(r.count for r in msg.reactions),
                "tem_anexo": len(msg.attachments) > 0,
                "menciona_alguem": len(msg.mentions) > 0,
                "mentioned_users": ",".join(mentioned_users) if mentioned_users else "",
                "reply_to_id": reply_to_id,
                "message_length": len(msg.content) if msg.content else 0,
            })

            if username and len(mensagens) >= 500:
                break
            if not username and len(mensagens) >= 10000:
                break
        df = pd.DataFrame(mensagens)

        if username:
            user_folder = f"{root}/user_history/{guild.name}/{canal.name}"
            os.makedirs(user_folder, exist_ok=True)
            df.to_csv(f"{user_folder}/{username}.csv", index=False)
        else:
            server_folder = f"{root}/channel_history/{guild.name}"
            os.makedirs(server_folder, exist_ok=True)
            df.to_csv(f"{server_folder}/{canal.name}.csv", index=False)

        print(f"{len(df)} mensagens salvas!")
        await client.close()

    client.run(TOKEN)
    return meta.get('server'), meta.get('channel')

def collect_server_info():
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True  # necessário para pegar membros
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        guild = client.get_guild(int(os.getenv("SERVER_ID")))
        # Cria pasta com o nome do servidor
        server_folder = f"{root}/server_infos/{guild.name}"
        os.makedirs(server_folder, exist_ok=True)
        print(f"Coletando informações do servidor: {guild.name}")

        # --- Informações gerais do servidor ---
        server_info = {
            "nome": guild.name,
            "id": guild.id,
            "dono": str(guild.owner),
            "dono_id": guild.owner_id,
            "criado_em": format_data(guild.created_at),
            "total_membros": guild.member_count,
            "total_canais": len(guild.channels),
            "total_cargos": len(guild.roles),
            "descricao": guild.description or "",
        }
        pd.DataFrame([server_info]).to_csv(f"{server_folder}/info.csv", index=False)
        print("Informações gerais salvas!")

        # --- Membros ---
        membros = []
        for member in guild.members:
            membros.append({
                "id": member.id,
                "nome": str(member.name),
                "display_name": member.display_name,
                "e_bot": member.bot,
                "entrou_em": format_data(member.joined_at),
                "conta_criada_em": format_data(member.created_at),
                "cargos": ",".join([r.name for r in member.roles if r.name != "@everyone"]),
            })
        pd.DataFrame(membros).to_csv(f"{server_folder}/members.csv", index=False)
        print(f"{len(membros)} membros salvos!")

        # --- Canais ---
        canais = []
        for channel in guild.channels:
            canais.append({
                "id": channel.id,
                "nome": channel.name,
                "tipo": str(channel.type),
                "categoria": channel.category.name if channel.category else "",
            })
        pd.DataFrame(canais).to_csv(f"{server_folder}/channels.csv", index=False)
        print(f"{len(canais)} canais salvos!")

        # --- Cargos/Roles ---
        cargos = []
        for role in guild.roles:
            if role.name == "@everyone":
                continue
            cargos.append({
                "id": role.id,
                "nome": role.name,
                "cor": str(role.colour),
                "total_membros": len(role.members),
                "e_admin": role.permissions.administrator,
                "criado_em": format_data(role.created_at),
            })
        pd.DataFrame(cargos).to_csv(f"{server_folder}/roles.csv", index=False)
        print(f"{len(cargos)} cargos salvos!")

        await client.close()

    client.run(TOKEN)