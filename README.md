# Discord API — Coleta e Análise de Dados

## Configuração

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
TOKEN=seu_token_do_bot
SERVER_ID=id_do_servidor
CANAL_ID=id_do_canal
```

## Dependências

Certifique-se de ter o [direnv](https://direnv.net/) instalado para carregar as variáveis de ambiente automaticamente.

Em seguida, instale as dependências do projeto com (tenha certeza de ter instalado o uv):

```bash
uv sync
```

## Execução

```bash
python main.py
```
