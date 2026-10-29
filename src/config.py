import os

from dotenv import load_dotenv

load_dotenv()  # lê as variáveis do arquivo .env

# Fonte: https://dados.recife.pe.gov.br/dataset/perfil-das-pessoas-vacinadas-covid-19
URL_API = "https://dados.recife.pe.gov.br/api/3/action/datastore_search"
RESOURCE_ID = "164e7f11-2855-421b-89ea-64ea485ae09d"
LIMITE_REGISTROS = 5000  # não alterar: todos os grupos usam a mesma amostra

# Bronze - MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI")
MONGO_BANCO = "vacinacao"
MONGO_COLECAO = "bronze_vacinados"

# Silver - NeonDB
NEON_URL = os.getenv("NEON_URL")
TABELA_SILVER = "silver_vacinados"