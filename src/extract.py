import pandas as pd
import requests
from pymongo import MongoClient
from sqlalchemy import create_engine

from src import config


class Extract:
    """Lê os dados: da API da Prefeitura, da BRONZE (MongoDB Atlas) ou da SILVER (NeonDB)."""

    def ler_api(self):
        registros = []
        offset = 0

        # a API devolve no máximo 500 registros por vez, então pedimos página por página
        while len(registros) < config.LIMITE_REGISTROS:
            params = {"resource_id": config.RESOURCE_ID, "limit": 500, "offset": offset}
            resposta = requests.get(config.URL_API, params=params)
            pagina = resposta.json()["result"]["records"]

            if len(pagina) == 0:
                break

            registros.extend(pagina)
            offset += 500
            print(f"[extract] {len(registros)} registros baixados da API")

        return registros

    def ler_bronze(self):
        cliente = MongoClient(config.MONGO_URI)
        colecao = cliente[config.MONGO_BANCO][config.MONGO_COLECAO]
        documentos = list(colecao.find({}, {"_id": 0, "_data_carga": 0}))  # sem as colunas técnicas
        cliente.close()

        print(f"[extract] {len(documentos)} documentos lidos da bronze")
        return pd.DataFrame(documentos)

    def ler_silver(self):
        engine = create_engine(config.NEON_URL)
        df = pd.read_sql(f"SELECT * FROM {config.TABELA_SILVER}", engine)
        engine.dispose()

        print(f"[extract] {len(df)} linhas lidas da silver")
        return df

    def ler_perfis(self, arquivo="perfis_novos.csv"):
        df = pd.read_csv(arquivo)
        print(f"[extract] {len(df)} perfis novos lidos de {arquivo}")
        return df