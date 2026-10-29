from datetime import datetime

from pymongo import MongoClient
from sqlalchemy import create_engine

from src import config


class Load:
    """Grava os dados: na camada BRONZE (MongoDB Atlas) ou na SILVER (NeonDB)."""

    def salvar_bronze(self, registros):
        for registro in registros:
            registro["_data_carga"] = datetime.now()

        cliente = MongoClient(config.MONGO_URI)
        colecao = cliente[config.MONGO_BANCO][config.MONGO_COLECAO]
        colecao.delete_many({})  # apaga a carga anterior para não duplicar
        colecao.insert_many(registros)
        cliente.close()

        print(f"[load] {len(registros)} documentos gravados na bronze")

    def salvar_silver(self, df):
        engine = create_engine(config.NEON_URL)
        df.to_sql(config.TABELA_SILVER, engine, if_exists="replace", index=False)
        print(f"[load] {len(df)} linhas gravadas na tabela '{config.TABELA_SILVER}'")