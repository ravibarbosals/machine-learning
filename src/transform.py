import pandas as pd


class Transform:
    """Limpa e padroniza os dados da BRONZE para a SILVER."""

    def limpar(self, df):
        df = self.limpar_textos(df)
        df = self.converter_tipos(df)
        df = self.criar_colunas(df)
        df = self.remover_invalidos(df)
        print(f"[transform] {len(df)} linhas prontas para a silver")
        return df

    def limpar_textos(self, df):
        colunas_texto = ["faixa_etaria", "sexo", "raca_cor", "municipio", "grupo",
                         "categoria", "lote", "vacina_fabricante", "cnes", "sistema_origem"]
        for coluna in colunas_texto:
            df[coluna] = df[coluna].str.strip().str.upper()
            df[coluna] = df[coluna].replace("", None)  # texto vazio vira nulo
        return df

    def converter_tipos(self, df):
        df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
        df["dose"] = pd.to_numeric(df["descricao_dose"], errors="coerce")
        df["data_vacinacao"] = pd.to_datetime(df["data_vacinacao"], errors="coerce").dt.date
        return df

    def criar_colunas(self, df):
        # "3 - COMIRNATY (PFIZER)" -> "COMIRNATY (PFIZER)"
        df["vacina"] = df["vacina_fabricante"].str.split(" - ", n=1).str[1]

        # "DS 4: CNES: 0000639 - UBS JOAQUIM CAVALCANTE" -> "DS 4" e "UBS JOAQUIM CAVALCANTE"
        df["distrito_sanitario"] = df["cnes"].str.split(":").str[0]
        df["unidade_saude"] = df["cnes"].str.split(" - ", n=1).str[1]

        return df.drop(columns=["vacina_fabricante", "cnes", "descricao_dose"])

    def remover_invalidos(self, df):
        df = df.dropna(subset=["idade", "data_vacinacao"])
        return df.drop_duplicates()