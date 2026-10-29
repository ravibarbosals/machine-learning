import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


class Classificador:
    """Prevê qual vacina a pessoa recebeu a partir do perfil (dados da SILVER)."""

    ALVO = "vacina"
    NUMERICAS = ["idade", "dose", "ano", "mes"]
    CATEGORICAS = ["sexo", "raca_cor", "distrito_sanitario"]

    def __init__(self, profundidade=4, min_registros=50):
        self.min_registros = min_registros
        self.modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42)

    def preparar(self, df):
        df = df.copy()
        data = pd.to_datetime(df["data_vacinacao"])
        df["ano"] = data.dt.year
        df["mes"] = data.dt.month
        df = df.dropna(subset=self.NUMERICAS + [self.ALVO])

        # remove vacinas raras (poucos exemplos para aprender)
        contagem = df[self.ALVO].value_counts()
        manter = contagem[contagem >= self.min_registros].index
        df = df[df[self.ALVO].isin(manter)]

        colunas = self.NUMERICAS + self.CATEGORICAS
        X = pd.get_dummies(df[colunas], columns=self.CATEGORICAS, dtype=int)
        y = df[self.ALVO]
        return X, y

    def dividir(self, X, y):
        # não alterar: todos os grupos usam a mesma divisão treino/teste
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def linha_de_base(self, X_treino, y_treino, X_teste, y_teste):
        chute = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
        acuracia = accuracy_score(y_teste, chute.predict(X_teste))
        print(f"[model] linha de base: {acuracia:.1%}")

    def treinar(self, X_treino, y_treino):
        self.modelo.fit(X_treino, y_treino)
        self.colunas = list(X_treino.columns)

    def avaliar(self, X_teste, y_teste):
        previsto = self.modelo.predict(X_teste)
        print(f"[model] acurácia no teste: {accuracy_score(y_teste, previsto):.1%}")
        print(classification_report(y_teste, previsto, zero_division=0))
        return previsto

    def explicar(self):
        importancia = pd.Series(self.modelo.feature_importances_, index=self.colunas)
        print("[model] colunas mais importantes:")
        print(importancia.sort_values(ascending=False).head(5).round(3))
        print(export_text(self.modelo, feature_names=self.colunas, max_depth=3))

    def matriz_confusao(self, X_teste, y_teste, arquivo="matriz_confusao.png"):
        ConfusionMatrixDisplay.from_estimator(self.modelo, X_teste, y_teste, xticks_rotation=45, cmap="Oranges")
        plt.tight_layout()
        plt.savefig(arquivo, dpi=150)
        print(f"[model] matriz de confusão salva em {arquivo}")