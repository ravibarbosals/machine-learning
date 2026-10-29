import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

class Classificador:
    """Prevê qual vacina a pessoa recebeu a partir do perfil (dados da SILVER)."""

    ALVO = "vacina"
    NUMERICAS = ["idade", "dose", "dia_ano"]
    CATEGORICAS = ["sexo", "raca_cor", "distrito_sanitario"]
    GRADE = {
        "max_depth": [4, 6, 8, 10, 12, 15, None],
        "min_samples_leaf": [1, 3, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    }

    def __init__(self, profundidade=4, min_registros=50):
        self.min_registros = min_registros
        self.modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42)

    def preparar(self, df):
        df = df.copy()
        data = pd.to_datetime(df["data_vacinacao"])
        df["dia_ano"] = data.dt.dayofyear
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

    def validar(self, X_treino, y_treino, folds=5):
        # mede modelo treino, o teste fica guardado pro final
        notas = cross_val_score(self.modelo, X_treino, y_treino, cv=folds, scoring="accuracy")
        print(f"[model] acuracia na validação: {notas.mean():.1%} (+- {notas.std():.1%})")
        return notas.mean()

    def buscar_parametros(self, X_treino, y_treino, folds=5):
            # testa todas as combinações da GRADE com validação cruzada, só no treino
            busca = GridSearchCV(self.modelo, self.GRADE, cv=folds, scoring="accuracy",
                                return_train_score=True)
            busca.fit(X_treino, y_treino)
            self.modelo = busca.best_estimator_

            melhor = busca.best_index_
            treino = busca.cv_results_["mean_train_score"][melhor]
            print(f"[model] melhores parâmetros: {busca.best_params_}")
            print(f"[model] acurácia no treino: {treino:.1%} | na validação: {busca.best_score_:.1%}")
            return busca.best_params_

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