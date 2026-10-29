from src.extract import Extract
from src.ml_model import Classificador

extract = Extract()
classificador = Classificador(profundidade=4)

# SILVER (NeonDB) -> modelo
df = extract.ler_silver()
X, y = classificador.preparar(df)
X_treino, X_teste, y_treino, y_teste = classificador.dividir(X, y)

classificador.linha_de_base(X_treino, y_treino, X_teste, y_teste)
classificador.buscar_parametros(X_treino, y_treino)
classificador.treinar(X_treino, y_treino)
classificador.avaliar(X_teste, y_teste)
classificador.explicar()
classificador.matriz_confusao(X_teste, y_teste)