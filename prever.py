from src.extract import Extract
from src.load import Load
from src.ml_model import Classificador

extract = Extract()
load = Load()
classificador = Classificador.carregar()  # nunca treina: só usa o modelo salvo pelo train.py

# 1) conferência: o modelo carregado tem que acertar o mesmo que o recém-treinado
df = extract.ler_silver()
X, y = classificador.preparar(df)
_, X_teste, _, y_teste = classificador.dividir(X, y)
classificador.avaliar(X_teste, y_teste)

# 2) perfis novos -> previsão -> GOLD
perfis = extract.ler_perfis()
previsoes = classificador.prever(perfis)
print(previsoes[["perfil", "vacina_prevista", "confianca"]])
load.salvar_gold(previsoes)