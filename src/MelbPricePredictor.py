import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

class MelbPricePredictor:
    def __init__(self, file_path, random_state=42):
        self.file_path = file_path
        self.random_state = random_state
        self.base = None
        self.X = None
        self.Y = None
        self.model = DecisionTreeRegressor(random_state=self.random_state)
        self.x_treino, self.x_teste, self.y_treino, self.y_teste = [None] * 4

    def carregar_e_limpar_dados(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"O arquivo {self.file_path} não foi encontrado.")

        self.base = pd.read_csv(self.file_path)

        colunas_remover = [c for c in ["BuildingArea", "YearBuilt"] if c in self.base.columns]
        self.base = self.base.drop(colunas_remover, axis=1)

        colunas_selecionadas = ["Price", "Rooms", "Bathroom", "Bedroom2", "Car", "Landsize"]
        if not all(col in self.base.columns for col in colunas_selecionadas):
            missing = [c for c in colunas_selecionadas if c not in self.base.columns]
            raise ValueError(f"Colunas ausentes no dataset: {missing}")

        dados_limpos = self.base[colunas_selecionadas].dropna(axis=0)

        self.Y = dados_limpos.Price
        self.X = dados_limpos.drop('Price', axis=1)

    def preparar_treino(self, test_size=0.25):
        if self.X is None or self.Y is None:
            raise RuntimeError("Os dados precisam ser carregados antes do treino.")

        self.x_treino, self.x_teste, self.y_treino, self.y_teste = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=self.random_state
        )

    def treinar(self):
        if self.x_treino is None:
            raise RuntimeError("Os dados de treino não foram preparados.")
        self.model.fit(self.x_treino, self.y_treino)

    def avaliar(self):
        if self.x_teste is None:
            raise RuntimeError("Não existem dados de teste para avaliação.")

        previsoes = self.model.predict(self.x_teste)
        mse = mean_squared_error(self.y_teste, previsoes)
        r2 = r2_score(self.y_teste, previsoes)
        return {"MSE": mse, "R2": r2}

if __name__ == "__main__":
    try:
        predictor = MelbPricePredictor("melb_data.csv")
        predictor.carregar_e_limpar_dados()
        predictor.preparar_treino()
        predictor.treinar()
        resultados = predictor.avaliar()

        print(f"MSE: {resultados['MSE']:,.2f}")
        print(f"R2: {resultados['R2']:.4f}")

    except Exception as e:
        print(f"Erro na execução: {e}")