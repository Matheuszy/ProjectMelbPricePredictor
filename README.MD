# Melbourne Housing Price Predictor

Este projeto utiliza Machine Learning para prever preços de imóveis em Melbourne, Austrália, baseando-se em características como número de quartos, casas de banho e tamanho do terreno.

## 🚀 Funcionalidades
* Processamento automático de dados (limpeza de valores nulos).
* Seleção de características (Features) baseada em correlação.
* Treino de modelo de Regressão com Árvore de Decisão (`DecisionTreeRegressor`).
* Avaliação de métricas de erro (MSE e R²).

## 🛠️ Tecnologias Utilizadas
* **Python 3.12**
* **Pandas**: Manipulação de dados.
* **Scikit-Learn**: Modelagem de Machine Learning.
* **Matplotlib/Seaborn**: Visualização de dados.

## 📋 Como Executar

**Clonar o projeto:**
   ```bash ```
   git clone <url-do-repositorio>

Instalar dependências: Certifique-se de que o ficheiro requirements.txt está na raiz do projeto e execute:


pip install -r requirements.txt
Executar o preditor: Execute o script principal:


python MelbPricePredictor.py
📊 Estrutura do Projeto
MelbPricePredictor.py: Código principal em Orientação a Objetos.

melb_data.csv: Base de dados (deve estar na raiz).

requirements.txt: Lista de bibliotecas necessárias.