import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

IMAGES_DIR = "images"


def salvar_grafico(nome):

    os.makedirs(IMAGES_DIR, exist_ok=True)

    caminho = os.path.join(IMAGES_DIR, nome)

    plt.savefig(caminho, dpi=300, bbox_inches="tight")

    print(f"Gráfico salvo em: {caminho}")


# ------------------------------------------------
# REAL VS PREVISTO
# ------------------------------------------------

def grafico_real_vs_previsto(preco_real, preco_previsto):

    plt.figure(figsize=(8,8))

    sns.scatterplot(x=preco_real, y=preco_previsto, alpha=0.4)

    plt.plot(
        [preco_real.min(), preco_real.max()],
        [preco_real.min(), preco_real.max()],
        color="red"
    )

    plt.xlabel("Preço Real (R$)")
    plt.ylabel("Preço Previsto (R$)")

    plt.title("Preço Real vs Preço Previsto")

    salvar_grafico("preco_real_vs_previsto.png")

    plt.show()


# ------------------------------------------------
# DISTRIBUIÇÃO DO ERRO
# ------------------------------------------------

def distribuicao_erro(preco_real, preco_previsto):

    erro = preco_previsto - preco_real

    plt.figure(figsize=(10,6))

    sns.histplot(erro, bins=40)

    plt.title("Distribuição do Erro do Modelo")

    plt.xlabel("Erro (R$)")
    plt.ylabel("Quantidade")

    salvar_grafico("distribuicao_erro_modelo.png")

    plt.show()


# ------------------------------------------------
# IMPORTÂNCIA DAS VARIÁVEIS
# ------------------------------------------------

def importancia_variaveis(modelo, nomes_features):

    importancia = modelo.feature_importances_

    df = pd.DataFrame({
        "variavel": nomes_features,
        "importancia": importancia
    })

    df = df.sort_values("importancia", ascending=False).head(10)

    plt.figure(figsize=(10,6))

    sns.barplot(data=df, x="importancia", y="variavel")

    plt.title("Variáveis Mais Importantes no Modelo")

    plt.xlabel("Importância")
    plt.ylabel("Variável")

    salvar_grafico("variaveis_importantes.png")

    plt.show()