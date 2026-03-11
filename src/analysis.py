import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import os

sns.set_theme(style="darkgrid")

IMAGES_DIR = "images"


def salvar_grafico(nome):

    os.makedirs(IMAGES_DIR, exist_ok=True)

    caminho = os.path.join(IMAGES_DIR, nome)

    plt.savefig(caminho, dpi=300, bbox_inches="tight")

    print(f"Gráfico salvo em: {caminho}")


# ------------------------------------------------
# CONVERSÃO USD → BRL
# ------------------------------------------------

def obter_taxa_dolar():

    url = "https://open.er-api.com/v6/latest/USD"

    response = requests.get(url)

    data = response.json()

    return data["rates"]["BRL"]


def converter_preco_para_reais(df):

    taxa = obter_taxa_dolar()

    print(f"Taxa USD -> BRL: {taxa}")

    df["preco_brl"] = df["price"] * taxa

    return df


# ------------------------------------------------
# DISTRIBUIÇÃO DE PREÇOS
# ------------------------------------------------

def distribuicao_precos(df):

    plt.figure(figsize=(10,6))

    sns.histplot(df["preco_brl"], bins=40)

    plt.title("Distribuição de Preços dos Airbnbs")

    plt.xlabel("Preço (R$)")
    plt.ylabel("Quantidade de imóveis")

    salvar_grafico("distribuicao_precos.png")

    plt.show()


# ------------------------------------------------
# DISTRIBUIÇÃO DOS TIPOS DE IMÓVEL
# ------------------------------------------------

def distribuicao_tipos_imovel(df):

    plt.figure(figsize=(8,5))

    sns.countplot(data=df, x="room_type")

    plt.title("Distribuição dos Tipos de Imóvel")

    plt.xlabel("Tipo de imóvel")
    plt.ylabel("Quantidade")

    salvar_grafico("tipos_imoveis.png")

    plt.show()


# ------------------------------------------------
# PREÇO MEDIANO POR TIPO
# ------------------------------------------------

def preco_mediano_por_tipo(df):

    mediana = df.groupby("room_type")["preco_brl"].median().reset_index()

    plt.figure(figsize=(12,5))

    sns.barplot(data=mediana, x="room_type", y="preco_brl")

    plt.title("Preço Mediano por Tipo de Imóvel")

    plt.xlabel("Tipo de imóvel")
    plt.ylabel("Preço mediano (R$)")

    salvar_grafico("preco_mediano_tipo.png")

    plt.show()


# ------------------------------------------------
# BAIRROS MAIS CAROS
# ------------------------------------------------

def bairros_mais_caros(df):

    top = (
        df.groupby("neighbourhood")["preco_brl"]
        .median()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,6))

    sns.barplot(x=top.values, y=top.index)

    plt.title("Top 10 Bairros Mais Caros")

    plt.xlabel("Preço mediano (R$)")
    plt.ylabel("Bairro")

    salvar_grafico("top_10_bairros.png")

    plt.show()


# ------------------------------------------------
# DISTRIBUIÇÃO DA CAPACIDADE
# ------------------------------------------------

def distribuicao_capacidade(df):

    plt.figure(figsize=(10,6))

    sns.histplot(df["accommodates"], bins=10)

    plt.title("Distribuição da Capacidade de Hóspedes")

    plt.xlabel("Número de hóspedes")
    plt.ylabel("Quantidade de imóveis")

    salvar_grafico("distribuicao_capacidade.png")

    plt.show()


# ------------------------------------------------
# MAPA DE DENSIDADE
# ------------------------------------------------

def mapa_densidade_airbnb(df):

    fig = px.density_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        radius=10,
        zoom=10,
        center=dict(
            lat=df["latitude"].mean(),
            lon=df["longitude"].mean()
        ),
        mapbox_style="carto-darkmatter",
        title="Densidade de Airbnbs no Rio de Janeiro"
    )

    fig.show()