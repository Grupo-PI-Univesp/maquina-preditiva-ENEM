from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Predição da nota média do ENEM", layout="wide")


@st.cache_resource
def get_model():
    return load_model("new_best_model")


def predict(model, data):
    prediction = predict_model(model, data=data)
    return prediction


model = get_model()


@st.cache_data
def gerar_df():
    df = pd.read_csv("unique_values.csv", sep=";")
    return df


df_form = gerar_df()

with st.sidebar:
    st.subheader("Projeto Integrador em Computação IV", divider=True)
    st.markdown(
        """ 
        ***Integrantes***:
        * David Barbosa de Araújo - RA 2006662
        * Elizabeth Fernandes Oliveira - RA 2005141
        * Luiz Felipe Louzas - RA 2105343
        * Newton de Lima Carlini - RA 2102889
        * Wellington Zanelatto - RA 2010657
        
        Orientadora: ***Vitória Silveira Teixeira Medrado*** 
       """
    )

st.title("Predição de desempenho do ENEM a partir de variáveis socioeconômicas")
st.markdown(
    """
    ***Este formulário foi elaborado com base no questionário socioeconômico exigido pelo MEC,
      no momento da inscrição do ENEM 2023. O objetivo é utilizar técnicas de machine learning 
      para analisar as respostas do questionário e prever a nota média obtida no exame.
      O questionário completo e demais informações 
      sobre os microdados do ENEM 2023 podem ser encontrados 
      [nesse link](https://www.gov.br/inep/pt-br/assuntos/noticias/enem/microdados-e-sinopse-estatistica-do-enem-2023-disponiveis)***
"""
)

st.subheader("Questionário socioeconômico", divider=True)

with st.form("Questionário socioeconômico"):
    formacao_mae = st.selectbox(
        "Até que série sua mãe, ou a mulher responsável por você, estudou?",
        df_form["educacao_mae"].unique(),
        index=None,
        placeholder="Escolha sua opção",
    )

    ocupacao_mae = st.selectbox(
        " indique o grupo que contempla a ocupação mais próxima da ocupação da sua mãe ou da mulher responsável por você.",
        df_form["ocupacao_mae"].unique(),
        index=None,
        placeholder="Escolha sua opção",
    )

    tem_empregado_domestico = st.selectbox(
        "Em sua residência trabalha empregado(a) doméstico(a)?",
        df_form["tem_empregado_domestico"].unique(),
        index=None,
        placeholder="Escolha sua opção",
    )

    num_quartos = st.selectbox(
        "Na sua residência tem quartos para dormir?",
        df_form["num_quartos"].unique(),
        index=None,
        placeholder="Escolha sua opção",
    )

    num_carros = st.selectbox(
        "Na sua residência tem carro?",
        df_form["num_carros"].unique(),
        index=None,
        placeholder="Escolha sua opção",
    )

    num_computadores = st.selectbox(
        "Na sua residência tem computador?",
        df_form["num_computadores"].unique(),
        index=None,
        placeholder="Escolha sua opção",
    )

    faixa_renda = st.selectbox(
        "Qual é a renda mensal de sua família? (Some a sua renda com a dos seus familiares.)",
        df_form["faixa_renda"].unique(),
        index=None,
        placeholder="Escolha sua opção",
    )

    enviar = st.form_submit_button("enviar")

# Dicionário para montaro dataframe de previsão
input_dict = {
    "educacao_mae": formacao_mae,
    "ocupacao_mae": ocupacao_mae,
    "tem_empregado_domestico": tem_empregado_domestico,
    "num_quartos": num_quartos,
    "num_carros": num_carros,
    "num_computadores": num_computadores,
    "faixa_renda": faixa_renda,
}

input_df = pd.DataFrame([input_dict])

if enviar:
    output = predict(model, input_df)
    previsao = float(output["prediction_label"])

    st.subheader(
        f"Com base nas respostas do questionário a nota média prevista é de: {previsao:.2f}",
        divider=True,
    )
