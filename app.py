import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Probabilidade de Contratação - Decision", layout="wide")
st.title("Probabilidade de Contratação — Decision")
st.caption("Envie o modelo calibrado (.joblib) e um CSV com dados de candidaturas para obter as probabilidades de contratação.")

def get_pre_from_calibrated(model):
    """
    Retorna o ColumnTransformer 'pre' a partir de um CalibratedClassifierCV.
    """
    cc_list = getattr(model, "calibrated_classifiers_", None)
    if not cc_list:
        return None
    inner = cc_list[0]
    base = getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None)
    if base is None:
        return None
    try:
        return base.named_steps["pre"]
    except Exception:
        return None

def expected_input_columns(model):
    """
    Extrai a lista de colunas que a pipeline original esperava (antes do OHE).
    """
    pre = get_pre_from_calibrated(model)
    if pre is None:
        return None
    cols = []
    try:
        for name, transformer, columns in pre.transformers_:
            if name == "remainder":
                continue
            if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                cols.extend(list(columns))
    except Exception:
        return None
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out or None

def align_columns_for_inference(df_in: pd.DataFrame, expected_cols):
    """
    Garante que df_in tenha exatamente as colunas esperadas pela pipeline (mesma ordem).
    Cria ausentes como NaN e ignora extras.
    """
    X = df_in.copy()
    missing = [c for c in expected_cols if c not in X.columns]
    for c in missing:
        X[c] = np.nan
    X = X[expected_cols]  
    extras = [c for c in df_in.columns if c not in expected_cols]
    return X, missing, extras

def detect_group_column(df: pd.DataFrame):
    cands = [c for c in df.columns if "vaga" in c.lower() and "id" in c.lower()]
    if cands: return cands[0]
    for c in ["id_vaga", "vaga_id", "idvaga", "id da vaga", "id_da_vaga"]:
        if c in df.columns: return c
    return None

# Sidebar
st.sidebar.header("⚙️ Arquivos")
model_file = st.sidebar.file_uploader("Modelo (.joblib)", type=["joblib", "pkl"])
threshold  = st.sidebar.slider("Limiar de classificação", 0.0, 1.0, 0.50, 0.01)
topk       = st.sidebar.number_input("3 (TOP-3 por vaga)", min_value=1, max_value=50, value=3, step=1)

# Carregar modelo
model = None
if model_file is not None:
    try:
        model = load(model_file)
        st.success("Modelo carregado.")
    except Exception as e:
        st.error(f"Falha ao carregar o modelo: {e}")

# Upload de dados
st.subheader("Dados de entrada (CSV)")
data_file = st.file_uploader("Envie o CSV com as mesmas colunas de treino (exceto alvo/IDs).", type=["csv"])

if model is not None and data_file is not None:
    try:
        df_raw = pd.read_csv(data_file)
    except Exception:
        data_file.seek(0)
        df_raw = pd.read_csv(data_file, sep=";")

    st.write("Dimensões:", df_raw.shape)
    st.dataframe(df_raw.head(20), use_container_width=True)
    exp_cols = expected_input_columns(model)
    if exp_cols is None:
        st.warning("Não consegui detectar as colunas esperadas da pipeline. "
                   "Vou usar todas as colunas do CSV (exceto alvo/IDs).")
        exp_cols = [c for c in df_raw.columns]

    drop_alvo = ["status_contratacao", "status-contratacao"]
    drop_ids  = ["id_candidato","codigo_profissional","codigo","id_profissional","cpf",
                 "nome","email","telefone","id_vaga","vaga_id","idvaga","id_da_vaga","ID_VAGA","ID VAGA"]
    df_feat = df_raw.drop(columns=[c for c in drop_alvo + drop_ids if c in df_raw.columns], errors="ignore")

    X, missing, extras = align_columns_for_inference(df_feat, exp_cols)
    if missing:
        st.info(f"ℹ️ Criadas {len(missing)} colunas ausentes como NaN: {missing[:15]}{' ...' if len(missing)>15 else ''}")
    if extras:
        st.info(f"ℹ️ Ignoradas {len(extras)} colunas não usadas pela pipeline: {extras[:15]}{' ...' if len(extras)>15 else ''}")

    # Predições calibradas
    with st.spinner("Calculando probabilidades…"):
        probs = model.predict_proba(X)[:, 1]
        pred  = (probs >= threshold).astype(int)

    # Saída básica
    st.subheader("Resultados")
    out = df_raw.copy()
    out["prob_contratacao"] = probs
    out["predito"] = pred
    st.dataframe(out.head(20), use_container_width=True)

    # Download
    st.download_button(
        "⬇️ Baixar resultados (CSV)",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name="predicoes_calibradas.csv",
        mime="text/csv",
    )

    # TOP-3 por vaga
    st.subheader("TOP-3 por vaga")
    default_group = detect_group_column(df_raw)
    group_col = st.selectbox(
        "Selecione a coluna de grupo (vaga), se existir:",
        ["(nenhuma)"] + sorted(df_raw.columns.tolist()),
        index=(0 if default_group is None else 1 + sorted(df_raw.columns.tolist()).index(default_group))
    )
    if group_col != "(nenhuma)":
        if group_col not in out.columns:
            st.warning(f"A coluna '{group_col}' não está disponível no resultado.")
        else:
            topk_df = (
                out.sort_values([group_col, "prob_contratacao"], ascending=[True, False])
                   .groupby(group_col, as_index=False)
                   .head(int(topk))
                   .reset_index(drop=True)
            )
            st.dataframe(topk_df, use_container_width=True)
            st.download_button(
                f"Baixar TOP-{topk} por vaga (CSV)",
                data=topk_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"top{topk}_por_vaga.csv",
                mime="text/csv",
            )

elif model is None:
    st.info("Envie o **modelo calibrado (.joblib)** na barra lateral.")
elif data_file is None:
    st.info("⬆Envie o **CSV** com os candidatos para pontuar.")
