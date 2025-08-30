import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(layout="wide")


#  Utilities 
def ensure_cluster_column(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "Cluster" in df2.columns:
        df2 = df2.reset_index(drop=True)
        return df2

    idx_name = df2.index.name if df2.index.name is not None else "index"
    df2 = df2.reset_index()
    df2 = df2.rename(columns={idx_name: "Cluster"})
    return df2


#  Data Loader 
@st.cache_data
def load_summary(path: str) -> pd.DataFrame:
    df = joblib.load(path)
    return ensure_cluster_column(df)


#  Plot Functions 
def plot_cluster_proportion(df: pd.DataFrame) -> None:
    fig = px.pie(
        df,
        names="Cluster",
        values="Cluster Size",
        title="Distribusi Ukuran Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_attrition_proportion(df: pd.DataFrame) -> None:
    fig = px.bar(
        df,
        x="Cluster",
        y="Attrition Proportion",
        title="Attrition Proportion per Cluster",
        text="Attrition Proportion"
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)


#  Display Functions 
def show_summary(df: pd.DataFrame) -> None:
    st.subheader("📊 Ringkasan Cluster")
    st.dataframe(df)

    col1, col2 = st.columns(2)
    with col1:
        plot_cluster_proportion(df)
    with col2:
        plot_attrition_proportion(df)


def show_detail(df: pd.DataFrame) -> None:
    st.subheader("🔍 Analisis Detail per Cluster")

    cluster_options = df["Cluster"].tolist()
    selected_cluster = st.selectbox("Pilih Cluster", cluster_options)

    cluster_row = df.loc[df["Cluster"] == selected_cluster].squeeze()

    # --- Karakteristik Cluster (Improved) ---
    st.write("### Karakteristik Cluster")

    # Tampilkan tabel ringkas
    char_df = pd.DataFrame(cluster_row).reset_index()
    char_df.columns = ["Atribut", "Nilai"]
    st.dataframe(char_df, use_container_width=True)

    # Pisahkan numerik & kategorik dari Series
    exclude_cols = ["Cluster"]
    char_data = cluster_row.drop(labels=exclude_cols, errors="ignore")

    num_data = char_data[char_data.apply(lambda x: pd.api.types.is_numeric_dtype(type(x)))]
    cat_data = char_data[~char_data.apply(lambda x: pd.api.types.is_numeric_dtype(type(x)))]


    # Kalau ada kategorik, tampilkan tabel kecil
    if not cat_data.empty:
        st.write("#### Atribut Kategorikal")
        st.table(cat_data.reset_index().rename(columns={"index": "Atribut", 0: "Nilai"}))

# --- Mapping Cluster Number -> Label ---
def map_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {0: "A", 1: "B"}
    df = df.copy()
    if df["Cluster"].dtype != "object":  # kalau masih angka
        df["Cluster"] = df["Cluster"].map(mapping)
    return df


def show_insight(df: pd.DataFrame) -> None:
    # --- Ranking Cluster berdasarkan attrition ---
    ranking_df = df[["Cluster", "Cluster Size", "Attrition Proportion"]].copy()
    ranking_df = ranking_df.sort_values("Attrition Proportion", ascending=False).reset_index(drop=True)

    # --- Tampilkan cluster dengan attrition tertinggi & terendah ---
    col1, col2 = st.columns(2)
    top = ranking_df.iloc[0]
    low = ranking_df.iloc[-1]

    with col1:
        st.subheader("Cluster Attrition Tertinggi")
        st.metric(label="Attrition Rate",
                  value=f"{top['Attrition Proportion']:.2%}",
                  delta=f"Ukuran: {top['Cluster Size']} karyawan")

    with col2:
        st.subheader("Cluster Attrition Terendah")
        st.metric(label="Attrition Rate",
                  value=f"{low['Attrition Proportion']:.2%}",
                  delta=f"Ukuran: {low['Cluster Size']} karyawan")

    # --- Insight & Rekomendasi ---
    st.subheader("Insight & Rekomendasi Strategi Retensi")

    st.markdown(f"""
    - Fokus utama strategi retensi perlu diarahkan ke **Cluster {top['Cluster']}** dengan prioritas pada pengembangan karier, kompensasi, dan adaptasi manajemen baru.  
    - Sementara itu, **Cluster {low['Cluster']}** perlu tetap dijaga stabilitasnya melalui penghargaan, fleksibilitas, dan motivasi kerja jangka panjang.  

    Berdasarkan hasil clustering:

    - **Cluster {top['Cluster']}** memiliki tingkat attrition **paling tinggi** sebesar **{top['Attrition Proportion']:.2%}**.
      Karakteristik umum: usia & masa kerja relatif lebih muda, jabatan dan pendapatan cenderung lebih rendah.  
      Rekomendasi: fokus pada *career development program*, peningkatan kompensasi, dan program engagement untuk generasi muda.

    - **Cluster {low['Cluster']}** memiliki tingkat attrition **paling rendah** sebesar **{low['Attrition Proportion']:.2%}**.
      Karakteristik umum: karyawan lebih senior, pendapatan lebih tinggi, loyalitas lebih baik.  
      Rekomendasi: tetap pertahankan kepuasan kerja dengan *recognition*, fleksibilitas, dan peluang kepemimpinan.

    Secara keseluruhan, strategi retensi bisa diarahkan dengan memprioritaskan **Cluster {top['Cluster']}**,
    tanpa mengabaikan kebutuhan pengembangan & motivasi bagi **Cluster {low['Cluster']}**.
    """)




# --- KPI Section ---
def show_kpi(df: pd.DataFrame) -> None:
    total_employees = df["Cluster Size"].sum()
    n_clusters = df["Cluster"].nunique()
    avg_attrition = df["Attrition Proportion"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Karyawan", f"{total_employees}")
    col2.metric("🔎 Jumlah Cluster", f"{n_clusters}")
    col3.metric("📉 Rata-rata Attrition", f"{avg_attrition:.2%}")




# --- Main App (ubah sedikit) ---
def main() -> None:
    st.title("📈 Dashboard HR Analytics - Attrition Clustering")

    df_summary = load_summary("data_summary.joblib")  
    df_summary = map_cluster_labels(df_summary)  

    show_kpi(df_summary)   # <---- Tambahan KPI

    tab1, tab2, tab3 = st.tabs(["Ringkasan", "Detail Cluster", "Insight"])

    with tab1:
        show_summary(df_summary)    

    with tab2:
        show_detail(df_summary)

    with tab3:
        show_insight(df_summary)



if __name__ == "__main__":
    main()
