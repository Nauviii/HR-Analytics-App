import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

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

    # Kolom numerik (exclude beberapa kolom)
    exclude_cols = ["Cluster", "Cluster Size", "Attrition Proportion"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    num_df = df[feature_cols].select_dtypes(include="number")

    if not num_df.empty:
        # --- 1. Bar Chart: Top Feature Paling Beda ---
        st.write("### Top Feature yang Membedakan")
        cluster_mean = num_df.loc[df["Cluster"] == selected_cluster].mean()
        global_mean = num_df.mean()

        diff = ((cluster_mean - global_mean) / global_mean).abs().sort_values(ascending=False)

        # Drop semua kolom yang mengandung kata "monthly" dan "income"
        drop_candidates = [c for c in diff.index if "monthly" in c.lower() and "income" in c.lower()]
        diff = diff.drop(drop_candidates, errors="ignore")

        top_features = diff.head(5).index

        compare_df = pd.DataFrame({
            "Feature": top_features,
            "Cluster": cluster_mean[top_features].values,
            "Global": global_mean[top_features].values
        })

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=compare_df["Feature"], y=compare_df["Cluster"], name="Cluster"))
        fig_bar.add_trace(go.Bar(x=compare_df["Feature"], y=compare_df["Global"], name="Global"))
        fig_bar.update_layout(barmode="group", title="Feature yang Paling Berpengaruh")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- 2. Radar Chart: Profil Cluster ---
        st.write("### 🕸️ Profil Cluster (Radar Chart)")
        radar_features = diff.head(6).index  # ambil 6 feature teratas setelah drop
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_mean[radar_features].values,
            theta=radar_features,
            fill="toself",
            name=f"Cluster {selected_cluster}"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=global_mean[radar_features].values,
            theta=radar_features,
            fill="toself",
            name="Global"
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Radar Chart – Cluster vs Global"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- Karakteristik Cluster (Ringkasan) dipindah ke bawah ---
    st.write("### 📊 Karakteristik Umum")
    char_df = pd.DataFrame(cluster_row).reset_index()
    char_df.columns = ["Atribut", "Nilai"]
    st.dataframe(char_df, use_container_width=True)


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
