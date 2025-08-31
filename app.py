# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ====================================
# CONFIG
# ====================================
st.set_page_config(page_title="Dashboard Attrition", layout="wide")
sns.set_theme(style="whitegrid")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# Inject CSS styling
st.markdown("""
    <style>
    /* Background utama */
    .stApp {
        background-color: white;
        color: black;
    }

    /* Styling metric box */
    div[data-testid="stMetric"] {
        background-color: #E3F2FD !important;
        border: 1px solid #90CAF9;
        padding: 15px;
        border-radius: 10px;
    }

    [data-testid="stMetricValue"], 
    [data-testid="stMetricLabel"] {
        color: black !important;
    }

    /* Enhanced Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F5F5F5 !important;
        border-right: 2px solid #E0E0E0;
    }
    
    /* Sidebar content styling */
    section[data-testid="stSidebar"] > div {
        background-color: #F5F5F5 !important;
        padding-top: 20px;
    }
    
    /* Sidebar text styling */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #333333 !important;
    }
    
    /* Sidebar title styling */
    section[data-testid="stSidebar"] h1 {
        color: #1976D2 !important;
        font-weight: bold;
        border-bottom: 2px solid #90CAF9;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Radio button styling in sidebar */
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
        color: #333333 !important;
    }
    
    /* Radio button options */
    section[data-testid="stSidebar"] .stRadio > div {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #E0E0E0;
    }
    
    /* Radio button text */
    section[data-testid="stSidebar"] .stRadio label span {
        color: #333333 !important;
        font-weight: 500;
    }
    
    /* Radio button options on hover */
    section[data-testid="stSidebar"] .stRadio > div:hover {
        background-color: #E3F2FD;
        border-color: #90CAF9;
    }
    
    /* Selected radio button */
    section[data-testid="stSidebar"] .stRadio > div[data-checked="true"] {
        background-color: #E3F2FD !important;
        border-color: #1976D2 !important;
    }
    
    /* Sidebar close button */
    section[data-testid="stSidebar"] button[kind="header"] {
        color: #333333 !important;
        background-color: white !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 4px;
    }
    
    /* Sidebar close button hover */
    section[data-testid="stSidebar"] button[kind="header"]:hover {
        background-color: #F0F0F0 !important;
        border-color: #BDBDBD !important;
    }
    
    /* All sidebar text elements */
    section[data-testid="stSidebar"] * {
        color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)


# ====================================
# LOADERS
# ====================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource
def load_evaluation_results(path: str):
    return joblib.load(path)

# ====================================
# FEATURE COMPATIBILITY CHECK
# ====================================
def check_model_compatibility(df: pd.DataFrame, model):
    """Check if dataset features match model expectations"""
    expected_features = getattr(model, 'feature_names_in_', None)
    if expected_features is None:
        return True, "Cannot determine expected features"
    
    X = df.drop("Attrition", axis=1) if "Attrition" in df.columns else df
    current_features = set(X.columns)
    expected_features = set(expected_features)
    
    if current_features == expected_features:
        return True, "Features match perfectly"
    else:
        return False, f"Feature mismatch detected"

# ====================================
# UTILS (Plot)
# ====================================
def plot_confusion_matrix_from_results(cm_data):
    """Plot confusion matrix from saved results"""
    import numpy as np
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No", "Yes"], 
                yticklabels=["No", "Yes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def plot_roc_curve_from_results(y_true, y_prob):
    """Plot ROC curve from saved results"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
    ax.plot([0,1], [0,1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

def show_classification_report(report_dict):
    """Display classification report in a nice format"""
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Display overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = report_dict.get('accuracy', 0)
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        macro_f1 = report_dict.get('macro avg', {}).get('f1-score', 0)
        st.metric("Macro F1-Score", f"{macro_f1:.3f}")
        
    with col3:
        weighted_f1 = report_dict.get('weighted avg', {}).get('f1-score', 0)
        st.metric("Weighted F1-Score", f"{weighted_f1:.3f}")
    
    # Show detailed report
    st.write("**Detailed Classification Report:**")
    # Filter out accuracy row and format nicely
    display_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    st.dataframe(display_df.round(3), use_container_width=True)

def plot_feature_importance(model, features):
    if hasattr(model, "feature_importances_"):
        feature_imp = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=feature_imp, x="Importance", y="Feature", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.info("Model ini tidak mendukung feature importance secara langsung.")

# ====================================
# SECTIONS
# ====================================
def show_summary(df: pd.DataFrame):
    st.title("📊 Dashboard Analisis Attrition Karyawan")

    col1, col2, col3 = st.columns(3)
    total_karyawan = len(df)
    
    # Debug: Check unique values in Attrition column
    attrition_values = df["Attrition"].unique()
    
    # More robust attrition calculation
    if "Yes" in attrition_values:
        total_attrition = df[df["Attrition"] == "Yes"].shape[0]
    elif 1 in attrition_values:  # If encoded as 1/0
        total_attrition = df[df["Attrition"] == 1].shape[0]
    elif "yes" in attrition_values:  # If lowercase
        total_attrition = df[df["Attrition"] == "yes"].shape[0]
    else:
        # Count non-zero/non-"No" values
        total_attrition = df[~df["Attrition"].isin(["No", "no", 0])].shape[0]
    
    persentase_attrition = round((total_attrition / total_karyawan) * 100, 2) if total_karyawan > 0 else 0

    col1.metric("Total Karyawan", total_karyawan)
    col2.metric("Attrition (Yes)", total_attrition)
    col3.metric("Persentase Attrition", f"{persentase_attrition}%")
    

def show_eda(df):
    st.subheader("Exploratory Data Analysis")

    # Fitur Numerik
    st.markdown("### 📈 Fitur Numerik")
    numeric_features = ["TotalWorkingYears", "Age", "YearsAtCompany", "YearsWithCurrManager", "MonthlyIncome"]
    cols = st.columns(len(numeric_features))
    for i, feat in enumerate(numeric_features):
        with cols[i]:
            st.markdown(f"**{feat}**")  # Bold title above plot
            fig, ax = plt.subplots(figsize=(3, 2.5))
            
            # Check if Attrition column exists and has valid values
            if "Attrition" in df.columns:
                sns.histplot(data=df, x=feat, hue="Attrition", kde=True, ax=ax)
            else:
                sns.histplot(data=df, x=feat, kde=True, ax=ax)
                
            ax.set_xlabel(feat, fontsize=10, fontweight='bold', color='black')
            ax.set_ylabel("Count", fontsize=10, fontweight='bold', color='black')
            ax.tick_params(axis='x', labelsize=8, colors='black')
            ax.tick_params(axis='y', labelsize=8, colors='black')
            
            # Safe legend handling
            legend = ax.get_legend()
            if legend is not None:
                legend.set_frame_on(True)
                if hasattr(legend, 'set_facecolor'):
                    legend.set_facecolor('white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Fitur Kategorikal
    st.markdown("### 📈 Fitur Kategorikal")
    categorical_features = ["Department", "JobRole", "Gender", "MaritalStatus", "OverTime"]
    cols = st.columns(len(categorical_features))
    for i, feat in enumerate(categorical_features):
        with cols[i]:
            st.markdown(f"**{feat}**")  # Bold title above plot
            
            # Check if feature exists in dataframe
            if feat not in df.columns:
                st.warning(f"Column '{feat}' not found in data")
                continue
                
            fig, ax = plt.subplots(figsize=(3, 2.5))
            
            # Check if Attrition column exists and has valid values
            if "Attrition" in df.columns:
                sns.countplot(data=df, x=feat, hue="Attrition", ax=ax)
            else:
                sns.countplot(data=df, x=feat, ax=ax)
                
            ax.set_xlabel(feat, fontsize=10, fontweight='bold', color='black')
            ax.set_ylabel("Count", fontsize=10, fontweight='bold', color='black')
            ax.tick_params(axis='x', labelsize=8, colors='black')
            ax.tick_params(axis='y', labelsize=8, colors='black')
            
            # Fix label rotation and visibility
            if feat in ["JobRole", "Department"]:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8, color='black')
            else:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=20, fontsize=8, color='black')
            
            # Safe legend handling
            legend = ax.get_legend()
            if legend is not None:
                legend.set_frame_on(True)
                if hasattr(legend, 'set_facecolor'):
                    legend.set_facecolor('white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

def show_model_results():
    st.subheader("🤖 Hasil Model Klasifikasi")

    try:
        # Load saved evaluation results
        results = load_evaluation_results("model_evaluation.joblib")
        
        # Display classification report metrics
        if 'classification_report' in results:
            show_classification_report(results['classification_report'])
            st.markdown("---")
        
        # Create three columns for visualizations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Confusion Matrix**")
            if 'confusion_matrix' in results:
                plot_confusion_matrix_from_results(results['confusion_matrix'])
            else:
                st.info("Confusion matrix tidak tersedia")
            
        with col2:
            st.write("**ROC Curve**")
            if 'y_true' in results and 'y_prob' in results:
                plot_roc_curve_from_results(results['y_true'], results['y_prob'])
            else:
                st.info("Data untuk ROC curve tidak tersedia")
            
        with col3:
            st.write("**Model Info**")
            st.info(f"✅ Evaluation results loaded successfully")
            if 'accuracy' in results:
                st.write(f"**Model Accuracy:** {results['accuracy']:.3f}")
            if 'feature_names' in results:
                st.write(f"**Features used:** {len(results['feature_names'])}")
                with st.expander("Show feature names"):
                    st.write(", ".join(results['feature_names']))
            
    except FileNotFoundError:
        st.error("❌ **File `model_evaluation.joblib` tidak ditemukan!**")
        st.info("Pastikan file evaluation results sudah di-upload ke direktori yang sama")
    except Exception as e:
        st.error(f"❌ **Error loading evaluation results:** {str(e)}")
        st.info("Periksa format file model_evaluation.joblib")

def show_insight():
    st.subheader("💡 Insight & Rekomendasi")
    st.write("""
    **Insight dari Analisis Attrition**

    1. Kepuasan kerja dan lingkungan rendah jadi pemicu utama
    Karyawan dengan tingkat kepuasan kerja dan lingkungan rendah lebih rentan meninggalkan perusahaan.

    2. Attrition tinggi pada karyawan muda
    Sebagian besar attrition terjadi pada karyawan dengan usia muda, yang umumnya berada di tahap awal karier.

    3. Departemen Research & Development paling terdampak
    R&D memiliki tingkat attrition tertinggi dibandingkan departemen lain, menunjukkan adanya tantangan khusus di divisi ini (beban kerja, tekanan inovasi, atau kesempatan karier).

    4. Tenure singkat dan kepemimpinan baru berhubungan dengan attrition
    Mayoritas karyawan yang keluar memiliki masa kerja 0–5 tahun, terutama saat terjadi transisi manajemen. Hal ini mengindikasikan periode adaptasi yang kurang optimal.

    5. Pendapatan relatif rendah berkontribusi terhadap keputusan keluar
    Karyawan dengan gaji lebih rendah cenderung lebih sering melakukan attrition, menandakan isu kompensasi yang mempengaruhi retensi.
    
    **Rekomendasi Strategis:**
    
    1. Tingkatkan kepuasan kerja & lingkungan
        - Lakukan survey engagement secara rutin untuk mengidentifikasi faktor ketidakpuasan.
        - Sediakan jalur komunikasi terbuka (feedback channel) dan program peningkatan kesejahteraan (well-being program).

    2. Strategi retensi khusus untuk karyawan muda
        - Kembangkan program mentorship dan career development plan untuk karyawan baru.
        - Fokuskan pelatihan soft skill dan peluang rotasi kerja untuk memperluas pengalaman.

    3. Perbaikan manajemen di R&D
        - Tinjau workload dan budaya kerja di R&D.
        - Pertimbangkan insentif dan reward khusus untuk inovasi agar karyawan merasa dihargai.

    4. Dukungan saat transisi manajemen
        - Buat program onboarding manager baru untuk mengurangi gesekan dengan tim.
        - Lakukan check-in meeting rutin selama periode kepemimpinan baru agar adaptasi lebih lancar.

    5. Tinjauan kompensasi dan benefit
        - Lakukan benchmarking gaji dengan standar industri.
        - Tawarkan benefit non-finansial seperti fleksibilitas kerja, pengembangan skill, atau bonus berbasis kinerja untuk menambah daya tarik perusahaan.

    """)

# ====================================
# MAIN APP
# ====================================
def main():
    try:
        df = load_data("employee_data.csv")  
        
        show_summary(df)
        st.markdown("---")

        # Sidebar untuk navigasi
        st.sidebar.title("📌 Navigasi")
        page = st.sidebar.radio("Pilih Halaman:", ["📈 EDA", "🤖 Hasil Model", "💡 Insight"])

        if page == "📈 EDA":
            show_eda(df)
        elif page == "🤖 Hasil Model":
            show_model_results()
        elif page == "💡 Insight":
            show_insight()
            
    except FileNotFoundError as e:
        st.error(f"❌ **File tidak ditemukan:** {e}")
        st.info("Pastikan file `employee_data.csv` dan `model_evaluation.joblib` ada di direktori yang sama dengan app.py")
    except Exception as e:
        st.error(f"❌ **Error:** {e}")
        st.info("Periksa format file dan kompatibilitas data")

if __name__ == "__main__":
    main()
