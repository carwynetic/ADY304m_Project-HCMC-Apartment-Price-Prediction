import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import base64
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. CẤU HÌNH TRANG (ADVANCED PAGE CONFIG)
# ==========================================
st.set_page_config(
    page_title="FPT HCMC Apartment Group 6",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DEFINITIONS (CUSTOM TRANSFORMERS & MATH)
#   - LogAreaTransformer: Biến đổi log cho diện tích để giảm độ lệch phải
# ==========================================
class LogAreaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame): s = X.iloc[:, 0]
        else: s = pd.Series(np.ravel(X))
        s = pd.to_numeric(s, errors="coerce")
        return np.log1p(s.to_numpy(dtype=float)).reshape(-1, 1)

class FurnitureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame): s = X.iloc[:, 0]
        else: s = pd.Series(np.ravel(X))
        s = s.astype(str).str.strip().str.lower()
        mapped = s.map({"không": 0, "khong": 0, "ko": 0, "có": 1, "co": 1})
        return mapped.to_numpy(dtype=float).reshape(-1, 1)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('outputs/models/best_model.pkl')
        return model
    except FileNotFoundError:
        st.error("FileNotFound: outputs/models/best_model.pkl")
        st.stop()

# ==========================================
# 3. GLOBAL STYLING (THE SUPER BEAUTIFUL PART)
# ==========================================
FPT_ORANGE = "#F37021"
FPT_BLUE = "#004B8F"
DARK_TEXT = "#E0E0E0"

# Custom CSS cho Glassmorphism KPI, Fonts, and Sidebar
st.markdown(f"""
    <style>
    /* Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"]  {{
        font-family: 'Roboto', sans-serif;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: #0E1117;
        border-right: 1px solid #30363d;
    }}
    section[data-testid="stSidebar"] h1 {{
        color: {DARK_TEXT};
        font-size: 1.2rem;
        font-weight: 300;
        text-align: center;
        margin-top: -30px;
    }}
    
    /* Title styling */
    .fpt-title {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {DARK_TEXT};
        margin-bottom: 0px;
        letter-spacing: -1px;
    }}
    .fpt-subtitle {{
        font-size: 1rem;
        color: #A0A0A0;
        margin-bottom: 20px;
    }}
    
    /* Glassmorphism KPI Cards */
    .kpi-card {{
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
    }}
    .kpi-card:hover {{
        border: 1px solid rgba(243, 112, 33, 0.5); /* FPT Orange hover */
        transform: translateY(-3px);
    }}
    .kpi-label {{
        color: #A0A0A0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 300;
    }}
    .kpi-value {{
        color: {FPT_ORANGE};
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }}
    .kpi-unit {{
        color: #E0E0E0;
        font-size: 0.9rem;
        font-weight: 400;
    }}
    .kpi-icon {{
        font-size: 1.5rem;
        float: right;
        opacity: 0.5;
    }}
    
    /* Form & Input styling */
    .stForm {{
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border: none;
        border-radius: 5px;
        color: #A0A0A0;
        font-weight: 400;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {FPT_ORANGE};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(243, 112, 33, 0.1);
        color: {DARK_TEXT};
        font-weight: 700;
    }}
    
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. DATA LOADING (CACHING)
# ==========================================
@st.cache_data
def load_and_cache_data():
    try:
        # Load file train để lấy danh sách Quận/Huyện chuẩn
        df_full = pd.read_csv('data/stage2_final.csv')
        df_full['District'] = df_full['District'].astype(str).str.strip().str.title()
        df_full['Furniture'] = df_full['Furniture'].astype(str).str.strip()
        
        # Load file Feature Importance (Hủy diệt "placeholder")
        fi_df = pd.read_csv('outputs/tables/random_forest_permutation_importance_top10.csv')
        return df_full, fi_df
    except FileNotFoundError:
        st.error("Missing critical data files. Check structure.")
        st.stop()

df_raw, fi_df = load_and_cache_data()
model = load_model()

# ==========================================
# 5. SIDEBAR (GLOBAL CONTROLS + FPT LOGO)
# ==========================================

# 1. HACK ĐỂ LOAD SVG LOGO FPT
def render_svg_logo(svg_path, width=200):
    """Đọc file SVG và render bằng HTML img tag"""
    try:
        with open(svg_path, "rb") as f:
            svg_str = f.read().decode("utf-8")
        # Base64 encode để an toàn khi nhúng HTML
        b64_svg = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
        html = f"""
            <div style="display: flex; justify-content: center; margin-bottom: 20px; margin-top: -40px;">
                <img src="data:image/svg+xml;base64,{b64_svg}" width="{width}" />
            </div>
        """
        st.sidebar.markdown(html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.sidebar.error("MISSING: Logo_Trường_Đại_học_FPT.svg")

# 2. RENDER LOGO
render_svg_logo("Logo_Trường_Đại_học_FPT.svg", width=180)

# 3. Sidebar content
st.sidebar.markdown("<h1>ADY201m | Group 6</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

all_districts = sorted(df_raw['District'].dropna().unique())
selected_districts = st.sidebar.multiselect("📍 Vị trí", options=all_districts, default=all_districts)

min_area, max_area = float(df_raw['Area_m2'].min()), float(df_raw['Area_m2'].max())
selected_area = st.sidebar.slider("📏 Diện tích (m²)", min_area, max_area, (min_area, max_area))

min_price, max_price = float(df_raw['Price_Million_VND'].min()), float(df_raw['Price_Million_VND'].max())
selected_price = st.sidebar.slider("💰 Tổng Giá (Triệu VNĐ)", min_price, max_price, (min_price, max_price))

# Filter
df_filtered = df_raw[
    (df_raw['District'].isin(selected_districts)) &
    (df_raw['Area_m2'].between(selected_area[0], selected_area[1])) &
    (df_raw['Price_Million_VND'].between(selected_price[0], selected_price[1]))
]

if df_filtered.empty:
    st.warning("No data found for selected filters.")
    st.stop()

# ==========================================
# 6. HELPERS FOR SUPER BEAUTIFUL CHARTS
# ==========================================
def custom_plotly_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(color="#A0A0A0", size=15, weight='normal'),
        font=dict(color="#A0A0A0", size=12),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title_font=dict(size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)", title_font=dict(size=11)),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig

#Icon library HTML/CSS
ICONS = {
    "count": '<i class="fa fa-home kpi-icon"></i>',
    "price_per_m2": '<i class="fa fa-tag kpi-icon"></i>',
    "area": '<i class="fa fa-expand-arrows-alt kpi-icon"></i>',
    "top_dist": '<i class="fa fa-trophy kpi-icon"></i>',
    "robot": '<i class="fa fa-robot kpi-icon"></i>',
    "target": '<i class="fa fa-bullseye kpi-icon"></i>'
}

# ==========================================
# 7. MAIN AREA LAYOUT
# ==========================================
#Thư viện Icon FontAwesome
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">', unsafe_allow_html=True)

st.markdown('<p class="fpt-title">HCMC Apartment | Trí Tuệ Nhân Tạo</p>', unsafe_allow_html=True)
st.markdown('<p class="fpt-subtitle">Phân tích và dự đoán đơn giá căn hộ tại TP.HCM | Project ADY201m | FPT University</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Tổng quan Thị trường", "🔍 EDA", "⚖️ Predictor"])

# -------------------------------------------------------------------------
# TAB 1: TỔNG QUAN THỊ TRƯỜNG
# -------------------------------------------------------------------------
with tab1:
    # --- KPI Glassmorphism Cards ---
    st.markdown("<br>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_listings = len(df_filtered)
    avg_price_m2 = df_filtered['Price_Per_M2'].mean()
    avg_area = df_filtered['Area_m2'].mean()
    top_district_series = df_filtered.groupby('District')['Price_Per_M2'].mean()
    if not top_district_series.empty:
        top_district_name = top_district_series.idxmax()
        top_district_val = top_district_series.max()
    else:
        top_district_name = "N/A"
        top_district_val = 0

    #Card 1: Tổng tin đăng
    kpi1.markdown(f"""
        <div class="kpi-card">
            {ICONS['count']}
            <div class="kpi-label">Tổng số tin đăng</div>
            <div class="kpi-value">{total_listings/1000:.2f} <span class="kpi-unit">k</span></div>
        </div>
    """, unsafe_allow_html=True)
    
    #Card 2: Giá m2 trung bình
    kpi2.markdown(f"""
        <div class="kpi-card">
            {ICONS['price_per_m2']}
            <div class="kpi-label">Giá trung bình</div>
            <div class="kpi-value">{avg_price_m2:.1f} <span class="kpi-unit">Tr/m²</span></div>
        </div>
    """, unsafe_allow_html=True)

    #Card 3: Diện tích trung bình
    kpi3.markdown(f"""
        <div class="kpi-card">
            {ICONS['area']}
            <div class="kpi-label">Diện tích trung bình</div>
            <div class="kpi-value">{avg_area:.1f} <span class="kpi-unit">m²</span></div>
        </div>
    """, unsafe_allow_html=True)

    #Card 4: Quận đắt nhất
    kpi4.markdown(f"""
        <div class="kpi-card">
            {ICONS['top_dist']}
            <div class="kpi-label">Khu vực đắt nhất</div>
            <div class="kpi-value" style="color: {DARK_TEXT}; font-size: 1.2rem; margin-top:15px; text-transform:uppercase;">{top_district_name}</div>
            <div class="kpi-unit" style="margin-top:-5px;">{top_district_val:.1f} Tr/m²</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Charts Area: Cố định tỷ lệ [7, 3] Super Beautiful ---
    col_charts_70, col_charts_30 = st.columns([7, 3], gap="large")
    
    with col_charts_70:
        # 1. Bar chart: Quận giá cao
        st.markdown('<p style="color:#A0A0A0; font-size:1.1rem; weight:300;">🥇 TOP 10 KHU VỰC DẪN ĐẦU VỀ GIÁ</p>', unsafe_allow_html=True)
        df_top10 = df_filtered.groupby('District')['Price_Per_M2'].mean().sort_values(ascending=False).head(10).reset_index()
        # Scale màu Cividis chuyên nghiệp
        fig_bar = px.bar(df_top10, x='Price_Per_M2', y='District', orientation='h',
                         color='Price_Per_M2', color_continuous_scale="cividis")
        fig_bar = custom_plotly_theme(fig_bar)
        fig_bar.update_layout(height=380, margin=dict(t=0), coloraxis_showscale=False)
        fig_bar.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info("Insight: Sự phân hóa giá rõ rệt: Nhóm lõi Q.1, Q.2, Q.3 đắt gấp 2-3 lần vùng ven.")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # 2. Histogram: Phân phối giá
        st.markdown('<p style="color:#A0A0A0; font-size:1.1rem; weight:300;">📈 PHÂN PHỐI ĐƠN GIÁ TOÀN THỊ TRƯỜNG (TRIỆU/m²)</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(df_filtered, x='Price_Per_M2', nbins=50,
                                color_discrete_sequence=[FPT_BLUE], opacity=0.8)
        fig_hist = custom_plotly_theme(fig_hist)
        fig_hist.update_layout(height=380, margin=dict(t=0))
        # Add mean line
        mean_val = df_filtered['Price_Per_M2'].mean()
        fig_hist.add_vline(x=mean_val, line_dash="dash", line_color=FPT_ORANGE)
        fig_hist.update_xaxes(ticksuffix=" Tr/m²")
        fig_hist.update_yaxes(title="Số lượng tin đăng")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.info(f"Insight: Phân phối lệch phải, đơn giá tập trung phổ biến ở mức {mean_val:.1f} Tr/m².")

    with col_charts_30:
        # 3. Donut: Nội thất (Thu nhỏ về tỷ lệ cột 3)
        st.markdown('<p style="color:#A0A0A0; font-size:1.1rem; weight:300;">🛋️ NỘI THẤT</p>', unsafe_allow_html=True)
        df_furn = df_filtered['Furniture'].value_counts().reset_index()
        df_furn.columns = ['Furniture', 'Count']
        fig_donut = px.pie(df_furn, values='Count', names='Furniture', hole=0.7,
                           color='Furniture', color_discrete_map={'có': FPT_ORANGE, 'không': '#444'})
        fig_donut = custom_plotly_theme(fig_donut)
        fig_donut.update_layout(height=400, margin=dict(t=0, l=30, r=30), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_donut, use_container_width=True)
        st.info("Insight: Thị trường đa phần là nhà trống hoặc cơ bản.")

# -------------------------------------------------------------------------
# TAB 2: PHÂN TÍCH CHUYÊN SÂU (EDA)
# -------------------------------------------------------------------------
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    # 1. Scatter: Top 5 Volume Districts
    st.markdown('<p style="color:#A0A0A0; font-size:1.1rem; weight:300;">📐 TƯƠNG QUAN DIỆN TÍCH vs TỔNG GIÁ (Top 5 Volume Khu vực)</p>', unsafe_allow_html=True)
    top5_vol = df_filtered['District'].value_counts().head(5).index.tolist()
    df_scatter = df_filtered[df_filtered['District'].isin(top5_vol)]
    
    fig_scatter = px.scatter(df_scatter, x='Area_m2', y='Price_Million_VND', color='District',
                             color_discrete_sequence=px.colors.diverging.Tealrose_r, opacity=0.7)
    fig_scatter = custom_plotly_theme(fig_scatter)
    fig_scatter.update_layout(height=550)
    fig_scatter.update_xaxes(title="Diện tích (m²)")
    fig_scatter.update_yaxes(title="Tổng giá (Triệu VNĐ)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 2. Boxplots
    st.markdown('<p style="color:#A0A0A0; font-size:1.1rem; weight:300;">📦 BIÊN ĐỘ BIẾN ĐỘNG GIÁ</p>', unsafe_allow_html=True)
    col_box1, col_box2 = st.columns(2, gap="large")
    
    with col_box1:
        # Box: Bedroom
        df_bed = df_filtered[df_filtered['Bedroom'] <= 4]
        df_bed['Bedroom'] = df_bed['Bedroom'].astype(str) + " PN"
        fig_box1 = px.box(df_bed, x='Bedroom', y='Price_Per_M2', color='Bedroom',
                          color_discrete_sequence=px.colors.sequential.YlGnBu)
        fig_box1 = custom_plotly_theme(fig_box1)
        fig_box1.update_layout(height=400, showlegend=False)
        fig_box1.update_yaxes(title="Đơn giá (Tr/m²)")
        st.plotly_chart(fig_box1, use_container_width=True)
        
    with col_box2:
        # Box: Furniture
        fig_box2 = px.box(df_filtered, x='Furniture', y='Price_Per_M2', color='Furniture',
                          color_discrete_map={'có': FPT_ORANGE, 'không': '#444'})
        fig_box2 = custom_plotly_theme(fig_box2)
        fig_box2.update_layout(height=400, showlegend=False)
        fig_box2.update_yaxes(title="Đơn giá (Tr/m²)")
        st.plotly_chart(fig_box2, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 3:  PRICE PREDICTOR (DẤU ẤN ML)
# -------------------------------------------------------------------------
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p style="color:{DARK_TEXT}; font-size:1.5rem; weight:700;">🤖 CÔNG CỤ ĐỊNH GIÁ  [BẢN GỐC ADY201m]</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col_in, col_out = st.columns([4, 6], gap="large")
    
    with col_in:
        with st.form("fpt_prediction_form"):
            st.markdown('<p style="color:#A0A0A0; font-size:1.1rem; weight:300;">📝 Thông số Căn hộ</p>', unsafe_allow_html=True)
            
            p_district = st.selectbox("📍 Vị trí", options=all_districts)
            p_area = st.number_input("📏 Diện tích (m²)", min_value=15.0, max_value=500.0, value=70.0, step=1.0)
            p_furniture = st.selectbox("🛋️ Tình trạng Nội thất", options=['có', 'không'])
            
            st.markdown("<br>", unsafe_allow_html=True)
            # FPT Orange Button
            predict_btn = st.form_submit_button("🔮 BẮT ĐẦU ĐỊNH GIÁ", use_container_width=True)

    with col_out:
        if predict_btn:
            # 1. Tạo input DataFrame
            input_row = pd.DataFrame({
                'Area_m2': [p_area],
                'Furniture': [p_furniture],
                'District': [p_district]
            })
            
            with st.spinner("Đang tính toán dữ liệu phi tuyến (Random Forest)..."):
                try:
                    # 2. Predict (Nhả ra Log)
                    pred_log = model.predict(input_row)[0]
                    
                    # 3. Expm1
                    pred_price_m2 = np.expm1(pred_log)
                    
                    # 4. Tính toán Tỷ VNĐ: (Giá * Diện tích) / 1000
                    pred_total_vnd_billion = (pred_price_m2 * p_area) / 1000
                    
                    # Render outputs
                    st.success("✅ Phân tích Hoàn tất!")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Hai col KPI style mới
                    rcol1, rcol2 = st.columns(2)
                    
                    # Tổng giá
                    rcol1.markdown(f"""
                        <div class="kpi-card" style="border: 2px solid {FPT_ORANGE}; background: rgba(243, 112, 33, 0.05);">
                            {ICONS['target']}
                            <div class="kpi-label">DỰ ĐOÁN TỔNG GIÁ</div>
                            <div class="kpi-value">{pred_total_vnd_billion:.2f} <span class="kpi-unit">Tỷ VNĐ</span></div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Đơn giá
                    rcol2.markdown(f"""
                        <div class="kpi-card">
                            {ICONS['robot']}
                            <div class="kpi-label">ĐƠN GIÁ THAM KHẢO</div>
                            <div class="kpi-value" style="color: {DARK_TEXT};">{pred_price_m2:.1f} <span class="kpi-unit">Tr/m²</span></div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during  inference: {e}")
        else:
            # Placeholder khi chưa ấn nút
            st.markdown(f"""
                <div style="background-color: rgba(255,255,255,0.02); padding: 50px; border-radius:10px; border:1px dashed #444; text-align:center; color:#7f7f7f; margin-top:20px;">
                    Vui lòng nhập thông số bên trái và ấn [🔮 BẮT ĐẦU ĐỊNH GIÁ] để phân tích.
                </div>
            """, unsafe_allow_html=True)

    # --- 8. Explainable : Feature Importance ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<p style="color:#A0A0A0; font-size:1.1rem; weight:300;">🧠 GIẢI THÍCH MÔ HÌNH (Explainable - Feature Importance)</p>', unsafe_allow_html=True)
    
    # Sort for bar chart
    fi_df_sorted = fi_df.sort_values("Importance_Mean", ascending=True).tail(10)
    
    # Scale 'Cividis' for FI
    fig_fi = px.bar(fi_df_sorted, x='Importance_Mean', y='Feature', orientation='h',
                    color='Importance_Mean', color_continuous_scale="cividis",
                    text_auto='.4f')
    fig_fi = custom_plotly_theme(fig_fi)
    fig_fi.update_layout(height=450, coloraxis_showscale=False)
    fig_fi.update_xaxes(title="Độ ảnh hưởng (Suy giảm RMSE Trung bình)")
    fig_fi.update_traces(textposition='outside', textfont=dict(color="#A0A0A0"))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.info("Insight từ Permutation Importance: Vị trí địa lý (Quận 2, Quận 1) đóng vai trò then chốt nhất, áp đảo hoàn toàn các yếu tố khác.")
