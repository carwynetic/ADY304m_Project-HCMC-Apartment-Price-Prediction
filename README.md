# ADY304m_Project-HCMC-Apartment-Price-Prediction
# 🏙️ HCMC Apartment Price Prediction (Dự đoán giá căn hộ tại TP.HCM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![Selenium](https://img.shields.io/badge/Selenium-Web%20Scraping-yellow.svg)](https://www.selenium.dev/)

[dashboards/Dashboards.png]

Đây là kho mã nguồn chính thức của đồ án **"Phân tích và dự đoán đơn giá căn hộ tại TP.HCM bằng các mô hình học máy"**. Dự án xây dựng một đường ống dữ liệu (Data Pipeline) hoàn chỉnh từ khâu thu thập dữ liệu tự động (Web Scraping), tiền xử lý, phân tích khám phá (EDA) đến huấn luyện và đánh giá mô hình học máy.

---

## 🎯 Mục tiêu dự án
* Thu thập và làm sạch dữ liệu thực tế từ thị trường bất động sản trực tuyến (Batdongsan.com.vn).
* Phân tích các yếu tố ảnh hưởng đến đơn giá căn hộ tại TP.HCM (vị trí, diện tích, nội thất).
* Đánh giá và so sánh 10 thuật toán học máy (Linear Models, KNN, SVM, Tree-based Ensembles).
* Xây dựng mô hình dự báo tối ưu và giải thích mức độ đóng góp của từng đặc trưng (Feature Importance).

---

## 📂 Cấu trúc Kho lưu trữ (Repository Structure)

Dự án được tổ chức theo tiêu chuẩn khoa học dữ liệu hiện đại:

```text
ADY304m_Project_HCMC_Apartment/
│
├── .gitignore                      # Cấu hình ẩn file rác
├── README.md                       # Tài liệu dự án (Đã có bản xịn)
├── requirements.txt                # Danh sách thư viện (Cần có: streamlit, plotly, scikit-learn, joblib...)
│
├── data/                           
│   ├── raw.csv                     
│   ├── stage1_clean_full.csv       
│   └── stage2_final.csv            
│
├── scripts/                        
│   ├── 01_scraper.py               
│   ├── 02_data_cleaning.py         
│   ├── 03_eda_analysis.py          
│   ├── 04_model_training.py        
│   └── 05_permutation_importance.py
│
├── sql/                            
│   └── 01_outlier_removal.sql      
│
├── dashboards/                     
│   └── app.py                      # File code Dashboard (Chạy bằng lệnh streamlit run)
│
└── outputs/                        
    ├── metadata/
    │   └── experiment_summary.json 
    ├── models/
    │   └── best_model.pkl          # File model để Tab 3 của Dashboard gọi lên dự đoán
    └── tables/
        ├── best_model_top10_details.csv                 
        ├── model_comparison_results.csv                 
        ├── ols_top_coefficients_for_paper.csv           
        └── random_forest_permutation_importance_top10.csv # File giải thích AI sinh ra từ script 05
```
## 🚀 Hướng dẫn Cài đặt & Sử dụng (How to Run)

Bước 1: Clone kho lưu trữ

- Bash git clone [https://github.com/your-username/ADY304m_Project_HCMC_Apartment.git](https://github.com/carwynetic/ADY304m_Project_HCMC_Apartment.git) cd ADY304m_Project_HCMC_Apartment

Bước 2: Cài đặt thư viện

- Bash pip install -r requirements.txt

Bước 3: Thực thi quy trình (Pipeline)
Chạy các script theo thứ tự vòng đời dữ liệu:

- python scripts/01_scraper.py (Tùy chọn: Chạy để cào dữ liệu mới)

- python scripts/02_data_cleaning.py (Làm sạch lần 1)

- Chạy script SQL trong thư mục sql/ hoặc chạy trực tiếp bằng tool quản trị DB (Làm sạch lần 2 - Loại bỏ Outliers).

- python scripts/03_eda_analysis.py (Xuất biểu đồ phân tích)

- python scripts/04_model_training.py (Train và so sánh mô hình)

- python scripts/05_permutation_importance.py (Phân tích tính minh bạch của mô hình)

Bước 4: Khởi chạy Dashboard
Bash
- Di chuyển vào thư mục dashboards và chạy file app.py (tuỳ thuộc thư viện bạn dùng: Streamlit/Dash)
- Ví dụ với Streamlit:
  streamlit run dashboards/app.py

## 📊 Tóm tắt Kết quả Thực nghiệm

Sau khi làm sạch, tập dữ liệu giữ lại 16,520 quan sát hợp lệ. Thử nghiệm trên 10 thuật toán Hồi quy khác nhau cho thấy nhóm mô hình phi tuyến (Tree-based Ensemble) hoàn toàn vượt trội so với các mô hình tuyến tính truyền thống.

Mô hình tốt nhất: Random Forest Regression

MAE: 20.17 (Triệu VNĐ/m²)

RMSE: 38.51 (Triệu VNĐ/m²)

R² Score: 0.5413

MAPE: 23.04%

Diễn giải mô hình (Permutation Importance):

Phân tích độ suy giảm hiệu năng (RMSE) khi hoán vị đặc trưng chỉ ra rằng Vị trí địa lý (đặc biệt là Quận 2 và Quận 1) và Quy mô diện tích là hai rường cột định hình cấu trúc giá căn hộ tại TP.HCM. Tình trạng nội thất chỉ đóng vai trò thứ yếu (đóng góp ~3%).

## 👥 Nhóm Tác giả (Authors)
Nguyễn Đoàn Bảo Phúc (SE201883) - Trưởng nhóm / Kỹ sư Dữ liệu

Trần Nguyễn Minh Hải (SE203718)

Hà Phước Khôi (SE203349)

Đơn vị: Đại học FPT TP.HCM | Ngành Trí tuệ Nhân tạo (AI).
***
