import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid")

INPUT_PATH = "data/stage2_final.csv"
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# =========================================================
# 1. Phân phối đơn giá, diện tích và tổng giá
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df["Price_Per_M2"], kde=True, ax=axes[0], color="blue", bins=50)
axes[0].set_title("Phân phối Đơn giá (Price_Per_M2)")
axes[0].set_xlabel("Triệu VNĐ / m2")

sns.histplot(df["Area_m2"], kde=True, ax=axes[1], color="green", bins=50)
axes[1].set_title("Phân phối Diện tích (Area_m2)")
axes[1].set_xlabel("m2")

sns.histplot(df["Price_Million_VND"], kde=True, ax=axes[2], color="red", bins=50)
axes[2].set_title("Phân phối Tổng giá (Price_Million_VND)")
axes[2].set_xlabel("Triệu VNĐ")

plt.tight_layout()
plt.show()

# =========================================================
# 2. Phân phối Price_Per_M2 trước và sau log-transform
# =========================================================
df["Log_Price_Per_M2"] = np.log1p(df["Price_Per_M2"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df["Price_Per_M2"], kde=True, ax=axes[0], color="blue", bins=50)
axes[0].set_title("Phân phối gốc của Đơn giá (Price_Per_M2)")
axes[0].set_xlabel("Triệu VNĐ / m2")

sns.histplot(df["Log_Price_Per_M2"], kde=True, ax=axes[1], color="purple", bins=50)
axes[1].set_title("Phân phối Đơn giá sau Log-transform")
axes[1].set_xlabel("log(1 + Price_Per_M2)")

plt.tight_layout()
plt.show()

# =========================================================
# 3. Density theo nội thất và số phòng ngủ
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.kdeplot(
    data=df,
    y="Price_Per_M2",
    hue="Furniture",
    fill=True,
    ax=axes[0],
    common_norm=False,
    palette="Set2"
)
axes[0].set_title("Density: Đơn giá theo Tình trạng Nội thất")
axes[0].set_ylabel("Đơn giá (Triệu VNĐ / m2)")
axes[0].set_xlabel("Density")

df_bed = df[df["Bedroom"].isin([1, 2, 3])]
sns.kdeplot(
    data=df_bed,
    y="Price_Per_M2",
    hue="Bedroom",
    fill=True,
    ax=axes[1],
    common_norm=False,
    palette="tab10"
)
axes[1].set_title("Density: Đơn giá theo Số Phòng ngủ (1-3 PN)")
axes[1].set_ylabel("Đơn giá (Triệu VNĐ / m2)")
axes[1].set_xlabel("Density")

plt.tight_layout()
plt.show()

# =========================================================
# 4. Mối quan hệ giữa diện tích và đơn giá
# =========================================================
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df,
    x="Area_m2",
    y="Price_Per_M2",
    scatter_kws={"alpha": 0.3, "color": "blue"},
    line_kws={"color": "red", "linewidth": 2}
)
plt.title("Mối quan hệ giữa Diện tích và Đơn giá")
plt.tight_layout()
plt.show()

# =========================================================
# 5. Quan hệ log-log giữa diện tích và đơn giá
# =========================================================
df["Log_Area_m2"] = np.log1p(df["Area_m2"])

plt.figure(figsize=(8, 5))
sns.regplot(
    data=df,
    x="Log_Area_m2",
    y="Log_Price_Per_M2",
    scatter_kws={"alpha": 0.3, "color": "blue"},
    line_kws={"color": "red", "linewidth": 2}
)
plt.title("Quan hệ log-log giữa Diện tích và Đơn giá")
plt.xlabel("log(1 + Area_m2)")
plt.ylabel("log(1 + Price_Per_M2)")
plt.tight_layout()
plt.show()

# =========================================================
# 6. So sánh phân phối đơn giá theo quận, số phòng ngủ và nội thất
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

top7_dist = df["District"].value_counts().nlargest(7).index
sns.boxplot(
    data=df[df["District"].isin(top7_dist)],
    x="District",
    y="Price_Per_M2",
    hue="District",
    legend=False,
    ax=axes[0],
    palette="Set2",
    whis=6.5
)
axes[0].tick_params(axis="x", rotation=45)
axes[0].set_title("Đơn giá theo Top 7 Quận")

sns.boxplot(
    data=df[df["Bedroom"] <= 4],
    x="Bedroom",
    y="Price_Per_M2",
    hue="Bedroom",
    legend=False,
    ax=axes[1],
    palette="pastel",
    whis=6.5
)
axes[1].set_title("Đơn giá theo Số Phòng Ngủ")

sns.boxplot(
    data=df,
    x="Furniture",
    y="Price_Per_M2",
    hue="Furniture",
    legend=False,
    ax=axes[2],
    palette="Set3",
    whis=6.5
)
axes[2].set_title("Đơn giá theo Nội Thất")

plt.tight_layout()
plt.show()

# =========================================================
# 7. Ma trận tương quan Pearson
# =========================================================
plt.figure(figsize=(8, 6))
cols_to_corr = ["Price_Million_VND", "Price_Per_M2", "Area_m2", "Bedroom", "Toilet"]
corr_matrix = df[cols_to_corr].corr(method="pearson")

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="RdBu_r",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)
plt.title("Correlation Heatmap (Ma trận tương quan Pearson)")
plt.tight_layout()
plt.show()

# =========================================================
# 8. VIF
# =========================================================
features = ["Area_m2", "Bedroom", "Toilet"]
X = df[features].dropna().copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.dropna()

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

def vif_interpret(v):
    if v < 5:
        return "Chấp nhận được"
    elif v < 10:
        return "Đa cộng tuyến đáng chú ý"
    else:
        return "Đa cộng tuyến nghiêm trọng"

vif_data["Interpretation"] = vif_data["VIF"].apply(vif_interpret)
vif_data["VIF"] = vif_data["VIF"].round(4)

print(vif_data)

os.makedirs("outputs/tables", exist_ok=True)
vif_data.to_csv("outputs/tables/vif_result.csv", index=False, encoding="utf-8-sig")