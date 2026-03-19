import pandas as pd
import numpy as np
import re
import math

# ==============================================================================
# 1. CÁC HÀM XỬ LÝ 
# ==============================================================================

def clean_price_to_million(x):
    if pd.isna(x): return np.nan
    x = str(x).lower().replace(',', '.')
    match = re.search(r"[-+]?\d*\.?\d+", x)
    if not match: return np.nan
    val = float(match.group())
    if 'tỷ' in x: return val * 1000
    elif 'triệu' in x or 'tr' in x: return val
    elif "nghìn" in x: return val / 1000
    return val

def clean_area(x):
    if pd.isna(x): return np.nan
    x = str(x).lower().replace('m²', '').replace('m2', '').replace('·', '').replace(',', '.').strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", x)
    if match: return float(match.group())
    return np.nan

def extract_district(location):
    x = str(location).lower()
    districts = {
        'tp thủ đức': 'tp thủ đức', 'thủ đức': 'tp thủ đức',
        'quận 12': 'quận 12', 'quận 11': 'quận 11', 'quận 10': 'quận 10',
        'quận 1': 'quận 1', 'quận 2': 'quận 2', 'quận 3': 'quận 3', 
        'quận 4': 'quận 4', 'quận 5': 'quận 5', 'quận 6': 'quận 6', 
        'quận 7': 'quận 7', 'quận 8': 'quận 8', 'quận 9': 'quận 9', 
        'bình thạnh': 'bình thạnh', 'gò vấp': 'gò vấp', 'phú nhuận': 'phú nhuận', 
        'tân bình': 'tân bình', 'tân phú': 'tân phú', 'bình tân': 'bình tân', 
        'nhà bè': 'nhà bè', 'hóc môn': 'hóc môn', 'bình chánh': 'bình chánh', 
        'củ chi': 'củ chi', 'cần giờ': 'cần giờ'
    }
    parts = [p.strip() for p in x.split(',')]
    for part in reversed(parts):
        for key, val in districts.items():
            if key in part: return val
    return 'khác'

def extract_from_title(row, col_name, keywords):
    if pd.notna(row[col_name]) and row[col_name] != 'N/A':
        return row[col_name]
    title = str(row['Title']).lower()
    found_numbers = []
    for kw in keywords:
        pattern = rf"([\d\-\.,/ ]+)\s*{kw}"
        matches = re.findall(pattern, title)
        for m in matches:
            nums = re.findall(r'\d+', m)
            found_numbers.extend([int(n) for n in nums])
    if found_numbers:
        avg_val = sum(found_numbers) / len(found_numbers)
        return math.ceil(avg_val)
    return np.nan

def check_furniture(title):
    t = " ".join(str(title).lower().split())
    negative_pattern = r"(không|ko|kng|chưa|no)\s*[,:.]*\s*(?:có)?\s*(nội\s*thất|nt|furniture)"
    if re.search(negative_pattern, t):
        return 'không'
    hard_negatives = ['nhà trống', 'giao thô', 'nhà thô', 'cơ bản', 'nhà cơ bản']
    for kw in hard_negatives:
        if kw in t:
            return 'không'
    positive_pattern = r"(nội\s*thất|nt|furniture|full\s*option|đủ\s*đồ|tặng\s*hết|full\s*đồ)"
    if re.search(positive_pattern, t):
        return 'có'
    return 'không'

# ==============================================================================
# 2. CHƯƠNG TRÌNH CHÍNH (UPDATE FILTERING)
# ==============================================================================
if __name__ == "__main__":
    print(">>> Đang xử lý dữ liệu...")
    
    df = pd.read_csv('raw.csv')
    raw_rows = len(df)
    
    # 1. Xử lý sơ bộ
    df.drop_duplicates(subset=['Title', 'Price', 'Area', 'Location'], inplace=True)
    df = df[~df['Price'].astype(str).str.lower().str.contains('thỏa thuận|liên hệ', na=False)]
    df['Title'] = df['Title'].astype(str).str.replace('\n', ' ').str.replace('\r', '').str.strip().str.lower()
    
    # 2. Làm sạch Giá, Diện tích, Quận
    df['Price_Million_VND'] = df['Price'].apply(clean_price_to_million)
    df['Area_m2'] = df['Area'].apply(clean_area)
    df['District'] = df['Location'].apply(extract_district)
    
    df['Price_Per_M2'] = df['Price_Million_VND'] / df['Area_m2']
    df['Price_Per_M2'] = df['Price_Per_M2'].round(2)
    
    # Lọc sơ bộ (chỉ cần có Giá và Diện tích là giữ lại để tính toán tiếp)
    df_clean = df.dropna(subset=['Price_Million_VND', 'Area_m2']).copy()

    # 3. Điền khuyết & Feature Engineering
    df_clean['Bedroom'] = df_clean.apply(lambda row: extract_from_title(row, 'Bedroom', ['pn', 'phòng ngủ', 'phòng']), axis=1)
    df_clean['Toilet'] = df_clean.apply(lambda row: extract_from_title(row, 'Toilet', ['wc', 'toilet', 'vs']), axis=1)
    df_clean['Furniture'] = df_clean['Title'].apply(check_furniture)

    # 4. Chọn cột
    desired_columns = [
        'Title', 'Price_Million_VND', 'Area_m2', 'Price_Per_M2', 
        'Bedroom', 'Toilet', 'Furniture', 'District', 'Link'
    ]
    
    try:
        df_final = df_clean[desired_columns].copy()
    except KeyError as e:
        print(f"Lỗi thiếu cột: {e}, fallback về df_clean")
        df_final = df_clean.copy()
    
    # ==============================================================================
    # 5. BƯỚC QUAN TRỌNG: LỌC "FULL THÔNG TIN" (STRICT MODE)
    # ==============================================================================
    print("\n>>> Đang thực hiện lọc nghiêm ngặt (Strict Filtering)...")
    count_before = len(df_final)
    
    # Hàm dropna() sẽ xóa bất kỳ dòng nào có ít nhất 1 ô bị NaN (trống)
    # Chủ yếu sẽ xóa các dòng không tìm thấy Bedroom hoặc Toilet trong Title
    df_final.dropna(inplace=True)
    
    count_after = len(df_final)
    print(f"   - Số dòng ban đầu (đã có giá/diện tích): {count_before}")
    print(f"   - Số dòng đầy đủ Full cột (có cả PN/WC): {count_after}")
    print(f"   - Đã loại bỏ: {count_before - count_after} dòng thiếu thông tin.")
    
    # Convert Bedroom/Toilet sang số nguyên (int) cho đẹp (vì sau khi dropna không còn NaN nữa)
    df_final['Bedroom'] = df_final['Bedroom'].astype(int)
    df_final['Toilet'] = df_final['Toilet'].astype(int)

    # 6. Lưu file
    print("\n5 dòng đầu tiên của dữ liệu Full Option:")
    print(df_final.head())

    output_file = 'stage1_clean_full.csv'
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f">>> [OK] Đã lưu file sạch vào: {output_file}")


