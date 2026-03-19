import os
import sys
import io

# Fix lỗi "No module named distutils" trên Python 3.13
try:
    import setuptools
except ImportError:
    print(" Thiếu thư viện setuptools. Vui lòng chạy: pip install setuptools")
    sys.exit(1)

import time
import random
import csv

# SỬ DỤNG THƯ VIỆN UNDETECTED CHROMEDRIVER
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException

# Configuration
BASE_URL = "https://batdongsan.com.vn/ban-can-ho-chung-cu-tp-hcm"
CSV_FILE = "batdongsan_hcm.csv"
START_PAGE = 6
END_PAGE = 950  

# --- HÀM KHỞI TẠO DRIVER ---
def setup_driver():
    print(" Đang khởi tạo lại Chrome (New Session)...")
    options = uc.ChromeOptions()
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-first-run")
    
    # Fix thêm lỗi phiên bản cho Python 3.13
    for _ in range(3):
        try:
            # version_main=None giúp tự động tìm Chrome cài trên máy
            driver = uc.Chrome(options=options, version_main=None)
            driver.set_page_load_timeout(30)
            return driver
        except Exception as e:
            print(f" Lỗi mở Chrome: {e}. Thử lại...")
            time.sleep(2)
    return None

# --- HÀM CÀO DỮ LIỆU ---
def get_listings_from_page(driver, url):
    driver.get(url)
    
    # 1. Kiểm tra Cloudflare
    title = driver.title
    if "Just a moment" in title or "bảo mật" in title.lower():
        print(f" Phát hiện màn hình chặn tại {url}")
        time.sleep(10)
    
    # 2. Chờ thẻ tin đăng
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".js__card")))
    except TimeoutException:
        raise Exception("Timeout: Không tìm thấy thẻ tin đăng .js__card")

    # 3. Cuộn trang mồi
    driver.execute_script("window.scrollTo(0, 1000);")
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    listings = driver.find_elements(By.CSS_SELECTOR, ".js__card")
    page_data = []

    for item in listings:
        try:
            def get_text(selector):
                try: return item.find_element(By.CSS_SELECTOR, selector).get_attribute("textContent").strip()
                except: return "N/A"

            title = get_text(".js__card-title")
            
            link = "N/A"
            try:
                link_elem = item.find_element(By.CSS_SELECTOR, ".js__card-title")
                if link_elem.tag_name == 'a': link = link_elem.get_attribute('href')
                else: link = item.find_element(By.TAG_NAME, "a").get_attribute('href')
                if link and link.startswith("/"): link = "https://batdongsan.com.vn" + link
            except: pass

            location = "N/A"
            try:
                loc_raw = item.find_element(By.CSS_SELECTOR, ".re__card-location").get_attribute("textContent")
                location = loc_raw.replace('·', '').strip()
            except: pass

            page_data.append({
                "Title": title, "Price": get_text(".re__card-config-price"), 
                "Area": get_text(".re__card-config-area"), "Price/m2": get_text(".re__card-config-price_per_m2").replace('·', '').strip(),
                "Bedroom": get_text(".re__card-config-bedroom"), "Toilet": get_text(".re__card-config-toilet"),
                "Location": location, "Link": link
            })
        except: continue
        
    return page_data

def save_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    fieldnames = ["Title", "Price", "Area", "Price/m2", "Bedroom", "Toilet", "Location", "Link"]
    with open(filename, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerows(data)

# --- PHẦN CHẠY CHÍNH ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if __name__ == "__main__":
    print("--- BẮT ĐẦU CHƯƠNG TRÌNH ---")
    driver = setup_driver()
    if not driver:
        print("Không thể mở Chrome. Dừng chương trình.")
        sys.exit()

    total_listings = 0
    current_page = START_PAGE

    try:
        driver.get(BASE_URL)
        time.sleep(3)

        while current_page <= END_PAGE:
            url = BASE_URL if current_page == 1 else f"{BASE_URL}/p{current_page}"
            print(f"\n--- Đang xử lý trang {current_page} ---")
            
            max_retries = 3
            success = False

            for attempt in range(max_retries):
                try:
                    data = get_listings_from_page(driver, url)
                    
                    if data and len(data) > 0:
                        save_to_csv(data, CSV_FILE)
                        count = len(data)
                        total_listings += count
                        print(f"Trang {current_page}: Lấy được {count} tin. Tổng cộng: {total_listings}")
                        success = True
                        break 
                    else:
                        raise Exception("Lấy về 0 tin")

                except Exception as e:
                    print(f"Lỗi ở trang {current_page} (Lần thử {attempt+1}/{max_retries}): {e}")
                    print("Đang khởi động lại Driver...")
                    try: driver.quit()
                    except: pass
                    driver = setup_driver()
                    time.sleep(3)
            
            if success:
                current_page += 1
                sleep_time = random.uniform(4, 7)
                print(f"Nghỉ {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(f"Thất bại trang {current_page}. Bỏ qua.")
                current_page += 1

    except KeyboardInterrupt:
        print("\nĐã dừng chương trình.")
    finally:
        try: driver.quit()
        except: pass
        print(f"\nHOÀN TẤT! File lưu tại: {os.path.abspath(CSV_FILE)}")