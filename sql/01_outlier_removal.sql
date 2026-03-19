-- =======================================================
-- PROJECT: REAL ESTATE PRICE ANALYSIS - HCMC
-- GIAI ĐOẠN 2: DATA CLEANING & OUTLIER DETECTION
-- =======================================================
USE Real_Estate_HCM;
GO

-- 1. KIỂM TRA SỐ LƯỢNG TRƯỚC KHI LỌC
SELECT COUNT(*) AS So_Luong_Ban_Dau FROM Apartments;

-- 2. XEM TRƯỚC CÁC DÒNG SẼ BỊ XÓA (OUTLIERS)
SELECT * FROM Apartments
WHERE Price_Per_M2 < 10 OR Price_Per_M2 > 500
   OR Area_m2 < 15 OR Area_m2 > 500;

-- 3. THỰC HIỆN LỆNH XÓA (DELETE)
DELETE FROM Apartments
WHERE Price_Per_M2 < 10 OR Price_Per_M2 > 500
   OR Area_m2 < 15 OR Area_m2 > 500;

-- 4. KIỂM TRA LẠI KẾT QUẢ
SELECT COUNT(*) AS So_Luong_Con_Lai FROM Apartments;

-- 5. XEM DỮ LIỆU SẠCH (TOP 20)
SELECT TOP 20 * FROM Apartments ORDER BY Price_Per_M2 DESC;