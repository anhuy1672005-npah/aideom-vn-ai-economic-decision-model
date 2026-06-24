import streamlit as st
import pandas as pd
from utils.style import load_css, hero, card

st.set_page_config(
    page_title="AIDEOM-VN Webapp",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

hero(
    title="🇻🇳 AIDEOM-VN Webapp",
    subtitle="Webapp mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI. Hệ thống đã hoàn thành đầy đủ 12 bài thực hành, gồm: mô hình toán học, dữ liệu, tính toán Python, trực quan hóa kết quả và phân tích chính sách.",
    badges=["Hoàn thành 12/12", "Python", "Streamlit", "AIDEOM-VN"]
)

# =========================
# 1. THỐNG KÊ TỔNG QUAN
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Số bài thực hành", "12")
with col2:
    st.metric("Đã hoàn thành", "12/12")
with col3:
    st.metric("Tiến độ", "100%")
with col4:
    st.metric("Dữ liệu", "2020-2025")

st.progress(1.0, text="Tiến độ hoàn thành toàn bộ hệ thống: 12/12 bài")

st.markdown("---")

# =========================
# 2. TỔNG QUAN HỆ THỐNG
# =========================

st.subheader("📌 Tổng quan hệ thống")

c1, c2 = st.columns(2)

with c1:
    card(
        "Mục tiêu webapp",
        """
        Webapp hỗ trợ chạy đầy đủ các mô hình ra quyết định bằng Python,
        hiển thị bảng kết quả, biểu đồ, phân tích chính sách và xuất file
        phục vụ nộp bài cuối kỳ.
        """
    )

with c2:
    card(
        "Tình trạng hoàn thành",
        """
        Toàn bộ 12 bài thực hành đã được xây dựng trong hệ thống.
        Người dùng có thể chọn từng bài ở thanh menu bên trái để xem mô hình,
        dữ liệu, kết quả tính toán, biểu đồ và phần hàm ý chính sách.
        """
    )

st.subheader("🧭 Bản đồ 12 bài thực hành")

# =========================
# 3. BẢN ĐỒ 12 BÀI
# =========================

data = pd.DataFrame([
    ["Dễ", "Bài 1", "Cobb-Douglas mở rộng với AI và số hóa", "Đã hoàn thành"],
    ["Dễ", "Bài 2", "LP phân bổ ngân sách số", "Đã hoàn thành"],
    ["Dễ", "Bài 3", "Chỉ số ưu tiên ngành", "Đã hoàn thành"],
    ["Trung bình", "Bài 4", "LP phân bổ ngân sách theo ngành - vùng", "Đã hoàn thành"],
    ["Trung bình", "Bài 5", "MIP lựa chọn dự án chuyển đổi số", "Đã hoàn thành"],
    ["Trung bình", "Bài 6", "TOPSIS xếp hạng vùng kinh tế", "Đã hoàn thành"],
    ["Khá khó", "Bài 7", "Tối ưu đa mục tiêu Pareto NSGA-II", "Đã hoàn thành"],
    ["Khá khó", "Bài 8", "Tối ưu động 2026-2035", "Đã hoàn thành"],
    ["Khá khó", "Bài 9", "Tác động AI tới thị trường lao động", "Đã hoàn thành"],
    ["Khó", "Bài 10", "Quy hoạch ngẫu nhiên hai giai đoạn", "Đã hoàn thành"],
    ["Khó", "Bài 11", "Q-learning cho chính sách kinh tế thích nghi", "Đã hoàn thành"],
    ["Khó", "Bài 12", "Dashboard tích hợp AIDEOM-VN", "Đã hoàn thành"],
], columns=["Cấp độ", "Bài", "Tên bài", "Trạng thái"])

st.dataframe(data, width="stretch", hide_index=True)

st.markdown("---")

# =========================
# 4. CẤU TRÚC MỖI BÀI
# =========================

st.subheader("🧩 Cấu trúc mỗi bài")

col_a, col_b, col_c = st.columns(3)

with col_a:
    card(
        "1. Mô hình",
        "Trình bày bài toán, công thức, biến quyết định, tham số và ràng buộc."
    )

with col_b:
    card(
        "2. Tính toán",
        "Chạy Python với numpy, pandas, scipy, pulp, scikit-learn hoặc các thư viện phù hợp."
    )

with col_c:
    card(
        "3. Chính sách",
        "Diễn giải kết quả, đánh giá đánh đổi và đề xuất hàm ý chính sách cho Việt Nam."
    )

st.markdown("---")

# =========================
# 5. GỢI Ý SỬ DỤNG
# =========================

st.subheader("🚀 Hướng dẫn xem bài")

card(
    "Cách xem kết quả",
    """
    Chọn bài cần xem ở thanh menu bên trái. Mỗi trang bài đã được thiết kế để có thể
    trình bày trực tiếp khi báo cáo: có phần mục tiêu, mô hình, dữ liệu đầu vào,
    kết quả tính toán, biểu đồ minh họa, nhận xét và kết luận chính sách.
    """
)
