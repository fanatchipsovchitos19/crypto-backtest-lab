import streamlit as st
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.title("Тест PDF")

if st.button("Создать PDF"):
    st.write("Кнопка нажата!")
    
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "Test PDF", ln=True)
    
    path = "reports/test.pdf"
    Path("reports").mkdir(exist_ok=True)
    pdf.output(path)
    
    st.success(f"PDF создан: {path}")
    
    with open(path, "rb") as f:
        st.download_button("Скачать", f, "test.pdf")

st.write("Конец страницы")