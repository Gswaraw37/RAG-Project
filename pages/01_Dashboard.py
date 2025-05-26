import streamlit as st
import os

st.set_page_config(page_title="Dashboard Gizi & KesMas", layout="wide")

st.title("📊 Dashboard Informasi Gizi dan Kesehatan Masyarakat")
st.markdown("Informasi umum dan statis seputar gizi dan kesehatan.")

# Path ke file HTML statis Anda
# Diasumsikan folder static berada satu level di atas folder pages
# Dari `pages/01_Dashboard.py` ke `static/dashboard.html` adalah `../static/dashboard.html`
# Path absolut lebih aman:
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # root proyek
static_html_path = os.path.join(base_dir, "static", "dashboard.html")


try:
    with open(static_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    # Menggunakan st.components.v1.html untuk menampilkan konten HTML
    st.components.v1.html(html_content, height=800, scrolling=True)
except FileNotFoundError:
    st.error(f"File dashboard.html tidak ditemukan di: {static_html_path}")
    st.markdown(f"""
        Pastikan file `dashboard.html` ada di dalam folder `static` di root proyek Anda.
        Lokasi yang diharapkan: `{static_html_path}`
    """)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat konten dashboard: {e}")

st.sidebar.header("Navigasi")
# Anda bisa menambahkan link atau tombol navigasi lain di sidebar jika perlu