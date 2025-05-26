import streamlit as st
from dotenv import load_dotenv
# import os
# utils_db akan diimpor dan menginisialisasi tabel + admin jika belum ada
# import utils_db # Ini akan menjalankan create_tables() dan add_admin_user_if_not_exists()
from utils_rag import initialize_rag_pipeline

# Muat variabel lingkungan dari .env
# Ini sebaiknya dilakukan di awal sebelum impor lain yang mungkin membutuhkannya
load_dotenv(override=True)

st.set_page_config(
    page_title="Info Gizi & KesMas",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main_app():
    st.sidebar.title(f"Selamat Datang, {st.session_state.get('username', 'Pengguna Tamu')}!")
    
    if st.session_state.get('logged_in_admin', False):
        st.sidebar.info("Anda login sebagai Admin.")
        if st.sidebar.button("Logout Admin", key="logout_button_main_app"):
            # Hapus semua state sesi terkait login
            keys_to_delete = ['logged_in_admin', 'username']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            # Hapus juga state RAG agar dimuat ulang jika perlu (opsional)
            # if 'rag_initialized' in st.session_state: del st.session_state['rag_initialized']
            # if 'llm' in st.session_state: del st.session_state['llm'] # dst.
            st.success("Logout berhasil.")
            st.rerun() # Ganti st.experimental_rerun dengan st.rerun
    else:
        st.sidebar.info("Anda adalah Pengguna Tamu.")
        # Tombol login bisa juga ditaruh di sini jika diinginkan
        # if st.sidebar.button("Login Admin"):
        #     st.switch_page("pages/03_Admin.py") # Navigasi ke halaman Admin untuk login

    st.title("🍎 Sistem Informasi Gizi dan Kesehatan Masyarakat")
    st.markdown("Navigasi melalui menu di bilah sisi kiri.")

    # Inisialisasi pipeline RAG jika belum dilakukan
    # Ini memastikan model dll. dimuat saat aplikasi dimulai, untuk halaman mana pun yang mungkin membutuhkannya.
    if 'rag_initialized_status' not in st.session_state:
        st.session_state.rag_initialized_status = "pending" # pending, success, failed

    if st.session_state.rag_initialized_status == "pending":
        with st.spinner("Memuat model AI dan basis pengetahuan... Ini mungkin memerlukan beberapa saat pada pemuatan pertama."):
            success = initialize_rag_pipeline() # Fungsi ini akan menampilkan log/status sendiri
            if success:
                st.session_state.rag_initialized_status = "success"
                # st.sidebar.success("Komponen AI Siap!") # Sudah ada di initialize_rag_pipeline
            else:
                st.session_state.rag_initialized_status = "failed"
                # st.sidebar.error("Beberapa komponen AI gagal dimuat.") # Sudah ada di initialize_rag_pipeline
            st.rerun() # Untuk merefleksikan status sidebar

    if st.session_state.rag_initialized_status == "failed":
         st.warning("Beberapa komponen AI gagal dimuat. Fitur chatbot mungkin tidak berfungsi dengan baik.")
    elif st.session_state.rag_initialized_status == "success":
        st.success("Sistem AI siap digunakan pada halaman Chatbot.")


    st.markdown(
        """
        Selamat datang di Sistem Informasi Gizi dan Kesehatan Masyarakat (SIGKesMas).
        Aplikasi ini menyediakan berbagai informasi dan layanan terkait gizi dan kesehatan.

        **Fitur Utama:**
        - **Dashboard**: Menampilkan informasi umum statis mengenai gizi dan kesehatan.
        - **Chatbot GiziAI**: Berinteraksi dengan asisten AI untuk mendapatkan jawaban atas pertanyaan spesifik Anda berdasarkan basis pengetahuan yang tersedia.
        - **Panel Admin**: (Khusus Admin) Mengelola file-file yang menjadi basis pengetahuan untuk chatbot.

        Silakan gunakan menu navigasi di sebelah kiri untuk mengakses berbagai fitur aplikasi.
        """
    )
    st.info("Pastikan model GGUF (`sahabatAI-9B-GiziAI-v1.i1-Q4_K_M.gguf`) telah diunduh dan ditempatkan di folder `model/` jika unduhan otomatis gagal.")


if __name__ == "__main__":
    # Inisialisasi state sesi jika belum ada
    if 'logged_in_admin' not in st.session_state:
        st.session_state['logged_in_admin'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = "Pengguna Tamu"
    
    main_app()