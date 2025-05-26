import streamlit as st
from dotenv import load_dotenv
import os
import utils_db
from utils_rag import initialize_rag_components

# Muat variabel lingkungan dari .env
load_dotenv(override=True)

st.set_page_config(
    page_title="Info Gizi & KesMas",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main_app_content(): # Mengganti nama fungsi agar lebih jelas
    st.sidebar.title(f"Selamat Datang, {st.session_state.get('username', 'Pengguna Tamu')}!")
    
    if st.session_state.get('logged_in_admin', False):
        st.sidebar.info("Anda login sebagai Admin.")
        if st.sidebar.button("Logout Admin", key="logout_button_main_app"):
            keys_to_delete = ['logged_in_admin', 'username', 'rag_initialized_status', 'initial_greeting_displayed', 'chat_history_display', 'session_id']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Logout berhasil. Silakan muat ulang halaman jika diperlukan.")
            st.rerun()
    else:
        st.sidebar.info("Anda adalah Pengguna Tamu.")

    st.title("🍎 Sistem Informasi Gizi dan Kesehatan Masyarakat")
    st.markdown("Navigasi melalui menu di bilah sisi kiri.")

    # Inisialisasi pipeline RAG jika belum dilakukan
    if 'rag_initialized_status' not in st.session_state:
        st.session_state.rag_initialized_status = "pending"

    # Selalu coba inisialisasi jika statusnya pending, atau jika diminta oleh tombol
    # Tombol ini bisa ditambahkan jika ingin ada cara manual re-init dari UI utama
    # if st.sidebar.button("Inisialisasi Ulang Sistem AI", key="reinit_ai_main"):
    #    st.session_state.rag_initialized_status = "pending"
    #    st.experimental_rerun() # atau st.rerun()

    if st.session_state.rag_initialized_status == "pending":
        # Tidak menggunakan st.spinner di sini karena initialize_rag_components sudah punya st.write/st.spinner internal
        # st.write("Memulai inisialisasi komponen RAG dari app.py...") # Untuk debug jika perlu
        success = initialize_rag_components() # Fungsi ini sekarang mengembalikan True/False
        if success:
            st.session_state.rag_initialized_status = "success"
            # st.rerun() # Mungkin tidak perlu rerun di sini, biarkan UI update alami
        else:
            st.session_state.rag_initialized_status = "failed"
            # st.rerun() # Mungkin tidak perlu rerun di sini

    # Tampilkan status setelah upaya inisialisasi
    if st.session_state.rag_initialized_status == "failed":
        st.sidebar.error("Komponen AI gagal dimuat. Chatbot mungkin tidak berfungsi.")
    elif st.session_state.rag_initialized_status == "success":
        st.sidebar.success("Komponen AI Siap!")
    else: # Pending (jika tidak ada rerun setelah status diubah)
        st.sidebar.info("Status inisialisasi AI: Tertunda.")


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
    llm_model_path_env = os.getenv("LLM_MODEL_PATH", "Belum dikonfigurasi")
    st.info(f"Pastikan model GGUF telah dikonfigurasi di LLM_MODEL_PATH (saat ini: {llm_model_path_env}) dan file model ada jika path tersebut lokal.")

if __name__ == "__main__":
    if 'logged_in_admin' not in st.session_state:
        st.session_state['logged_in_admin'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = "Pengguna Tamu"
    
    main_app_content()