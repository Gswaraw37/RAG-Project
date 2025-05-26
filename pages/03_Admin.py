import streamlit as st
import os
from werkzeug.utils import secure_filename # Untuk mengamankan nama file
from utils_db import store_file_metadata, update_file_status, verify_admin, get_active_knowledge_files
from utils_rag import initialize_rag_pipeline # Untuk re-indexing

st.set_page_config(page_title="Panel Admin", layout="centered")

UPLOAD_FOLDER = "base_knowledge" # Sesuai struktur proyek
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Pastikan folder ada

def admin_panel_content():
    st.title("🔑 Panel Admin - Manajemen Basis Pengetahuan")
    st.markdown("Halaman ini digunakan untuk mengunggah dan mengelola file basis pengetahuan untuk chatbot GiziAI.")

    st.subheader("Unggah File Baru (.pdf, .docx, .txt)")
    uploaded_files = st.file_uploader(
        "Pilih satu atau lebih file untuk diunggah sebagai basis pengetahuan RAG.",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader_admin"
    )

    if uploaded_files:
        all_files_processed_successfully = True
        newly_uploaded_paths = []

        for uploaded_file in uploaded_files:
            filename = secure_filename(uploaded_file.name)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Simpan file ke server
            try:
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.info(f"File '{filename}' berhasil disimpan sementara di server.")
            except Exception as e:
                st.error(f"Gagal menyimpan file '{filename}' di server: {e}")
                all_files_processed_successfully = False
                continue # Lanjut ke file berikutnya jika ada

            # Simpan metadata ke DB
            file_id = store_file_metadata(filename, filepath, status='processing')
            if file_id:
                st.success(f"Metadata untuk '{filename}' (ID: {file_id}) berhasil dicatat. Status: Processing.")
                newly_uploaded_paths.append(filepath)
                # Tandai sebagai aktif agar diproses oleh RAG pipeline
                update_file_status(file_id, 'active') 
            else:
                st.error(f"Gagal menyimpan metadata untuk '{filename}' ke database.")
                all_files_processed_successfully = False
        
        if newly_uploaded_paths and all_files_processed_successfully:
            st.info("Semua file berhasil diunggah dan metadata disimpan. Memulai proses re-indexing basis pengetahuan...")
            with st.spinner("Memperbarui basis pengetahuan dengan file baru... Ini mungkin memerlukan waktu."):
                # Panggil initialize_rag_pipeline dengan path file baru untuk memicu pemrosesan
                # Fungsi ini akan memuat ulang vector store dengan file aktif terbaru.
                if initialize_rag_pipeline(new_file_paths_to_process=newly_uploaded_paths):
                    st.success("Basis pengetahuan berhasil diperbarui dengan file baru!")
                else:
                    st.error("Gagal memperbarui basis pengetahuan dengan file baru. Periksa log untuk detail.")
                    # Kembalikan status file ke 'error' atau 'processing' jika gagal index
                    # Ini memerlukan logika tambahan untuk melacak file_id per path
        elif not all_files_processed_successfully:
            st.warning("Beberapa file gagal diunggah atau disimpan metadatanya. Proses re-indexing tidak dijalankan.")
        
        # Membersihkan uploader setelah diproses untuk menghindari re-upload otomatis
        # Ini adalah workaround umum di Streamlit, st.rerun() mungkin diperlukan
        st.session_state.file_uploader_admin = [] # Kosongkan file uploader
        st.rerun()


    st.subheader("Status Basis Pengetahuan Saat Ini")
    active_files = get_active_knowledge_files() # Fungsi dari utils_db.py
    if active_files:
        st.write("Daftar file yang saat ini **aktif** dalam basis pengetahuan RAG:")
        for f_path in active_files:
            st.markdown(f"- `{os.path.basename(f_path)}` (Path: `{f_path}`)")
    else:
        st.info("Belum ada file aktif dalam basis pengetahuan.")

    if st.button("Muat Ulang Seluruh Basis Pengetahuan (Re-Index Semua File Aktif)", key="reindex_button"):
        with st.spinner("Memuat ulang seluruh basis pengetahuan dari semua file aktif... Ini mungkin memerlukan waktu."):
            # Panggil initialize_rag_pipeline dengan force_reload_vectorstore=True
            if initialize_rag_pipeline(force_reload_vectorstore=True):
                st.success("Seluruh basis pengetahuan berhasil dimuat ulang!")
            else:
                st.error("Gagal memuat ulang seluruh basis pengetahuan. Periksa log untuk detail.")


# --- Logika Otentikasi Admin ---
if 'logged_in_admin' not in st.session_state:
    st.session_state.logged_in_admin = False

if not st.session_state.logged_in_admin:
    st.warning("Anda harus login sebagai admin untuk mengakses halaman ini.")
    
    admin_username_key = "admin_page_username_input"
    admin_password_key = "admin_page_password_input"

    with st.form("login_form_admin"):
        username = st.text_input("Username Admin", key=admin_username_key)
        password = st.text_input("Password Admin", type="password", key=admin_password_key)
        submitted = st.form_submit_button("Login")

        if submitted:
            if verify_admin(username, password): # Fungsi dari utils_db.py
                st.session_state['logged_in_admin'] = True
                st.session_state['username'] = username # Simpan username untuk tampilan
                st.success("Login berhasil!")
                st.rerun() # Muat ulang halaman untuk menampilkan konten admin
            else:
                st.error("Username atau password admin salah.")
else:
    # Jika sudah login, tampilkan konten panel admin
    admin_panel_content()
    
    if st.button("Logout Admin", key="logout_button_admin_page"):
        st.session_state.logged_in_admin = False
        st.session_state.username = "Pengguna Tamu"
        st.success("Logout berhasil.")
        st.rerun()