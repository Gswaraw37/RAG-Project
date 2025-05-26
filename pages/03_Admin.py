import streamlit as st
import os
from werkzeug.utils import secure_filename # Untuk mengamankan nama file
from utils_db import store_file_metadata, update_file_status, verify_admin, get_active_knowledge_files
from utils_rag import initialize_rag_components, process_document_to_vectorstore_streamlit # Impor fungsi yang relevan

st.set_page_config(page_title="Panel Admin", layout="centered")

# UPLOAD_FOLDER diambil dari .env atau default
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "base_knowledge") 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def admin_panel_content():
    st.title("🔑 Panel Admin - Manajemen Basis Pengetahuan")
    st.markdown("Halaman ini digunakan untuk mengunggah dan mengelola file basis pengetahuan untuk chatbot GiziAI.")

    st.subheader("Unggah File Baru (.pdf, .docx, .txt)")

    # Gunakan session state untuk mengelola akhiran kunci uploader agar bisa di-reset
    if 'uploader_key_suffix' not in st.session_state:
        st.session_state.uploader_key_suffix = 0

    uploader_key = f"file_uploader_admin_rag_{st.session_state.uploader_key_suffix}"

    uploaded_files = st.file_uploader(
        "Pilih satu atau lebih file untuk diunggah.",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=uploader_key # Gunakan kunci yang dinamis
    )

    if uploaded_files:
        files_processed_successfully_count = 0
        files_failed_count = 0

        for uploaded_file in uploaded_files:
            filename = secure_filename(uploaded_file.name)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            try:
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.info(f"File '{filename}' berhasil disimpan ke server.")
                
                # Simpan metadata ke DB dengan status awal 'processing'
                # Fungsi process_document_to_vectorstore_streamlit akan mengupdate ke 'active' atau 'error'
                file_id = store_file_metadata(filename, filepath, status='processing')
                
                if file_id:
                    st.success(f"Metadata untuk '{filename}' (ID DB: {file_id}) dicatat. Memulai proses embedding...")
                    # Langsung proses file setelah diunggah
                    with st.spinner(f"Memproses '{filename}' dan menambahkan ke basis pengetahuan..."):
                        if process_document_to_vectorstore_streamlit(filepath, file_id):
                            st.success(f"File '{filename}' berhasil diproses dan ditambahkan ke RAG.")
                            files_processed_successfully_count += 1
                        else:
                            st.error(f"Gagal memproses file '{filename}' untuk RAG. Status di DB akan menjadi 'error'.")
                            files_failed_count += 1
                else:
                    st.error(f"Gagal menyimpan metadata untuk '{filename}' ke database.")
                    files_failed_count += 1
            except Exception as e:
                st.error(f"Gagal menyimpan atau memproses file '{filename}': {e}")
                files_failed_count += 1
        
        if files_processed_successfully_count > 0 or files_failed_count > 0:
            st.info(f"Total file berhasil diproses: {files_processed_successfully_count}, Gagal: {files_failed_count}.")
            # Ubah akhiran kunci agar uploader di-reset pada run berikutnya
            st.session_state.uploader_key_suffix += 1
            st.rerun() # Panggil rerun untuk memperbarui UI dengan kunci baru dan membersihkan uploader

    st.subheader("Status Basis Pengetahuan Saat Ini")
    active_files_paths = get_active_knowledge_files()
    if active_files_paths:
        st.write("Daftar file yang saat ini **aktif** dalam basis pengetahuan RAG:")
        for f_path in active_files_paths:
            st.markdown(f"- `{os.path.basename(f_path)}`")
    else:
        st.info("Belum ada file aktif dalam basis pengetahuan.")

    if st.button("Muat Ulang Sistem RAG & Proses Dokumen Pending", key="reinit_rag_button_admin"):
        with st.spinner("Memuat ulang sistem RAG dan memproses dokumen yang mungkin tertunda..."):
            # Memanggil initialize_rag_components akan memuat ulang komponen
            # dan juga memanggil process_pending_documents_streamlit di dalamnya
            if initialize_rag_components(): 
                st.success("Sistem RAG berhasil dimuat ulang dan dokumen pending (jika ada) telah diproses!")
            else:
                st.error("Gagal memuat ulang sistem RAG.")

# --- Logika Otentikasi Admin ---
if 'logged_in_admin' not in st.session_state:
    st.session_state.logged_in_admin = False

if not st.session_state.logged_in_admin:
    st.warning("Anda harus login sebagai admin untuk mengakses halaman ini.")
    admin_username_key = "admin_page_username_input_v3" # Mengubah key untuk menghindari konflik jika ada sisa state
    admin_password_key = "admin_page_password_input_v3"
    with st.form("login_form_admin_v3"):
        username = st.text_input("Username Admin", key=admin_username_key)
        password = st.text_input("Password Admin", type="password", key=admin_password_key)
        submitted = st.form_submit_button("Login")
        if submitted:
            if verify_admin(username, password):
                st.session_state['logged_in_admin'] = True
                st.session_state['username'] = username
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Username atau password admin salah.")
else:
    admin_panel_content()
    if st.button("Logout Admin", key="logout_button_admin_page_v3"):
        keys_to_delete = ['logged_in_admin', 'username', 
                          'rag_initialized_status', 'initial_greeting_displayed', 
                          'chat_history_display', 'session_id',
                          'uploader_key_suffix'] 
        for key_to_del in keys_to_delete:
            if key_to_del in st.session_state:
                del st.session_state[key_to_del]
        st.success("Logout berhasil.")
        st.rerun()