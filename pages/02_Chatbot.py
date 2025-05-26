import streamlit as st
import uuid
import os # Untuk os.path.basename jika diperlukan nanti

# utils_db dan utils_rag akan diimpor oleh app.py dan komponennya sudah di-cache/inisialisasi
from utils_db import get_chat_history_from_db # insert_chat_log dipanggil dari dalam RAG system
from utils_rag import get_rag_response_streamlit # Menggunakan fungsi RAG baru
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Chatbot GiziAI", layout="wide")
st.title("💬 Chatbot GiziAI")
st.markdown("Ajukan pertanyaan Anda seputar gizi dan kesehatan masyarakat kepada GiziAI!")

# --- Cek Status Inisialisasi RAG ---
# Diasumsikan initialize_rag_components() sudah dipanggil di app.py atau halaman utama
# dan statusnya tersimpan di st.session_state.rag_initialized_status
if st.session_state.get('rag_initialized_status') != "success":
    st.warning("Komponen AI belum siap atau gagal dimuat. Chatbot tidak dapat berfungsi saat ini.")
    st.info("Silakan periksa halaman utama untuk status inisialisasi AI atau hubungi admin.")
    # Tombol untuk mencoba inisialisasi ulang bisa ada di halaman utama (app.py)
    st.stop()

# --- Manajemen Sesi Chat ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    print(f"Chatbot: Sesi chat baru dimulai: {st.session_state.session_id}")

# Muat riwayat chat dari DB untuk TAMPILAN.
# Riwayat chat untuk KONTEKS RAG akan diambil langsung oleh get_rag_response_streamlit
if "chat_history_display" not in st.session_state:
    # get_chat_history_from_db sudah mengembalikan list of HumanMessage/AIMessage
    st.session_state.chat_history_display = get_chat_history_from_db(st.session_state.session_id)
    print(f"Chatbot: Memuat {len(st.session_state.chat_history_display)} pesan untuk tampilan dari DB.")


# --- Tampilkan Sapaan Awal (Hanya jika chat history KOSONG dan BARU DIMULAI) ---
if "initial_greeting_displayed" not in st.session_state:
    st.session_state.initial_greeting_displayed = False

if not st.session_state.chat_history_display and not st.session_state.initial_greeting_displayed:
    initial_greeting_message = "Halo! Saya GiziAI, asisten virtual Anda untuk informasi gizi dan kesehatan. Ada yang bisa saya bantu?"
    with st.chat_message("ai", avatar="🍎"):
        st.markdown(initial_greeting_message)
    st.session_state.initial_greeting_displayed = True
    # Sapaan ini tidak ditambahkan ke chat_history_display agar tidak duplikat jika user langsung bertanya

# --- Tampilkan Riwayat Chat dari st.session_state.chat_history_display ---
for message in st.session_state.chat_history_display:
    avatar_map = {"human": "🧑‍💻", "ai": "🍎"}
    with st.chat_message(message.type, avatar=avatar_map.get(message.type)):
        st.markdown(message.content)

# --- Tangani Input Pengguna ---
user_query = st.chat_input("Ketik pertanyaan Anda di sini...")

if user_query:
    st.session_state.initial_greeting_displayed = True # Pastikan sapaan tidak muncul lagi

    # Tampilkan pesan pengguna di UI dan tambahkan ke histori display
    st.chat_message("human", avatar="🧑‍💻").markdown(user_query)
    st.session_state.chat_history_display.append(HumanMessage(content=user_query))
    
    # Dapatkan respons dari RAG system
    with st.spinner("GiziAI sedang berpikir dan mencari informasi... 🧠"):
        # Panggil fungsi RAG baru Anda
        # Fungsi ini sudah menangani penyimpanan ke DB di dalamnya
        ai_response_content = get_rag_response_streamlit(st.session_state.session_id, user_query)
    
    # Tampilkan respons AI di UI dan tambahkan ke histori display
    st.chat_message("ai", avatar="🍎").markdown(ai_response_content)
    st.session_state.chat_history_display.append(AIMessage(content=ai_response_content))
    
    # Tidak perlu insert_chat_log lagi di sini karena sudah dihandle di get_rag_response_streamlit
    # st.rerun() # Tidak selalu perlu, st.chat_input biasanya memicu rerun. Jika ada update aneh, baru tambahkan.