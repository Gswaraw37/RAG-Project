import streamlit as st
import uuid
import os
from utils_db import get_chat_history_from_db, insert_chat_log # Pastikan fungsi ini ada
from utils_rag import initialize_rag_pipeline # Untuk memastikan RAG siap
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Chatbot GiziAI", layout="wide")
st.title("💬 Chatbot GiziAI")
st.markdown("Ajukan pertanyaan Anda seputar gizi dan kesehatan masyarakat kepada GiziAI!")

# --- Inisialisasi Pipeline RAG ---
# (Kode inisialisasi RAG Anda di sini, pastikan berjalan dengan baik)
if 'rag_initialized_status' not in st.session_state or st.session_state.rag_initialized_status != "success":
    with st.spinner("Menyiapkan GiziAI... Ini mungkin memerlukan beberapa saat."):
        if initialize_rag_pipeline():
            st.session_state.rag_initialized_status = "success"
        else:
            st.session_state.rag_initialized_status = "failed"
            st.error("Gagal menyiapkan GiziAI. Beberapa fungsi mungkin tidak berjalan.")
            st.stop()

if st.session_state.rag_initialized_status != "success" or not st.session_state.get('rag_chain'):
    st.warning("Komponen AI tidak siap atau gagal dimuat. Chatbot tidak dapat berfungsi saat ini.")
    st.info("Silakan coba muat ulang aplikasi atau periksa log server untuk detail kesalahan.")
    if st.button("Coba Inisialisasi Ulang AI"):
        st.session_state.rag_initialized_status = "pending"
        st.rerun()
    st.stop()

# --- Manajemen Sesi Chat ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    print(f"Sesi chat baru dimulai: {st.session_state.session_id}")

# Muat riwayat chat dari DB atau inisialisasi sebagai list kosong
if "chat_history" not in st.session_state:
    st.session_state.chat_history = get_chat_history_from_db(st.session_state.session_id)

# --- Tampilkan Sapaan Awal (Hanya jika chat history KOSONG dan BARU DIMULAI) ---
# Kita gunakan variabel sesi lain untuk menandai apakah sapaan sudah ditampilkan
if "initial_greeting_displayed" not in st.session_state:
    st.session_state.initial_greeting_displayed = False

if not st.session_state.chat_history and not st.session_state.initial_greeting_displayed:
    initial_greeting_message = "Halo! Saya GiziAI, asisten virtual Anda untuk informasi gizi dan kesehatan. Ada yang bisa saya bantu?"
    with st.chat_message("ai", avatar="🍎"): # Tampilkan pesan AI
        st.markdown(initial_greeting_message)
    st.session_state.initial_greeting_displayed = True # Tandai bahwa sapaan sudah ditampilkan

# --- Tampilkan Riwayat Chat dari st.session_state.chat_history ---
# Ini akan menampilkan pesan yang benar-benar bagian dari percakapan yang disimpan
for message in st.session_state.chat_history:
    avatar_map = {"human": "🧑‍💻", "ai": "🍎"}
    with st.chat_message(message.type, avatar=avatar_map.get(message.type)):
        st.markdown(message.content)

# --- Tangani Input Pengguna ---
user_query = st.chat_input("Ketik pertanyaan Anda di sini...")

if user_query:
    # Saat ada input pengguna, pastikan sapaan (jika sebelumnya ditampilkan secara terpisah)
    # tidak lagi dianggap sebagai "belum ada chat"
    st.session_state.initial_greeting_displayed = True 

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("human", avatar="🧑‍💻"):
        st.markdown(user_query)

    with st.chat_message("ai", avatar="🍎"):
        message_placeholder = st.empty()
        message_placeholder.markdown("GiziAI sedang berpikir... 🧠")
        
        rag_chain = st.session_state.get('rag_chain')
        if rag_chain:
            try:
                # chat_history yang dikirim ke chain adalah yang ada di session_state
                # yang tidak akan berisi sapaan awal
                current_chat_history_for_chain = [m for m in st.session_state.chat_history if not (isinstance(m, HumanMessage) and m.content == user_query)]

                response_payload = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": current_chat_history_for_chain
                })
                
                ai_response_content = response_payload.get("answer", "Maaf, saya tidak dapat memproses permintaan Anda saat ini.")
                if not ai_response_content and isinstance(response_payload, str):
                    ai_response_content = response_payload

            except Exception as e:
                ai_response_content = f"Terjadi kesalahan internal saat memproses permintaan Anda: {e}"
                st.error(f"Error saat pemanggilan RAG chain: {e}")
        else:
            ai_response_content = "Maaf, GiziAI sedang tidak tersedia karena RAG chain belum siap."

        message_placeholder.markdown(ai_response_content)

    st.session_state.chat_history.append(AIMessage(content=ai_response_content))

    model_name_for_log = "LlamaCpp_GiziAI_Streamlit"
    if st.session_state.get('llm') and hasattr(st.session_state.llm, 'model_path'):
        model_name_for_log = os.path.basename(st.session_state.llm.model_path)
    
    insert_chat_log(st.session_state.session_id, user_query, ai_response_content, model_name_for_log)