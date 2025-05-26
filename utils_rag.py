import os
import streamlit as st # Menggunakan st untuk caching Streamlit
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough # Diganti dengan langchain_core.runnables
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.documents import Document # Tidak secara eksplisit digunakan di sini tapi baik untuk diketahui

import utils_db # Untuk mendapatkan file basis pengetahuan aktif

# Muat variabel lingkungan dari file .env
load_dotenv(override=True)

# Pastikan direktori yang diperlukan ada
os.makedirs("model", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)
os.makedirs("base_knowledge", exist_ok=True)

# Login ke Hugging Face jika token ada di .env (opsional, tergantung model)
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# --- Pemuatan LLM ---
@st.cache_resource # Cache resource LLM agar tidak dimuat ulang setiap saat
def load_llm_model():
    """Mengunduh (jika perlu) dan memuat model LlamaCpp."""
    # Path lengkap ke model GGUF, hf_hub_download akan mengelola ini di dalam cache_dir
    # hf_hub_download mengembalikan path absolut ke file yang diunduh.
    try:
        # hf_hub_download akan menyimpan file di MODEL_CACHE_DIR/nama_file jika tidak ada subfolder repo
        # atau MODEL_CACHE_DIR/blobs/hash_file jika menggunakan cache default HF.
        # Dengan cache_dir eksplisit, ia mencoba menempatkan langsung atau dalam struktur repo.
        # Kita asumsikan model berada di MODEL_CACHE_DIR/MODEL_FILENAME setelah diunduh.
        
        # Cek apakah file model sudah ada secara manual terlebih dahulu
        manual_model_path = os.path.join(os.getenv("MODEL_CACHE_DIR"), os.getenv("MODEL_FILENAME"))

        if not os.path.exists(manual_model_path):
            st.write(f"Model {os.getenv('MODEL_FILENAME')} tidak ditemukan di {manual_model_path}. Mencoba mengunduh...")
            try:
                # hf_hub_download akan menyimpan file di dalam MODEL_CACHE_DIR
                # dan mengembalikan path absolut ke file tersebut.
                downloaded_model_path = hf_hub_download(
                    repo_id=os.getenv("MODEL_REPO_ID"),
                    filename=os.getenv("MODEL_FILENAME"),
                    cache_dir=os.getenv("MODEL_CACHE_DIR"), # Menentukan direktori cache
                    local_dir=os.getenv("MODEL_CACHE_DIR"), # Mencoba menyimpan langsung di sini
                    local_dir_use_symlinks=False # Disarankan untuk cross-platform
                )
                # Pastikan path yang dikembalikan adalah yang kita gunakan
                # Terkadang hf_hub_download membuat struktur folder tambahan.
                # Jika downloaded_model_path adalah direktori, cari file di dalamnya.
                if os.path.isdir(downloaded_model_path):
                     potential_path = os.path.join(downloaded_model_path, os.getenv("MODEL_FILENAME"))
                     if os.path.exists(potential_path):
                         actual_model_path = potential_path
                     else: # fallback jika struktur tidak terduga
                         actual_model_path = manual_model_path # coba lagi path manual
                else:
                    actual_model_path = downloaded_model_path

                if not os.path.exists(actual_model_path): # Jika masih tidak ditemukan
                     st.error(f"Gagal mengunduh atau menemukan model di {actual_model_path} atau {manual_model_path}. Pastikan model ada atau dapat diunduh.")
                     return None
            except Exception as e:
                st.error(f"Error saat mengunduh model: {e}. Pastikan Anda memiliki koneksi internet dan token HF jika diperlukan.")
                return None
        else:
            actual_model_path = manual_model_path
            st.write(f"Model ditemukan di: {actual_model_path}")


        st.write(f"Memuat LLM dari: {actual_model_path}")
        llm = LlamaCpp(
            model_path=actual_model_path,
            n_gpu_layers=-1, # Sesuai PDF, -1 untuk menggunakan semua layer GPU jika ada, atau 1
            temperature=0.5, # Sesuai PDF
            top_p=0.95, # Sesuai PDF
            repeat_penalty=1.2, # Sesuai PDF
            stop=["Question:", "\n\n", "Human:"], # Sesuai PDF
            max_tokens=1024, # PDF: 1624, bisa disesuaikan
            n_ctx=8192, # PDF: 8192, sesuaikan dengan kemampuan model dan memori. Peringatan di PDF: n_ctx_per_seq < n_ctx_train
            n_batch=512, # Sesuai PDF
            verbose=False, # Sesuai PDF
            # f16_kv=True # Opsional, tergantung model dan hardware
        )
        st.success("LLM berhasil dimuat.")
        return llm
    except Exception as e:
        st.error(f"Error memuat model LlamaCpp: {e}")
        return None

# --- Fungsi Embedding ---
@st.cache_resource # Cache resource embedding model
def get_embedding_function():
    """Memuat model embedding Sentence Transformer."""
    try:
        st.write(f"Memuat model embedding: {os.getenv('EMBEDDING_MODEL_NAME')}")
        # model_kwargs={'device': 'cuda'} # Jika ingin menggunakan GPU untuk embedding
        # encode_kwargs={'normalize_embeddings': True} # Tergantung model dan kebutuhan
        embedding_func = SentenceTransformerEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME")
        )
        st.success("Model embedding berhasil dimuat.")
        return embedding_func
    except Exception as e:
        st.error(f"Error memuat model embedding: {e}")
        return None

# --- Pemrosesan Dokumen ---
def load_and_split_documents(file_paths: list):
    """Memuat dokumen dari path yang diberikan dan membaginya menjadi chunk."""
    docs = []
    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8') # Tambahkan TextLoader
            else:
                st.warning(f"Tipe file tidak didukung: {file_path}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error memuat dokumen {file_path}: {e}")

    if not docs:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Sesuai PDF
        chunk_overlap=300, # Sesuai PDF
        length_function=len # Sesuai PDF
    )
    splits = text_splitter.split_documents(docs)
    st.write(f"Memuat {len(docs)} dokumen, dibagi menjadi {len(splits)} chunk.")
    return splits

# --- Vector Store (ChromaDB) ---
# Cache vector store bisa rumit jika perlu sering update.
# Untuk aplikasi ini, kita akan memuatnya atau membuatnya ulang jika ada file baru.
def get_vectorstore(documents_splits: list = None, embedding_function=None, force_recreate=False):
    """Mendapatkan instance Chroma vector store. Membuat baru jika belum ada atau dipaksa."""
    if embedding_function is None:
        embedding_function = get_embedding_function()
        if embedding_function is None:
            st.error("Tidak dapat mendapatkan vector store tanpa fungsi embedding.")
            return None

    # Cek apakah direktori persistensi ada dan tidak kosong
    vectorstore_exists_on_disk = os.path.exists(os.getenv("CHROMA_PERSIST_DIR")) and len(os.listdir(os.getenv("CHROMA_PERSIST_DIR"))) > 0
    
    if not force_recreate and vectorstore_exists_on_disk and not documents_splits:
        # Coba muat dari disk jika tidak ada dokumen baru dan tidak dipaksa buat ulang
        st.write(f"Mencoba memuat vector store yang ada dari: {os.getenv('CHROMA_PERSIST_DIR')}")
        try:
            vectorstore = Chroma(
                persist_directory=os.getenv("CHROMA_PERSIST_DIR"),
                embedding_function=embedding_function,
                collection_name=os.getenv("CHROMA_COLLECTION_NAME")
            )
            # Verifikasi apakah koleksi ada dan berisi data
            if vectorstore._collection.count() > 0:
                st.success("Vector store berhasil dimuat dari disk.")
                return vectorstore
            else:
                st.warning("Vector store ada di disk tapi koleksi kosong. Akan dibuat ulang jika ada dokumen.")
                if not documents_splits: return None # Tidak bisa buat ulang tanpa dokumen
        except Exception as e:
            st.warning(f"Gagal memuat vector store dari disk: {e}. Akan dibuat ulang jika ada dokumen.")
            if not documents_splits: return None

    if documents_splits:
        # Buat vector store baru jika ada dokumen (atau jika gagal muat/dipaksa)
        st.write(f"Membuat vector store baru dari {len(documents_splits)} chunk dokumen.")
        try:
            vectorstore = Chroma.from_documents(
                documents=documents_splits,
                embedding=embedding_function,
                persist_directory=os.getenv("CHROMA_PERSIST_DIR"), # Menyimpan ke disk
                collection_name=os.getenv("CHROMA_COLLECTION_NAME")
            )
            st.success(f"Vector store berhasil dibuat dan disimpan di: {os.getenv('CHROMA_PERSIST_DIR')}")
            return vectorstore
        except Exception as e:
            st.error(f"Error membuat vector store: {e}")
            return None
    else:
        st.info("Tidak ada dokumen untuk membuat vector store, dan tidak ada vector store valid yang bisa dimuat.")
        return None

def add_documents_to_vectorstore(new_documents_splits: list, embedding_function=None):
    """Menambahkan dokumen baru ke vector store yang sudah ada."""
    if not new_documents_splits:
        st.info("Tidak ada dokumen baru untuk ditambahkan.")
        return False

    if embedding_function is None:
        embedding_function = get_embedding_function()
        if embedding_function is None:
            st.error("Tidak dapat menambahkan dokumen tanpa fungsi embedding.")
            return False
    
    try:
        # Muat vector store yang ada
        vectorstore = Chroma(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR"),
            embedding_function=embedding_function,
            collection_name=os.getenv("CHROMA_COLLECTION_NAME")
        )
        vectorstore.add_documents(new_documents_splits)
        st.success(f"Berhasil menambahkan {len(new_documents_splits)} chunk baru ke vector store.")
        return True
    except Exception as e:
        st.error(f"Error menambahkan dokumen ke vector store: {e}")
        return False

# --- RAG Chain (sesuai PDF untuk Conversational RAG) ---
def get_conversational_rag_chain(_llm, _retriever):
    """Membuat conversational RAG chain."""
    if _llm is None or _retriever is None:
        st.error("LLM atau Retriever tidak tersedia untuk membuat RAG chain.")
        return None

    # Prompt untuk kontekstualisasi pertanyaan (dari PDF)
    contextualize_q_system_prompt = (
        "Diberikan riwayat percakapan dan pertanyaan pengguna terbaru "
        "yang mungkin merujuk pada konteks dalam riwayat percakapan, "
        "formulasikan pertanyaan mandiri yang dapat dipahami "
        "tanpa riwayat percakapan. JANGAN menjawab pertanyaan, "
        "cukup formulasikan ulang jika diperlukan dan kembalikan apa adanya."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        _llm, _retriever, contextualize_q_prompt
    )

    # Prompt untuk menjawab pertanyaan dengan konteks (dari PDF, bagian RAG chain)
    # "Kamu adalah asisten ahli di bidang gizi dan kesehatan masyarakat.
    # PENTING: KELUARKAN KEMAMPUAN MAKSIMALMU untuk menjawab pertanyaan dengan natural dan terstruktur SESUAI KONTEKS yang diberikan.
    # Jika jawabannya tidak tahu, balas dengan: Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini.
    # Konteks: {context}
    # Pertanyaan: {question}
    # Jawaban:"
    # Di PDF, ini digabungkan dengan MessagesPlaceholder untuk history.
    
    # QA Prompt untuk create_stuff_documents_chain (sesuai PDF untuk create_retrieval_chain)
    # PDF menggunakan:
    # qa_prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    #     ("system", "Context: {context}"), # Ini mungkin salah, context biasanya dari retriever, bukan system message lagi
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("human", "{input}")
    # ])
    # Mari kita sesuaikan dengan prompt RAG yang lebih umum dan efektif:
    qa_system_prompt = """Kamu adalah asisten ahli di bidang gizi dan kesehatan masyarakat.
Gunakan potongan konteks berikut untuk menjawab pertanyaan pengguna.
Jika kamu tidak tahu jawabannya, katakan saja bahwa kamu tidak tahu, jangan mencoba membuat jawaban.
Jawablah dengan natural dan terstruktur.
Konteks:
{context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"), # Untuk menjaga percakapan
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(_llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    st.success("Conversational RAG chain berhasil dibuat.")
    return rag_chain

# --- Inisialisasi Pipeline RAG ---
def initialize_rag_pipeline(force_reload_vectorstore=False, new_file_paths_to_process=None):
    """Menginisialisasi semua komponen RAG: LLM, Embeddings, Vector Store, Retriever, dan RAG Chain."""
    
    # 1. Muat LLM
    if 'llm' not in st.session_state or st.session_state.llm is None:
        with st.spinner("Memuat Model Bahasa Besar (LLM)... Ini mungkin butuh waktu."):
            st.session_state.llm = load_llm_model()
    if st.session_state.llm is None:
        st.error("LLM gagal dimuat. Fungsi chatbot akan terbatas.")
        return False

    # 2. Muat Fungsi Embedding
    if 'embedding_function' not in st.session_state or st.session_state.embedding_function is None:
        with st.spinner("Memuat Model Embedding..."):
            st.session_state.embedding_function = get_embedding_function()
    if st.session_state.embedding_function is None:
        st.error("Fungsi embedding gagal dimuat. Fungsi RAG akan terbatas.")
        return False

    # 3. Vector Store
    # Jika ada file baru untuk diproses, kita perlu memperbarui/membuat ulang vector store
    if new_file_paths_to_process:
        st.write(f"Memproses file baru: {new_file_paths_to_process}")
        with st.spinner("Memproses dokumen baru dan memperbarui vector store..."):
            new_splits = load_and_split_documents(new_file_paths_to_process)
            if new_splits:
                # Opsi 1: Tambahkan ke vector store yang ada (jika sudah ada)
                # Opsi 2: Buat ulang vector store dengan semua file aktif + file baru (lebih sederhana untuk implementasi awal)
                # Untuk kesederhanaan, kita akan memuat semua file aktif dari DB dan membuat ulang.
                all_active_files = utils_db.get_active_knowledge_files() # Ini sudah termasuk yang baru ditandai aktif
                all_splits = load_and_split_documents(all_active_files)
                st.session_state.vectorstore = get_vectorstore(
                    documents_splits=all_splits,
                    embedding_function=st.session_state.embedding_function,
                    force_recreate=True # Paksa buat ulang karena ada file baru
                )
            else:
                st.warning("Tidak ada chunk valid dari file baru untuk memperbarui vector store.")
    elif force_reload_vectorstore or 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        # Pemuatan awal atau pemuatan ulang paksa
        with st.spinner("Memuat/Membuat Vector Store dari basis pengetahuan..."):
            active_kb_files = utils_db.get_active_knowledge_files()
            if active_kb_files:
                st.write(f"Memuat dokumen untuk vector store dari: {active_kb_files}")
                splits = load_and_split_documents(active_kb_files)
                st.session_state.vectorstore = get_vectorstore(
                    documents_splits=splits,
                    embedding_function=st.session_state.embedding_function,
                    force_recreate=force_reload_vectorstore
                )
            else: # Tidak ada file di DB, coba muat dari disk jika ada
                st.info("Tidak ada file basis pengetahuan aktif di DB. Mencoba memuat vector store dari disk jika ada.")
                st.session_state.vectorstore = get_vectorstore(
                    embedding_function=st.session_state.embedding_function,
                    force_recreate=False # Jangan paksa jika tidak ada file, hanya coba muat
                )

    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        st.warning("Vector store tidak tersedia. RAG tidak akan berfungsi optimal.")
        st.session_state.retriever = None
        st.session_state.rag_chain = None
        # return False # Jangan hentikan di sini, mungkin pengguna hanya ingin dashboard
        return True # Anggap saja inisialisasi parsial, chatbot tidak akan jalan

    # 4. Retriever (sesuai PDF)
    # search_kwargs dari PDF: {"k": 10, "fetch_k": 20, "lambda_mult": 0.75}
    # Di PDF lain: retriever.invoke("Berapa persentase bayi baru lahir yang mendapatkan Inisiasi Menyusu Dini (IMD) pada 2023?")
    # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.75})
    # Kita gunakan k yang lebih kecil untuk awal
    try:
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr", # Sesuai PDF
            search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.7} # Bisa disesuaikan
        )
        st.success("Retriever berhasil dibuat.")
    except Exception as e:
        st.error(f"Gagal membuat retriever: {e}")
        st.session_state.retriever = None
        st.session_state.rag_chain = None
        return False


    # 5. Conversational RAG Chain
    if st.session_state.llm and st.session_state.retriever:
         with st.spinner("Membuat RAG Chain..."):
            st.session_state.rag_chain = get_conversational_rag_chain(
                st.session_state.llm,
                st.session_state.retriever
            )
    else:
        st.session_state.rag_chain = None


    if st.session_state.rag_chain:
        st.sidebar.success("Pipeline RAG berhasil diinisialisasi!")
        return True
    else:
        st.sidebar.error("Gagal menginisialisasi RAG chain.")
        return False