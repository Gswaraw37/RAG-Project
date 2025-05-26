import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from huggingface_hub import login, hf_hub_download

import utils_db

load_dotenv(override=True)

llm = None
embedding_function = None
vectorstore = None
retriever = None
contextualize_q_chain = None
answer_generation_chain = None

MAX_HISTORY_MESSAGES_FOR_CONTEXTUALIZATION = int(os.getenv("MAX_HISTORY_MESSAGES_FOR_CONTEXTUALIZATION", 4))
MAX_STANDALONE_QUESTION_WORDS = int(os.getenv("MAX_STANDALONE_QUESTION_WORDS", 30))
MIN_VALID_ANSWER_LENGTH = int(os.getenv("MIN_VALID_ANSWER_LENGTH", 15))
MIN_CONTEXT_LENGTH_FOR_ANSWER = int(os.getenv("MIN_CONTEXT_LENGTH_FOR_ANSWER", 50))
MIN_KEYWORD_OVERLAP_FOR_RELEVANCE = int(os.getenv("MIN_KEYWORD_OVERLAP_FOR_RELEVANCE", 1))

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_streamlit_app")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "giziai_knowledge_app")
HF_TOKEN = os.getenv("HF_TOKEN")
KNOWLEDGE_BASE_DIR = os.getenv("UPLOAD_FOLDER", "base_knowledge")

# Pastikan direktori yang diperlukan ada
# Pindahkan pembuatan direktori model ke dalam load_llm_model jika path model ada
# if LLM_MODEL_PATH and not os.path.exists(os.path.dirname(LLM_MODEL_PATH)) and os.path.dirname(LLM_MODEL_PATH) != "":
#     os.makedirs(os.path.dirname(LLM_MODEL_PATH), exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
# Untuk MODEL_CACHE_DIR dari .env Anda (misal "model/"), pastikan dibuat sebelum hf_hub_download jika digunakan
MODEL_CACHE_DIR_FROM_ENV = os.getenv("MODEL_CACHE_DIR", "model/") # Ambil dari .env
os.makedirs(MODEL_CACHE_DIR_FROM_ENV, exist_ok=True)


def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def approximate_token_count(text):
    if not text: return 0
    return len(text) // 4

def get_keywords_from_query(query_text):
    if not query_text: return set()
    words = re.sub(r'[^\w\s]', '', query_text.lower()).split()
    keywords = {word for word in words if len(word) > 2}
    return keywords

@st.cache_resource(show_spinner="Menginisialisasi komponen inti RAG...")
def initialize_rag_components():
    global llm, embedding_function, vectorstore, retriever, contextualize_q_chain, answer_generation_chain

    st.write("Memulai inisialisasi komponen RAG...")
    all_components_initialized = True

    # Login ke Hugging Face jika token ada (DIPINDAHKAN KE SINI)
    if HF_TOKEN:
        st.info("Mencoba login ke Hugging Face Hub dengan token yang disediakan...")
        try:
            login(token=HF_TOKEN)
            st.success("Login ke Hugging Face Hub berhasil (jika token valid).")
        except Exception as e:
            st.warning(f"Gagal login ke Hugging Face Hub: {e}. Proses akan dilanjutkan.")

    # 1. Inisialisasi LLM
    if llm is None:
        # Pastikan direktori untuk model LLM ada jika pathnya adalah path file
        if LLM_MODEL_PATH and os.path.dirname(LLM_MODEL_PATH) and not os.path.exists(os.path.dirname(LLM_MODEL_PATH)):
            os.makedirs(os.path.dirname(LLM_MODEL_PATH), exist_ok=True)
            
        if not LLM_MODEL_PATH:
            st.error("Error: Path model LLM (LLM_MODEL_PATH) tidak dikonfigurasi di .env.")
            all_components_initialized = False
        elif not os.path.exists(LLM_MODEL_PATH):
            # Logika unduh model jika LLM_MODEL_PATH tidak ada dan MODEL_REPO_ID didefinisikan
            # Ini adalah contoh, sesuaikan dengan kebutuhan Anda jika ingin unduh otomatis
            model_repo_id = os.getenv("MODEL_REPO_ID")
            model_filename = os.getenv("MODEL_FILENAME")
            model_cache_dir = os.getenv("MODEL_CACHE_DIR", "model/") # default ke "model/"

            if model_repo_id and model_filename:
                st.write(f"Model {model_filename} tidak ditemukan di {LLM_MODEL_PATH}. Mencoba mengunduh dari {model_repo_id} ke {model_cache_dir}...")
                os.makedirs(model_cache_dir, exist_ok=True)
                try:
                    downloaded_model_path = hf_hub_download(
                        repo_id=model_repo_id,
                        filename=model_filename,
                        cache_dir=model_cache_dir,
                        local_dir=model_cache_dir, # Mencoba menyimpan langsung di sini
                        local_dir_use_symlinks=False
                    )
                    # hf_hub_download mengembalikan path absolut ke file yang diunduh
                    # Kita perlu memastikan LLM_MODEL_PATH sekarang menunjuk ke sana jika berhasil
                    # Namun, untuk konsistensi, pengguna harus mengatur LLM_MODEL_PATH ke path yang benar setelah unduh.
                    # Untuk sekarang, kita anggap jika LLM_MODEL_PATH tidak ada, itu error.
                    # Atau, kita bisa update LLM_MODEL_PATH di sini jika unduhan berhasil, tapi itu jadi stateful.
                    # Untuk kesederhanaan: jika LLM_MODEL_PATH tidak ada, maka error.
                    st.error(f"Model LLM tidak ditemukan di path: '{LLM_MODEL_PATH}'. Harap unduh model secara manual atau pastikan MODEL_REPO_ID & MODEL_FILENAME di .env benar untuk unduhan otomatis (fitur unduh belum sepenuhnya diimplementasikan di sini).")
                    all_components_initialized = False

                except Exception as e:
                    st.error(f"Error saat mencoba mengunduh model: {e}")
                    all_components_initialized = False
            else:
                st.error(f"Error: File model LLM tidak ditemukan di path: '{LLM_MODEL_PATH}' dan konfigurasi untuk unduh otomatis (MODEL_REPO_ID, MODEL_FILENAME) tidak ada.")
                all_components_initialized = False
        
        if all_components_initialized and os.path.exists(LLM_MODEL_PATH): # Hanya lanjut jika path valid
            try:
                st.write(f"Memuat LLM dari: {LLM_MODEL_PATH}")
                llm = LlamaCpp(
                    model_path=LLM_MODEL_PATH,
                    n_gpu_layers=int(os.getenv("LLM_N_GPU_LAYERS", -1)), temperature=float(os.getenv("LLM_TEMPERATURE", 0.5)),
                    top_p=float(os.getenv("LLM_TOP_P", 0.95)), repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", 1.2)),
                    stop=["Question:", "\n\n", "Human:"], max_tokens=int(os.getenv("LLM_MAX_TOKENS", 1024)),
                    n_ctx=int(os.getenv("LLM_N_CTX", 8192)), n_batch=int(os.getenv("LLM_N_BATCH", 512)),
                    verbose=False
                )
                st.success("LLM berhasil dimuat.")
            except Exception as e:
                st.error(f"Error saat memuat LLM LlamaCpp: {e}")
                llm = None
                all_components_initialized = False
    else:
        st.info("LLM sudah terinisialisasi sebelumnya.")

    # 2. Inisialisasi Embedding Function
    if embedding_function is None:
        try:
            st.write(f"Memuat model embedding: {EMBEDDING_MODEL_NAME}")
            embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            st.success(f"Embedding function '{EMBEDDING_MODEL_NAME}' berhasil dimuat.")
        except Exception as e:
            st.error(f"Error saat memuat embedding function '{EMBEDDING_MODEL_NAME}': {e}")
            embedding_function = None
            all_components_initialized = False
    else:
        st.info("Embedding function sudah terinisialisasi sebelumnya.")

    # 3. Inisialisasi Vector Store dan Retriever
    if vectorstore is None and embedding_function:
        try:
            st.write(f"Menginisialisasi vector store dari: {CHROMA_PERSIST_DIRECTORY}")
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                embedding_function=embedding_function
            )
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.7}
            )
            st.success(f"Vector store dan retriever berhasil diinisialisasi.")
            process_pending_documents_streamlit()
        except Exception as e:
            st.error(f"Error saat menginisialisasi ChromaDB/Retriever: {e}")
            vectorstore = None
            retriever = None
            all_components_initialized = False
    elif embedding_function is None and vectorstore is None:
        st.warning("Vector store/Retriever tidak dapat diinisialisasi karena embedding function gagal dimuat.")
        all_components_initialized = False
    elif vectorstore is not None and retriever is None and embedding_function : # Jika vectorstore ada tapi retriever belum
        try:
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.7}
            )
            st.info("Retriever berhasil dibuat dari vectorstore yang sudah ada.")
            process_pending_documents_streamlit() # Proses juga jika retriever baru dibuat
        except Exception as e:
            st.error(f"Error membuat retriever dari vectorstore yang ada: {e}")
            retriever = None
            all_components_initialized = False
    elif vectorstore is not None and retriever is not None:
         st.info("Vector store dan retriever sudah terinisialisasi sebelumnya.")
         if embedding_function: # Hanya proses jika embedding ada
            process_pending_documents_streamlit()


    # 4. Setup Chains
    if llm and retriever and (contextualize_q_chain is None or answer_generation_chain is None) :
        st.write("Membuat Langchain chains...")
        contextualize_q_system_prompt = (
            "Diberikan riwayat percakapan dan pertanyaan pengguna terbaru "
            "yang mungkin merujuk pada konteks dalam riwayat percakapan, "
            "formulasikan pertanyaan mandiri yang dapat dipahami "
            "tanpa riwayat percakapan. JANGAN menjawab pertanyaan, "
            "cukup formulasikan ulang jika diperlukan dan kembalikan apa adanya."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
        st.info("Contextualization chain berhasil dibuat.")

        qa_template_simple_text = """Kamu adalah asisten ahli di bidang gizi dan kesehatan masyarakat.
PENTING: KELUARKAN KEMAMPUAN MAKSIMALMU untuk menjawab pertanyaan dengan natural dan terstruktur SESUAI KONTEKS yang diberikan.
Jika jawabannya tidak ada didalam KONTEKS, HARUS balas dengan: Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:"""
        simple_qa_prompt_template = ChatPromptTemplate.from_template(qa_template_simple_text)

        answer_generation_chain = (
            RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever | RunnableLambda(docs2str))
            | simple_qa_prompt_template
            | llm
            | StrOutputParser()
        )
        st.info("Answer generation chain (RAG sederhana) berhasil dibuat.")
    elif not llm or not retriever:
        st.warning("Chains tidak dapat dibuat karena LLM atau Retriever tidak terinisialisasi.")
        all_components_initialized = False
    elif contextualize_q_chain and answer_generation_chain:
        st.info("Langchain chains sudah terinisialisasi sebelumnya.")

    if all_components_initialized and llm and retriever and contextualize_q_chain and answer_generation_chain:
        st.success("Semua komponen RAG berhasil diinisialisasi.")
    else:
        st.error("Beberapa komponen RAG gagal diinisialisasi sepenuhnya. Periksa log di atas.")
        all_components_initialized = False # Pastikan false jika ada yg gagal
    
    return all_components_initialized

def process_document_to_vectorstore_streamlit(filepath, file_id):
    global vectorstore, embedding_function
    if not vectorstore or not embedding_function:
        st.error("Error: Vectorstore atau embedding function belum terinisialisasi untuk memproses dokumen.")
        print("Error: Vectorstore atau embedding function belum terinisialisasi untuk memproses dokumen.")
        return False
    try:
        st.info(f"Memproses file: {os.path.basename(filepath)} (ID DB: {file_id})")
        print(f"Memproses file: {filepath} (ID: {file_id})")
        
        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filepath.endswith(".txt"):
            loader = TextLoader(filepath, encoding='utf-8')
        else:
            st.warning(f"Tipe file {os.path.basename(filepath)} tidak didukung.")
            print(f"Tipe file tidak didukung: {filepath}")
            utils_db.update_file_status(file_id, 'error')
            return False
            
        documents = loader.load()
        if not documents:
            st.warning(f"Tidak ada konten yang dapat dimuat dari {os.path.basename(filepath)}.")
            print(f"Tidak ada konten yang dapat dimuat dari {filepath}")
            utils_db.update_file_status(file_id, 'error')
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300, length_function=len)
        splits = text_splitter.split_documents(documents)

        if not splits:
            st.warning(f"Tidak ada teks yang dapat diekstrak (setelah split) dari {os.path.basename(filepath)}")
            print(f"Tidak ada teks yang dapat diekstrak dari {filepath}")
            utils_db.update_file_status(file_id, 'error')
            return False
        
        for split in splits:
            split.metadata["source"] = os.path.basename(filepath)
            split.metadata["file_id"] = str(file_id)

        vectorstore.add_documents(splits)
        st.success(f"Berhasil memproses dan menambahkan {len(splits)} chunk dari {os.path.basename(filepath)} ke vector store.")
        print(f"Berhasil memproses dan menambahkan {len(splits)} chunk dari {filepath} ke vector store.")
        utils_db.update_file_status(file_id, 'active')
        return True
    except Exception as e:
        st.error(f"Error saat memproses dokumen {os.path.basename(filepath)}: {e}")
        print(f"Error saat memproses dokumen {filepath}: {e}")
        utils_db.update_file_status(file_id, 'error')
        return False

def process_pending_documents_streamlit():
    global vectorstore, embedding_function
    if not vectorstore or not embedding_function:
        st.warning("Vectorstore atau embedding function belum siap untuk memproses dokumen pending.")
        print("Vectorstore atau embedding function belum siap untuk memproses dokumen pending.")
        return

    st.write("Mengecek dokumen yang belum diproses dari database...")
    print("Mengecek dokumen yang belum diproses...")
    unprocessed_files = utils_db.get_unprocessed_files_for_rag()
    
    if not unprocessed_files:
        st.info("Tidak ada dokumen baru untuk diproses dari database.")
        print("Tidak ada dokumen baru untuk diproses.")
        return

    processed_count = 0
    for db_file_entry in unprocessed_files:
        filepath = db_file_entry['filepath']
        file_id = db_file_entry['id']
        filename = db_file_entry['filename']

        if os.path.exists(filepath):
            st.write(f"Memulai pemrosesan untuk file: {filename} (ID DB: {file_id})")
            if process_document_to_vectorstore_streamlit(filepath, file_id):
                processed_count +=1
        else:
            st.error(f"File {filename} (path: {filepath}, ID DB: {file_id}) tidak ditemukan di sistem file.")
            print(f"File {filepath} (ID: {file_id}) tidak ditemukan.")
            utils_db.update_file_status(file_id, 'error')
    
    if processed_count > 0:
        st.success(f"Selesai memproses {processed_count} dokumen yang tertunda.")
    else:
        st.info("Tidak ada dokumen yang berhasil diproses pada sesi ini (mungkin sudah diproses atau ada error).")
    print("Selesai memproses dokumen yang tertunda.")

def get_rag_response_streamlit(session_uuid: str, user_input: str):
    global llm, contextualize_q_chain, answer_generation_chain, retriever

    if not llm or not contextualize_q_chain or not answer_generation_chain or not retriever:
        error_msg = "Sistem RAG belum siap sepenuhnya."
        st.error(error_msg)
        print(f"ERROR: {error_msg}")
        utils_db.insert_chat_log(session_uuid, user_input, error_msg, "N/A - RAG System Error")
        return error_msg

    chat_history_for_contextualization = utils_db.get_chat_history_from_db(session_uuid)
    
    if len(chat_history_for_contextualization) > MAX_HISTORY_MESSAGES_FOR_CONTEXTUALIZATION:
        start_index = len(chat_history_for_contextualization) - MAX_HISTORY_MESSAGES_FOR_CONTEXTUALIZATION
        chat_history_for_contextualization = chat_history_for_contextualization[start_index:]
    
    generated_standalone_question = user_input
    if chat_history_for_contextualization:
        try:
            # st.write(f"DEBUG (Streamlit): Input ke kontekstualisasi - History: {len(chat_history_for_contextualization)} pesan, Input: '{user_input}'")
            print(f"DEBUG: Input ke contextualize_q_chain - History: {len(chat_history_for_contextualization)} pesan, Input: '{user_input}'")
            raw_reformulated_question = contextualize_q_chain.invoke({
                "chat_history": chat_history_for_contextualization,
                "input": user_input
            })
            # st.write(f"DEBUG (Streamlit): Output mentah dari kontekstualisasi: '{raw_reformulated_question}'")
            print(f"DEBUG: Output mentah dari contextualize_q_chain: '{raw_reformulated_question}'")

            cleaned_question = raw_reformulated_question.strip()
            prefixes_to_remove = ["ai:", "jawaban:", "output anda:", "output saya:", "pertanyaan:"]
            for prefix in prefixes_to_remove:
                if cleaned_question.lower().startswith(prefix):
                    cleaned_question = cleaned_question[len(prefix):].strip()
            
            word_count = len(cleaned_question.split())
            is_likely_answer = (
                word_count > MAX_STANDALONE_QUESTION_WORDS or
                (not cleaned_question.endswith("?") and user_input.lower() != cleaned_question.lower())
            )

            if is_likely_answer and cleaned_question.lower() != user_input.lower() :
                # st.write(f"DEBUG (Streamlit): Output kontekstualisasi ('{cleaned_question}') tampak seperti jawaban/terlalu panjang. Menggunakan input asli.")
                print(f"DEBUG: Output kontekstualisasi ('{cleaned_question}') tampak seperti jawaban/terlalu panjang. Menggunakan input asli.")
                generated_standalone_question = user_input
            else:
                generated_standalone_question = cleaned_question
            # st.write(f"DEBUG (Streamlit): Pertanyaan asli: '{user_input}', Pertanyaan standalone: '{generated_standalone_question}'")
            print(f"DEBUG: Pertanyaan asli: '{user_input}', Pertanyaan standalone (setelah pembersihan): '{generated_standalone_question}'")
        except Exception as e:
            st.error(f"Error saat kontekstualisasi pertanyaan: {e}. Menggunakan input asli.")
            print(f"Error saat kontekstualisasi pertanyaan: {e}. Menggunakan input asli.")
            generated_standalone_question = user_input
    else:
        # st.write(f"DEBUG (Streamlit): Tidak ada histori, pertanyaan digunakan langsung: '{generated_standalone_question}'")
        print(f"DEBUG: Tidak ada histori, pertanyaan digunakan langsung: '{generated_standalone_question}'")

    standalone_question_for_rag = generated_standalone_question

    retrieved_docs_str = ""
    is_context_relevant_for_question = False
    try:
        docs = retriever.invoke(standalone_question_for_rag)
        retrieved_docs_str = docs2str(docs).strip()
        # st.write(f"DEBUG (Streamlit): Konteks diambil (Panjang: {len(retrieved_docs_str)} chars):\n---\n{retrieved_docs_str[:100]}...\n---")
        print(f"DEBUG: Konteks yang diambil (Panjang: {len(retrieved_docs_str)}):\n---\n{retrieved_docs_str[:200]}...\n---")

        if retrieved_docs_str and len(retrieved_docs_str) >= MIN_CONTEXT_LENGTH_FOR_ANSWER:
            question_keywords = get_keywords_from_query(standalone_question_for_rag)
            context_sample_for_keywords = retrieved_docs_str[:1000].lower()
            common_keyword_count = 0
            if question_keywords:
                for q_keyword in question_keywords:
                    if q_keyword in context_sample_for_keywords:
                        common_keyword_count += 1
            if common_keyword_count >= MIN_KEYWORD_OVERLAP_FOR_RELEVANCE:
                is_context_relevant_for_question = True
        # st.write(f"DEBUG (Streamlit): Apakah konteks relevan? {is_context_relevant_for_question}. Overlap kata kunci: {common_keyword_count if 'common_keyword_count' in locals() else 'N/A'}")
        print(f"DEBUG: Apakah konteks relevan untuk pertanyaan ('{standalone_question_for_rag}')? {is_context_relevant_for_question}. Overlap kata kunci: {common_keyword_count if 'common_keyword_count' in locals() else 'N/A'}")
    except Exception as e:
        st.error(f"DEBUG: Error saat mengambil dokumen: {e}")
        print(f"DEBUG: Error saat mengambil dokumen: {e}")
        retrieved_docs_str = ""

    fallback_message = "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."
    bot_answer = fallback_message

    try:
        if not is_context_relevant_for_question and retrieved_docs_str:
            # st.write("DEBUG (Streamlit): Konteks diambil tapi dianggap tidak relevan. Menggunakan fallback.")
            print("DEBUG: Konteks diambil tapi dianggap tidak relevan. Menggunakan fallback.")
            bot_answer = fallback_message
        elif not retrieved_docs_str:
            # st.write("DEBUG (Streamlit): Tidak ada konteks yang diambil. Menggunakan fallback.")
            print("DEBUG: Tidak ada konteks yang diambil. Menggunakan fallback.")
            bot_answer = fallback_message
        else:
            # st.write("DEBUG (Streamlit): Konteks relevan, melanjutkan ke LLM untuk jawaban.")
            print("DEBUG: Konteks relevan, melanjutkan ke LLM untuk jawaban.")
            bot_answer_raw = answer_generation_chain.invoke({
                "question": standalone_question_for_rag,
            })
            # st.write(f"DEBUG (Streamlit): Output mentah dari LLM: '{bot_answer_raw}'")
            print(f"DEBUG: Output mentah dari answer_generation_chain: '{bot_answer_raw}'")
            bot_answer_stripped = bot_answer_raw.strip()

            if not bot_answer_stripped or len(bot_answer_stripped) < MIN_VALID_ANSWER_LENGTH:
                if fallback_message.lower() not in bot_answer_stripped.lower():
                    # st.write(f"DEBUG (Streamlit) Post-Proc: Output LLM ('{bot_answer_stripped}') kosong/pendek. Fallback.")
                    print(f"DEBUG Post-Proc: Output LLM ('{bot_answer_stripped}') kosong/pendek. Fallback.")
                    bot_answer = fallback_message
                else:
                    bot_answer = bot_answer_stripped
            else:
                bot_answer = bot_answer_stripped
    except Exception as e:
        st.error(f"Error saat menjalankan answer generation chain: {e}")
        print(f"Error saat menjalankan answer generation chain: {e}")
        bot_answer = fallback_message
        utils_db.insert_chat_log(session_uuid, user_input, f"Error: {str(e)} | Fallback: {bot_answer}",
                                 os.path.basename(LLM_MODEL_PATH) if LLM_MODEL_PATH else "LlamaCpp_Unknown")
        return bot_answer

    model_name_for_log = os.path.basename(LLM_MODEL_PATH) if LLM_MODEL_PATH else "LlamaCpp_Unknown"
    utils_db.insert_chat_log(session_uuid, user_input, bot_answer, model_name_for_log)
    
    return bot_answer