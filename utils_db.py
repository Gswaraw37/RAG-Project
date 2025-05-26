import mysql.connector
import os
from dotenv import load_dotenv
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash # Untuk hashing password

# Muat variabel lingkungan dari file .env
load_dotenv(override=True)

# Konfigurasi database diambil dari variabel lingkungan
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT", "3306") # Default port 3306 jika tidak diset
}

def get_db_connection():
    """Membuat dan mengembalikan koneksi ke database MySQL."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Kesalahan koneksi ke MySQL: {err}")
        # Dalam aplikasi Streamlit, Anda mungkin ingin menggunakan st.error()
        return None

def create_tables():
    """Membuat tabel yang diperlukan dalam database jika belum ada."""
    conn = get_db_connection()
    if not conn:
        print("Tidak dapat membuat tabel, tidak ada koneksi DB.")
        return
    
    cursor = conn.cursor()

    # Tabel pengguna (untuk admin)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role ENUM('admin', 'user') DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Tabel untuk menyimpan metadata file basis pengetahuan
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            filepath VARCHAR(512) NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status ENUM('active', 'processing', 'inactive', 'error') DEFAULT 'processing'
        )
    ''')

    # Tabel untuk log chat (mirip application_logs di PDF)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            user_query TEXT,
            gpt_response TEXT,
            model_name VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX(session_id)
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()
    print("Pengecekan/pembuatan tabel database selesai.")

def add_admin_user_if_not_exists():
    """Menambahkan pengguna admin default jika belum ada."""
    conn = get_db_connection()
    if not conn: return

    cursor = conn.cursor()
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin_password") # Ambil password dari .env

    cursor.execute("SELECT id FROM users WHERE username = %s AND role = 'admin'", (admin_user,))
    if not cursor.fetchone():
        if not admin_pass:
            print("Password admin tidak diset di .env. Tidak dapat membuat admin.")
            cursor.close()
            conn.close()
            return
            
        hashed_password = generate_password_hash(admin_pass)
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                (admin_user, hashed_password, 'admin')
            )
            conn.commit()
            print(f"Pengguna admin '{admin_user}' berhasil dibuat.")
        except mysql.connector.Error as err:
            print(f"Gagal membuat pengguna admin: {err}")
            conn.rollback()
    cursor.close()
    conn.close()

def verify_admin(username, password):
    """Memverifikasi kredensial admin."""
    conn = get_db_connection()
    if not conn: return False
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT password_hash, role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user and user['role'] == 'admin' and check_password_hash(user['password_hash'], password):
            return True
    except mysql.connector.Error as err:
        print(f"Error saat verifikasi admin: {err}")
    finally:
        cursor.close()
        conn.close()
    return False

def store_file_metadata(filename, filepath, status='processing'):
    """Menyimpan metadata file yang diunggah ke database."""
    conn = get_db_connection()
    if not conn: return None
    
    cursor = conn.cursor()
    query = "INSERT INTO knowledge_files (filename, filepath, status) VALUES (%s, %s, %s)"
    try:
        cursor.execute(query, (filename, filepath, status))
        conn.commit()
        file_id = cursor.lastrowid
        return file_id
    except mysql.connector.Error as err:
        print(f"Error menyimpan metadata file: {err}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

def update_file_status(file_id, status):
    """Memperbarui status file dalam database."""
    conn = get_db_connection()
    if not conn: return
    
    cursor = conn.cursor()
    query = "UPDATE knowledge_files SET status = %s WHERE id = %s"
    try:
        cursor.execute(query, (status, file_id))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error memperbarui status file: {err}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def get_active_knowledge_files():
    """Mengambil daftar path file yang aktif dari database."""
    conn = get_db_connection()
    if not conn: return []
    
    cursor = conn.cursor(dictionary=True)
    files = []
    try:
        cursor.execute("SELECT filepath FROM knowledge_files WHERE status = 'active'")
        files = [row['filepath'] for row in cursor.fetchall()]
    except mysql.connector.Error as err:
        print(f"Error mengambil file aktif: {err}")
    finally:
        cursor.close()
        conn.close()
    return files

def insert_chat_log(session_id, user_query, gpt_response, model_name="LlamaCpp_GiziAI_Streamlit"):
    """Menyimpan log interaksi chat ke database (sesuai PDF)."""
    conn = get_db_connection()
    if not conn: return
    
    cursor = conn.cursor()
    query = '''
        INSERT INTO chat_logs (session_id, user_query, gpt_response, model_name)
        VALUES (%s, %s, %s, %s)
    '''
    try:
        cursor.execute(query, (session_id, user_query, gpt_response, model_name))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error menyisipkan log chat: {err}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def get_chat_history_from_db(session_id):
    """Mengambil riwayat chat berdasarkan session_id dari database (sesuai PDF)."""
    conn = get_db_connection()
    if not conn: return []
    
    cursor = conn.cursor(dictionary=True)
    messages = []
    # Langchain mengharapkan objek AIMessage dan HumanMessage
    from langchain_core.messages import HumanMessage, AIMessage
    try:
        cursor.execute(
            'SELECT user_query, gpt_response FROM chat_logs WHERE session_id=%s ORDER BY created_at ASC',
            (session_id,)
        )
        for row in cursor.fetchall():
            if row['user_query']: 
                messages.append(HumanMessage(content=row['user_query']))
            if row['gpt_response']:
                messages.append(AIMessage(content=row['gpt_response']))
    except mysql.connector.Error as err:
        print(f"Error mengambil riwayat chat: {err}")
    finally:
        cursor.close()
        conn.close()
    return messages

# Inisialisasi DB dan pengguna admin saat modul pertama kali diimpor
# Ini akan dijalankan sekali ketika aplikasi Streamlit dimulai
if __name__ != "__main__": # Hanya jalankan jika diimpor, bukan dijalankan langsung
    create_tables()
    add_admin_user_if_not_exists()