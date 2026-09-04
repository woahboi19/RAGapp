import streamlit as st
import json
from rag_engine import RAGEngine

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Uzman Yönlendirme Asistanı",
    page_icon="🧭",
    layout="centered"
)

# --- Engine'i Sadece Bir Kere Yükle (Cache) ---
# st.cache_resource sayesinde arayüz her yenilendiğinde 5GB'lık model ve Chroma baştan yüklenmez!
@st.cache_resource(show_spinner="Arka plan motoru ve yapay zeka yükleniyor... Lütfen bekleyin.")
def load_engine():
    return RAGEngine(model_name="gemma4", db_dir="./chroma_db_gercek")

engine = load_engine()

# --- Veritabanı İstatistiklerini Oku ---
@st.cache_data(ttl=3600)  # Her render'da disk okunmasını engelle
def get_db_stats():
    try:
        with open("data/meslekler_db.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return len(data)
    except Exception as e:
        return 0

# --- Yan Menü (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3273/3273397.png", width=100) # İsteğe bağlı şık bir ikon
    st.title("Sistem Durumu")
    st.success("Motor Aktif")
    st.info(f"Veritabanındaki Meslek Sayısı: **{get_db_stats()}**")
    st.markdown("---")
    st.markdown("""
    **Nasıl Kullanılır?**
    1. Yaşadığınız sorunu kısaca anlatın.
    2. Sistem arka planda en uygun meslekleri bulur.
    3. Size hangi uzmana başvurmanız gerektiğini tavsiye eder.
    
    *Not: Saçma veya doğaüstü sorular sormaktan kaçının.*
    """)

# --- Ana Ekran Başlığı ---
st.title("Uzman Yönlendirme Asistanı")
st.markdown("Hangi uzmana başvurmanız gerektiğini bilmiyor musunuz? Derdinizi anlatın, doğru kişiye yönlendireyim.")

# --- Sohbet Geçmişi (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Bugün size nasıl yardımcı olabilirim? Lütfen probleminizi anlatın."}
    ]

# Geçmiş mesajları ekrana bas
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Kullanıcı Girdisi ---
if prompt := st.chat_input("Örn: Kedim sabahtan beri kusuyor..."):
    # Kullanıcının mesajını ekrana bas ve hafızaya al
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Yapay zeka cevap üretirken dönen simge göster
    with st.chat_message("assistant"):
        with st.spinner("Sorununuz analiz ediliyor ve en uygun uzmanlar aranıyor..."):
            # RAG motorundan cevabı al
            response = engine.get_response(prompt)
            st.markdown(response)
    
    # Asistanın cevabını hafızaya al
    st.session_state.messages.append({"role": "assistant", "content": response})
