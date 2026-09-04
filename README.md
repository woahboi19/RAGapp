# Uzman Yönlendirme Asistanı (Local RAG)

Bu proje, kullanıcıların yaşadıkları günlük sorunları analiz edip onları en doğru meslek grubuna veya uzmana yönlendiren yerel bir RAG (Retrieval-Augmented Generation) uygulamasıdır. 
Tamamen yerel makinenizde çalışacak şekilde tasarlanmıştır ve sohbet verileriniz dışarıya (internete) çıkmaz.

## 🚀 Özellikler

- **Yerel Gizlilik:** Ollama aracılığıyla tamamen çevrimdışı (offline) LLM (varsayılan olarak `gemma4`) kullanımı.
- **Gelişmiş Vektör Arama:** Metin gömme (embeddings) işlemleri için HuggingFace `all-MiniLM-L6-v2` ve vektör veritabanı olarak `ChromaDB`.
- **Çoklu Kullanım Ortamı:** Hem **Streamlit** ile modern bir web arayüzü hem de **CLI** (terminal) üzerinden hızlı kullanım.
- **Otonom Veri Fabrikası:** Yapay zeka kullanılarak mesleklerin tanımlarının ve çözdükleri sorunların otonom şekilde üretilip veritabanına eklenmesi.
- **Failsafe (Güvenlik Duvarı):** RAG yapısının mantıksız, doğaüstü veya sistem dışı soruları ayırt etme özelliği (HyDE yaklaşımı ile entegre).

## 🛠️ Kurulum

### 1. Gereksinimler
- Python 3.9 veya üzeri
- [Ollama](https://ollama.com/) (Sisteminizde kurulu ve arka planda çalışıyor olmalıdır)
- Ollama üzerinden `gemma4` modelinin indirilmiş olması gerekir:
  ```bash
  ollama pull gemma4
  ```

### 2. Proje Kurulumu
Projeyi klonladıktan veya indirdikten sonra, proje dizininde bir sanal ortam oluşturup gerekli kütüphaneleri yükleyin:

```bash
# Sanal ortam oluşturma
python -m venv rag_env

# Sanal ortamı aktifleştirme (Windows için)
rag_env\Scripts\activate

# Gerekli bağımlılıkları yükleme
pip install -r requirements.txt
```

## 🔧 Kullanım

### Adım 1: Veritabanını Doldurma
Github'da büyük veritabanı klasörleri tutulmadığı için projeyi ilk indirdiğinizde ChromaDB'yi oluşturmanız gerekmektedir:
```bash
python veri_fabrikasi.py
```
Bu script, `data/meslekler_listesi.txt` dosyasındaki isimleri alacak ve arka planda çalışarak her bir meslek için veritabanını (`chroma_db_gercek`) inşa edecektir.

### Adım 2: Asistanı Başlatma

**Seçenek A: Web Arayüzü (Önerilen)**
Streamlit ile tarayıcınızda çalışan görsel arayüzü başlatmak için:
```bash
streamlit run app.py
```

**Seçenek B: Terminal (CLI) Arayüzü**
Eğer sadece terminal üzerinden hızlıca soru sormak isterseniz:
```bash
python main.py
```

### Adım 3: Stres Testi
Sistemin alakasız sorulara (örneğin uzaylılar veya nükleer sızıntı gibi) nasıl mantıklı sınırlar çizdiğini (Failsafe) test etmek isterseniz:
```bash
python test_stres.py
```

## 📂 Proje Yapısı

- `app.py`: Streamlit web arayüzü.
- `main.py`: Terminal kullanım arayüzü.
- `rag_engine.py`: İki aşamalı RAG (HyDE) mekanizmasının ve LLM/ChromaDB entegrasyonunun bulunduğu çekirdek motor.
- `db_manager.py`: JSON ve ChromaDB üzerindeki kayıt, güncelleme ve silme işlemlerini yöneten sınıf.
- `veri_fabrikasi.py`: Verilen bir meslek listesinden RAG için sentetik/gerçekçi dertler üreten otonom bot.
- `data/`: Kaynak TXT ve JSON yedeklerinin bulunduğu klasör.
- `chroma_db_gercek/`: (Çalıştırdıktan sonra oluşur) Vektörlerin tutulduğu yerel veritabanı.
