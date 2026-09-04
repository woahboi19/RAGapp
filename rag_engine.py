"""
rag_engine.py
Bu dosya projenin RAG (Retrieval-Augmented Generation) mekanizmasını ve Dil Modeli (LLM) yönetimini içerir.
Veri kaynağı (pdf, veritabanı vs.) şimdilik belirsiz olduğu için modüler bir şekilde tasarlandı.
Böylece ileride bir veri kaynağı eklemek istediğimizde sadece ilgili fonksiyonu güncelleyeceğiz.
"""

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

class RAGEngine:
    def __init__(self, model_name="gemma4", db_dir="./chroma_db"):
        """
        RAGEngine sınıfı başlatıldığında model ve embeddings ayarlarını yapar.
        :param model_name: Kullanılacak Ollama modelinin adı (Örn: llama3, mistral).
        :param db_dir: ChromaDB'nin kaydedileceği/yükleneceği klasör yolu.
        """
        self.model_name = model_name
        self.db_dir = db_dir
        
        # 1. LLM Kurulumu:
        # LangChain üzerinden yerel cihazımızda çalışan Ollama'ya bağlanıyoruz.
        # Bu sayede verilerimiz internete çıkmadan tamamen cihazımızda işleniyor.
        self.llm = OllamaLLM(model=self.model_name)
        
        # 2. Embeddings Kurulumu:
        # Metinleri sayısal vektörlere dönüştürmek için yerel bir model (HuggingFace) kullanıyoruz.
        # all-MiniLM-L6-v2, İngilizce için oldukça hızlı ve performanslı bir modeldir.
        # İleride Türkçe ağırlıklı bir projeye geçilirse buradaki model değiştirilebilir.
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 3. Vector Database (Vektör Veritabanı) Yüklenmesi:
        # Veritabanını başlatıyoruz. Eğer daha önceden veri eklenmemişse şimdilik boş bir veritabanı olur.
        self.vectordb = self._init_vector_db()
        


    def _init_vector_db(self):
        """
        ChromaDB veritabanını başlatır. Eğer belirtilen klasörde bir DB varsa onu yükler.
        """
        # Şimdilik DB'yi sadece embeddings ile başlatıyoruz. 
        # İleride veri yüklemek için `Chroma.from_documents` kullanılacak.
        return Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        
    def get_response(self, user_input: str) -> str:
        """
        Kullanıcıdan gelen mesajı İki Aşamalı (HyDE) RAG mantığıyla işler.
        """
        # 1. AŞAMA (HyDE - Soru Zenginleştirme):
        hyde_prompt = f"""Sen bir meslek tahmin asistanısın. Kullanıcının sorununa göre en uygun meslekleri Türkiye'deki yaygın günlük adlarıyla (Örn: Bilgisayar Tamircisi, Oto Kaportacı, Enstrüman Tamircisi, Usta vb.) Türkçe olarak yaz. Aşırı akademik unvanlar (Bilişim Uzmanı, Elektronik Mühendisi vb.) yerine pratik esnaf ve usta isimleri kullan.
Eğer sorun fantastik, doğaüstü veya mantıksızsa (Örn: UFO, büyü, uzaylı, nükleer sızıntı vb.) sadece 'YOK' yaz. Mecazi anlamda kullanılan 'cin gibi', 'büyü gibi' kelimelerine kanma, sorunun özüne bak.

ÖRNEKLER:
Soru: Kedim kusuyor.
Cevap: Veteriner Hekim

Soru: Dün gece bahçeme UFO düştü.
Cevap: YOK

Soru: Evin çatısı akıyor, eşimle boşanıyoruz ve dişim ağrıyor.
Cevap: Su Tesisatçısı, Boşanma Avukatı, Diş Hekimi

Soru: Bilgisayarım cin çarpmış gibi donuyor, ses çıkarıyor.
Cevap: Bilgisayar Tamircisi

ŞİMDİ SIRA SENDE:
Soru: {user_input}
Cevap:"""
        
        predicted_professions = self.llm.invoke(hyde_prompt).strip()
        print(f"    [HyDE TAHMİNİ]: {predicted_professions}")
        
        # Eğer Failsafe tetiklendiyse direkt durdur:
        if "YOK" in predicted_professions.upper():
            return "Üzgünüm, veri tabanımda bu sorunu çözecek uygun bir uzman bulunmuyor."
            
        # 2. AŞAMA (Vektör Arama): LLM'in ürettiği anahtar kelimelerle arama yap
        # K=10 yapıyoruz ki daha fazla günlük/sıradan meslek de havuza dahil olabilsin.
        docs = self.vectordb.similarity_search(predicted_professions, k=10)
        
        context = ""
        for i, doc in enumerate(docs):
            context += f"\n--- Uzman {i+1} ---\n{doc.page_content}\n"
            
        # 3. AŞAMA (Nihai Cevap Üretimi):
        # YENİ EKLENEN KISITLAMA: Modelin 'Maden Mühendisi' gibi uçuk kaçık akademik tavsiyeler vermesini engellemek için
        # "günlük hayata uygun, en basit ve mantıklı" çözümleri önceliklendirmesini söylüyoruz.
        final_prompt = f"""Sen çok zeki ve pratik bir yönlendirme asistanısın.
Kullanıcının sorununu okuyacaksın ve AŞAĞIDAKİ KATALOG'da yer alan uzmanlardan SADECE BİR VEYA BİRKAÇINI seçeceksin.

ÖNEMLİ KURALLAR:
1. Asla KATALOG'da olmayan bir mesleği önerme.
2. SADECE GÜNLÜK HAYATA UYGUN, EN BASİT VE MANTIKLI çözümü seç.
3. Seçtiğin uzmanların adını kalın (bold) yaz.
4. Kullanıcıya doğrudan hitap et (Örn: "Şu uzmana görünmelisiniz çünkü...").
5. Açıklamanı çok kısa, net ve anlaşılır tut.
6. [ÇOK ÖNEMLİ]: Eğer Katalogdaki HİÇBİR uzman kullanıcının sorununu DOĞRUDAN çözmüyorsa, yaratıcı ve mantıksız bağlantılar kurmaya ÇALIŞMA! (Örn: Gitar koptuysa Saat Tamircisi önerme!). Bunun yerine direkt olarak 'Üzgünüm, şu anki uzman ağımızda bu sorunu tam olarak çözecek bir uzman bulunmuyor' yaz ve bitir.

Kullanıcının Sorunu: '{user_input}'

KATALOG (Mevcut Uzmanlar ve Çözdükleri Örnek Sorunlar):
{context}

Cevabın:"""
        
        return self.llm.invoke(final_prompt)
