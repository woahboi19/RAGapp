from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from db_manager import DBManager
import time
import unicodedata
import os

def slugify(value):
    """Meslek isminden benzersiz ve düzgün bir ID (meslek_id) oluşturur (Örn: Su Tesisatçısı -> su_tesisatcisi)."""
    value = str(value).lower().replace(' ', '_')
    # Türkçe karakterleri (ş,ğ,ü vb.) İngilizce karşılıklarına çevirir.
    return unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')

def baslat():
    print("="*60)
    print("🤖 OTONOM VERİ FABRİKASI BAŞLATILIYOR 🤖")
    print("="*60)
    
    # 1. Ollama'ya bağlan (gemma4 modeli kullanılacak)
    llm = OllamaLLM(model="gemma4")
    
    # 2. Veritabanı Yöneticisini Başlat
    db = DBManager(json_path="data/meslekler_db.json", chroma_dir="./chroma_db_gercek")
    
    # Üretilmesini istediğimiz mesleklerin listesini TXT dosyasından oku
    meslekler_listesi = []
    txt_yolu = "data/meslekler_listesi.txt"
    if os.path.exists(txt_yolu):
        with open(txt_yolu, "r", encoding="utf-8") as f:
            for line in f:
                satir = line.strip()
                if satir:
                    meslekler_listesi.append(satir)
    else:
        print(f"[HATA] {txt_yolu} dosyası bulunamadı!")
        return
    
    # Prompt Kalıbı (Günlük Dil ve Eş Anlamlılar Eklenmiş Şablon)
    prompt_template = """Sen uzman bir veri hazırlama asistanısın. Görevin, verilen mesleğin halk arasındaki diğer yaygın/günlük isimlerini (abartmadan, normal bir dille) yazmak ve ardından bu mesleğin günlük hayatta çözdüğü spesifik dertleri SADECE 3 madde halinde yazmaktır. Ekstra yorum yapma.

ÖRNEK GİRDİ:
Meslek: Veteriner Hekim
ÖRNEK ÇIKTI:
Günlük İsimler: Veteriner, Hayvan Doktoru
Çözdüğü Sorunlar:
- Kedim sabahtan beri hiçbir şey yemiyor ve sürekli kusuyor.
- Köpeğimin aşı vakti geldi ama şu an çok halsiz.
- Kuşumun kanadı kapıya sıkıştı, acı çekiyor.

ÖRNEK GİRDİ:
Meslek: Su Tesisatçısı
ÖRNEK ÇIKTI:
Günlük İsimler: Tesisatçı, Su Ustası
Çözdüğü Sorunlar:
- Üst katın tuvaletinden bizim banyoya su damlıyor.
- Mutfak lavabosu tamamen tıkandı, su gitmiyor.
- Kombinin altındaki borulardan su sızıyor.

ŞİMDİ SIRA SENDE:
Meslek: {meslek_ismi}
ÇIKTI:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["meslek_ismi"])
    chain = prompt | llm
    
    print(f"Toplam {len(meslekler_listesi)} meslek işlenecek. Lütfen Ollama'nın çalışmasını bekleyin...\n")
    
    # Döngü ile hepsini otonom üret ve kaydet
    for meslek in meslekler_listesi:
        meslek_id = "meslek_" + slugify(meslek)
        
        # Eğer bu meslek zaten veritabanında varsa atla (veri tekrarını önler)
        if meslek_id in db.get_all():
            print(f"[ATLANDI] {meslek} zaten veritabanında mevcut.")
            continue
            
        print(f"-> Üretiliyor: {meslek}...")
        try:
            # Yapay zekaya soruyu sorup cevabı (açıklamayı) al
            aciklama = chain.invoke({"meslek_ismi": meslek})
            
            # Üretilen cevabı DB Manager aracılığıyla hem JSON hem de Chroma'ya kaydet
            db.add_profession(meslek_id=meslek_id, isim=meslek, aciklama=aciklama.strip())
            
        except Exception as e:
            print(f"[HATA] {meslek} üretilirken bir hata oluştu: {e}")
            
        # Sistemde tıkanma olmaması için model dinlendiriliyor
        time.sleep(1)

    print("\n="*60)
    print("✅ Üretim tamamlandı! Veriler ChromaDB'ye ve data/meslekler_db.json dosyasına başarıyla kaydedildi.")

if __name__ == "__main__":
    baslat()
