import warnings
# Gereksiz uyarıları gizle
warnings.filterwarnings("ignore")

from rag_engine import RAGEngine

def stres_testi():
    print("="*70)
    print("🚀 STRES TESTİ BAŞLATILIYOR (FAILSAFE KONTROLÜ) 🚀")
    print("="*70)
    
    # Motoru başlat (Gerçek veritabanını kullanıyoruz)
    print("Motor yükleniyor...\n")
    engine = RAGEngine(model_name="gemma4", db_dir="./chroma_db_gercek")
    
    sorular = [
        # 1. Karmaşık Çoklu İhtiyaç (Oto, Sağlık, Hukuk)
        "Arabamla kaza yaptım, kaporta tamamen göçtü. Üstüne boynum çok feci ağrıyor hareket ettiremiyorum ve karşı taraf bana tazminat davası açmış.",
        
        # 2. Yanıltıcı Bağlam (Hayvanlar geçiyor ama sorun insan sağlığı)
        "Dün gece aslanların ceylanları avladığı o vahşi belgeseli izlerken midem çok feci bulandı, sabahtan beri kusuyorum ve karnıma kramplar giriyor.",
        
        # 3. Şiirsel / Soyut Anlatım (Gitar/Enstrüman Tamiri)
        "Ruhumun aynası olan o eski ahşap dostumun telleri koptu, artık bana o güzel ezgileri fısıldamıyor. Onu hayata döndürmem lazım.",
        
        # 4. Failsafe (Güvenlik Duvarı) Şaşırtmacası (Mecaz anlamda cin/büyü kullanımı)
        "Oyun bilgisayarım o kadar yavaşladı ki sanki içine cin girmiş gibi, sürekli donuyor. Ayrıca ekran kartından uçak motoru gibi ses geliyor.",
        
        # 5. Aşırı Spesifik Endüstriyel Sorun
        "Fabrikamdaki CNC dik işlem merkezinde G-Code hatası alıyorum, ayrıca spindle motorundan tıkırtılar geliyor, işleme yapamıyoruz.",
        
        # 6. Psikolojik vs Fiziksel Karmaşası
        "Her şey üstüme üstüme geliyor, geceleri nefes alamıyorum, kalbim sıkışıyor ama hastanede EKG çektirdim fiziki hiçbir sorunum yokmuş.",
        
        # 7. Absürt / İmkansız İstek (Failsafe %100 çalışmalı)
        "Geçen hafta Mars'tan getirdiğim uzaylı evcil hayvanım radyasyon sızdırıyor, onun için acil bir yuva yapmam lazım."
    ]
    
    for i, soru in enumerate(sorular, 1):
        print(f"[{i}] KULLANICI: {soru}")
        print(">>> ASİSTAN DÜŞÜNÜYOR...")
        try:
            cevap = engine.get_response(soru)
            print(f"[ASİSTAN]:\n{cevap}\n")
        except Exception as e:
            print(f"[HATA]: {e}\n")
        print("-" * 70)

if __name__ == "__main__":
    stres_testi()
