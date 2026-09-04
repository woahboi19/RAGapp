import sys
from rag_engine import RAGEngine

def main():
    print("="*50)
    print("🤖 UZMAN YÖNLENDİRME ASİSTANI (RAG) BAŞLADI")
    print("Çıkmak için 'q' veya 'quit' yazın.")
    print("="*50)
    
    # Motoru başlat (Gemma 4 modeli ve doğru veritabanı)
    # RAGEngine sınıfından bir nesne (object) üretiyoruz.
    # Bu esnada arka planda Ollama modeli ve VectorDB yükleniyor.
    try:
        engine = RAGEngine(model_name="gemma4", db_dir="./chroma_db_gercek")
        print("\nSistem başarıyla başlatıldı. Sohbet etmeye başlayabilirsiniz.\n")
    except Exception as e:
        print(f"\nSistem başlatılırken bir hata oluştu: {e}")
        print("Lütfen Ollama'nın çalıştığından ve doğru modelin yüklü olduğundan emin olun.")
        sys.exit(1)

    # 2. Ana Sohbet Döngüsü (REPL - Read Eval Print Loop)
    # Kullanıcı çıkış yapana kadar sürekli girdi alıp cevap döndüren sonsuz bir döngü kuruyoruz.
    while True:
        try:
            # Kullanıcıdan girdi alıyoruz
            user_input = input("Siz: ")
            
            # Girdi boşsa veya çıkış komutlarından biriyse döngüyü kırıyoruz (uygulamayı kapatıyoruz)
            if not user_input.strip():
                continue
            
            if user_input.lower() in ['exit', 'quit', 'çıkış']:
                print("Chatbot kapatılıyor. İyi günler!")
                break
                
            # 3. Modelden Cevap Alma
            # Kullanıcının girdisini rag_engine modülümüze gönderiyoruz
            print("Chatbot Düşünüyor...")
            response = engine.get_response(user_input)
            
            # 4. Cevabı Ekrana Yazdırma
            print(f"Chatbot: {response}\n")
            
        except KeyboardInterrupt:
            # Kullanıcı CTRL+C'ye basarsa programın temiz kapanması için
            print("\nChatbot kapatılıyor. İyi günler!")
            break
        except Exception as e:
            # Beklenmedik bir hata olursa ekrana yazdırıyoruz ancak programın çökmemesi için döngü devam ediyor
            print(f"Bir hata oluştu: {e}")

# Bu dosya doğrudan çalıştırıldığında main() fonksiyonunu çağırır.
# Eğer bu dosya başka bir modül tarafından import edilirse main() otomatik çalışmaz.
if __name__ == "__main__":
    main()
