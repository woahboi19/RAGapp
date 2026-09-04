import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class DBManager:
    def __init__(self, json_path="data/meslekler_db.json", chroma_dir="./chroma_db_gercek"):
        self.json_path = json_path
        self.chroma_dir = chroma_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # JSON veritabanını yükle (yoksa oluştur)
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)
            
        # Vektör Veritabanını (Chroma) başlat
        self.vectordb = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embeddings)
        
    def add_profession(self, meslek_id, isim, aciklama):
        """
        Yeni bir meslek ekler veya var olanı günceller.
        Hem JSON dosyasına hem de ChromaDB'ye (aynı ID ile) kaydeder.
        """
        self.db[meslek_id] = {
            "isim": isim,
            "aciklama": aciklama
        }
        self._save_json()
        
        # ChromaDB için doküman formatına getir
        content = f"### {isim}\n{aciklama}"
        doc = Document(page_content=content, metadata={"source": meslek_id})
        
        # Aynı ID ile kaydetmek, eski verinin üzerine yazılmasını (update) sağlar.
        self.vectordb.add_documents(documents=[doc], ids=[meslek_id])
        print(f"[BAŞARILI] {isim} ({meslek_id}) veritabanına eklendi/güncellendi.")
        
    def delete_profession(self, meslek_id):
        """Belirtilen ID'ye sahip mesleği her iki sistemden de siler."""
        if meslek_id in self.db:
            isim = self.db[meslek_id]["isim"]
            del self.db[meslek_id]
            self._save_json()
            
            try:
                self.vectordb.delete(ids=[meslek_id])
                print(f"[SİLİNDİ] {isim} ({meslek_id}) başarıyla silindi.")
            except Exception as e:
                print(f"ChromaDB'den silinirken uyarı: {e}")
        else:
            print(f"[HATA] {meslek_id} ID'li meslek bulunamadı.")
            
    def _save_json(self):
        """JSON dosyasını diske kaydeder."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.db, f, ensure_ascii=False, indent=4)
            
    def get_all(self):
        """Tüm meslekleri getirir."""
        return self.db


