"""
Renkli Görüntü İşleme - Ana Demo
================================
Tüm modülleri kullanarak tam bir demo uygulaması.

Görevler:
1. RGB → HSV, LAB dönüşümü
2. HSV ile renk segmentasyonu
3. Delta E ile renk gruplama
4. Morfolojik iyileştirme
"""

import cv2
import numpy as np
import os

from segmentasyon import renk_segmentasyonu, maskeyi_uygula, RENK_ARALIKLARI
from morfoloji import tam_iyilestirme, acma, kapama
from renk_gruplama import dominant_renkler_bul, delta_e_cie76


def ornek_goruntu_olustur(boyut: tuple = (400, 600)) -> np.ndarray:
    """Test için örnek görüntü oluşturur."""
    h, w = boyut
    goruntu = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Arka plan - açık mavi gökyüzü
    goruntu[:] = (235, 206, 135)  # BGR - Açık mavi
    
    # Yeşil yapraklar (sol üst)
    cv2.rectangle(goruntu, (50, 50), (200, 150), (34, 139, 34), -1)
    cv2.ellipse(goruntu, (125, 100), (80, 50), 0, 0, 360, (0, 128, 0), -1)
    
    # Kırmızı araba (ortada)
    cv2.rectangle(goruntu, (220, 200), (380, 280), (0, 0, 200), -1)
    cv2.rectangle(goruntu, (250, 240), (350, 280), (50, 50, 50), -1)  # Tekerlek
    
    # Sarı güneş (sağ üst)
    cv2.circle(goruntu, (500, 80), 50, (0, 255, 255), -1)
    
    # Mor çiçekler (sağ alt)
    for x, y in [(450, 320), (500, 340), (550, 310)]:
        cv2.circle(goruntu, (x, y), 20, (128, 0, 128), -1)
    
    return goruntu


def renk_uzayi_goster(goruntu: np.ndarray, baslik: str = "Renk Uzayları"):
    """RGB, HSV ve LAB görünümlerini yan yana gösterir."""
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    
    # Her kanalı ayrı görselleştir
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    # Birleştir
    h_renkli = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
    
    print(f"\n=== {baslik} ===")
    print("Orijinal görüntü boyutu:", goruntu.shape)
    print("HSV - H aralığı:", h.min(), "-", h.max())
    print("LAB - L aralığı:", l.min(), "-", l.max())
    
    return {'hsv': hsv, 'lab': lab, 'h': h, 's': s, 'v': v, 'l': l, 'a': a, 'b': b}


def segmentasyon_demo(goruntu: np.ndarray, hedef_renk: str = 'kirmizi'):
    """Renk segmentasyonu demo."""
    print(f"\n=== {hedef_renk.upper()} Segmentasyonu ===")
    
    # Ham maske
    maske = renk_segmentasyonu(goruntu, hedef_renk)
    print(f"Ham maske - Beyaz piksel sayısı: {np.sum(maske > 0)}")
    
    # Morfolojik iyileştirme
    maske_iyilestirilmis = tam_iyilestirme(maske, acma_boyut=3, kapama_boyut=7)
    print(f"İyileştirilmiş - Beyaz piksel sayısı: {np.sum(maske_iyilestirilmis > 0)}")
    
    # Segmente edilmiş nesne
    nesne = maskeyi_uygula(goruntu, maske_iyilestirilmis)
    
    return maske, maske_iyilestirilmis, nesne


def delta_e_demo(goruntu: np.ndarray):
    """Delta E ile renk analizi demo."""
    print("\n=== Delta E Renk Analizi ===")
    
    # Baskın renkleri bul
    dominant = dominant_renkler_bul(goruntu, k=5)
    
    print(f"Baskın renkler (LAB):")
    for i, renk in enumerate(dominant):
        print(f"  {i+1}. L={renk[0]:.1f}, a={renk[1]:.1f}, b={renk[2]:.1f}")
    
    # Renk çiftleri arası Delta E
    print("\nRenkler arası Delta E:")
    for i in range(len(dominant)):
        for j in range(i+1, len(dominant)):
            de = delta_e_cie76(dominant[i], dominant[j])
            print(f"  Renk {i+1} - Renk {j+1}: ΔE = {de:.2f}")
    
    return dominant


def sonuclari_kaydet(sonuclar: dict, cikti_klasoru: str = "sonuclar"):
    """Tüm sonuçları dosyaya kaydet."""
    os.makedirs(cikti_klasoru, exist_ok=True)
    
    for isim, goruntu in sonuclar.items():
        yol = os.path.join(cikti_klasoru, f"{isim}.png")
        cv2.imwrite(yol, goruntu)
        print(f"Kaydedildi: {yol}")


def ana():
    """Ana demo fonksiyonu."""
    print("=" * 60)
    print("  RENKLİ GÖRÜNTÜ İŞLEME VE RENK UZAYLARI DEMO")
    print("=" * 60)
    
    # 1. Örnek görüntü oluştur veya yükle
    goruntu = ornek_goruntu_olustur()
    print("\nÖrnek görüntü oluşturuldu.")
    
    # 2. Renk uzayları göster
    uzaylar = renk_uzayi_goster(goruntu)
    
    # 3. Segmentasyon demoları
    sonuclar = {'orijinal': goruntu}
    
    for renk in ['kirmizi', 'yesil', 'sari', 'mor']:
        try:
            maske, maske_iyi, nesne = segmentasyon_demo(goruntu, renk)
            sonuclar[f'{renk}_maske'] = maske
            sonuclar[f'{renk}_iyilestirilmis'] = maske_iyi
            sonuclar[f'{renk}_nesne'] = nesne
        except Exception as e:
            print(f"{renk} segmentasyonu başarısız: {e}")
    
    # 4. Delta E analizi
    dominant = delta_e_demo(goruntu)
    
    # 5. Sonuçları kaydet
    sonuclari_kaydet(sonuclar)
    
    print("\n" + "=" * 60)
    print("  DEMO TAMAMLANDI")
    print("=" * 60)
    print("\nSonuçlar 'sonuclar/' klasörüne kaydedildi.")
    
    return sonuclar


if __name__ == "__main__":
    import sys
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sonuclar = ana()
