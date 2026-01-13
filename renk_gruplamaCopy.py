"""
Delta E ile Renk Gruplama
=========================
LAB uzayında benzer renkleri gruplar.

Delta E (ΔE): İki renk arasındaki algısal farkı ölçen metrik
- ΔE < 1: İnsan gözü farkı algılayamaz
- ΔE < 5: Yakından bakınca fark edilebilir
- ΔE < 20: Benzer renkler
- ΔE > 50: Tamamen farklı renkler
"""

import cv2  # OpenCV - Görüntü işleme kütüphanesi
import numpy as np  # NumPy - Sayısal işlemler için
from collections import defaultdict  # Otomatik liste oluşturan sözlük


def delta_e_cie76(lab1, lab2):
    """
    İki LAB rengi arasındaki Delta E mesafesi (CIE76 formülü).
    
    CIE76 formülü: En basit Delta E hesaplama yöntemi
    ΔE = √[(L₁-L₂)² + (a₁-a₂)² + (b₁-b₂)²]
    
    Bu 3 boyutlu Öklid mesafesidir (uzaydaki iki nokta arası mesafe gibi)
    LAB renk uzayı insan algısına göre tasarlandığı için bu mesafe
    algısal renk farkını iyi temsil eder.
    
    Args:
        lab1: İlk LAB rengi (L, a, b) - tuple veya list
        lab2: İkinci LAB rengi (L, a, b)
    
    Returns:
        float: Delta E değeri (0'a yakın = benzer, büyük = farklı)
    """
    # İki renk arasındaki farkın karesinin toplamının karekökü (Öklid mesafesi)
    return np.sqrt(np.sum((np.array(lab1) - np.array(lab2)) ** 2))


def dominant_renkler_bul(goruntu: np.ndarray, k: int = 5) -> np.ndarray:
    """
    K-Means ile görüntüdeki baskın renkleri bulur.
    
    K-Means Kümeleme Algoritması:
    1. Görüntüdeki tüm pikselleri al
    2. Bunları K gruba (küme) ayır
    3. Her kümenin merkezi (ortalama rengi) = baskın renk
    
    Neden K-Means?
    - Görüntüdeki milyonlarca rengi K ana renge indirger
    - En çok kullanılan renkleri bulur
    - Hızlı ve etkili
    
    Args:
        goruntu: BGR formatında görüntü
        k: Bulmak istediğin renk sayısı (örn: 5 ana renk)
    
    Returns:
        np.ndarray: Baskın renkler (LAB formatında, k satır x 3 sütun)
    """
    # LAB'a dönüştür (LAB uzayı Delta E hesaplamaları için gerekli)
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    
    # Görüntüyü 2D diziye çevir: (yükseklik × genişlik) satır, 3 sütun (L, a, b)
    # reshape(-1, 3): -1 = otomatik hesapla, 3 = 3 sütun (L, a, b)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    
    # K-Means parametreleri
    # TERM_CRITERIA: Ne zaman duracağını belirler
    # EPS: Değişim bu değerden küçükse dur
    # MAX_ITER: Maksimum 100 iterasyon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # K-Means kümeleme algoritmasını çalıştır
    # k: Küme sayısı
    # None: Her pikselin hangi kümeye ait olduğu (başlangıçta bilinmiyor)
    # criteria: Durma kriteri
    # 10: Algoritma 10 farklı başlangıçla çalışır, en iyisini seçer
    # KMEANS_RANDOM_CENTERS: Başlangıç merkezlerini rastgele seç
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # centers: K küme merkezleri = K baskın renk (LAB formatında)
    return centers


def renkleri_grupla(renkler_lab: list, esik: float = 10.0) -> dict:
    """
    Delta E eşiğine göre benzer renkleri gruplar.
    
    Çalışma prensibi:
    1. İlk rengi al, yeni bir grup oluştur
    2. Diğer renklerle Delta E hesapla
    3. Eşikten küçükse aynı gruba ekle
    4. İşaretlenmemiş renklerle devam et
    
    Örnek: 8 renk var, ΔE < 20 eşiği
    - Renk 1, 3, 5 birbirine benzer → Grup 0
    - Renk 2, 7 birbirine benzer → Grup 1
    - Renk 4, 6, 8 birbirine benzer → Grup 2
    
    Args:
        renkler_lab: LAB renk listesi [(L, a, b), (L, a, b), ...]
        esik: Maksimum Delta E değeri (aynı grup için kabul edilir)
    
    Returns:
        dict: {grup_id: [renkler]} şeklinde sözlük
              Örn: {0: [(50,20,30), (52,21,29)], 1: [(80,10,40)]}
    """
    # defaultdict: Yeni anahtar eklendiğinde otomatik boş liste oluşturur
    gruplar = defaultdict(list)
    atanmis = set()  # Hangi renklerin gruba atandığını takip et (set: hızlı arama)
    grup_id = 0  # Grup numarası
    
    # Her renk için döngü
    for i, renk1 in enumerate(renkler_lab):
        if i in atanmis:  # Bu renk zaten bir gruba atandıysa atla
            continue
        
        # Yeni grup oluştur ve bu rengi ekle
        gruplar[grup_id].append(renk1)
        atanmis.add(i)  # Bu rengi işaretle
        
        # Bu renkle benzer olan diğer renkleri bul
        for j, renk2 in enumerate(renkler_lab):
            if j in atanmis:  # Zaten atanmış renkleri atla
                continue
            
            # İki renk arasındaki Delta E'yi hesapla
            de = delta_e_cie76(renk1, renk2)
            
            # Eşikten küçükse aynı gruba ekle
            if de <= esik:
                gruplar[grup_id].append(renk2)
                atanmis.add(j)  # Bu rengi işaretle
        
        grup_id += 1  # Bir sonraki grup için ID artır
    
    # defaultdict'i normal dict'e çevir ve döndür
    return dict(gruplar)


def renk_haritasi_olustur(goruntu: np.ndarray, esik: float = 15.0) -> np.ndarray:
    """
    Benzer renkleri gruplandırarak renk haritası oluşturur.
    
    Amaç: Görüntüdeki benzer renkleri tek renge indirgemek
    Kullanım: Renk sayısını azaltma, posterize efekti
    
    Çalışma prensibi:
    1. Baskın renkleri bul (K-Means ile)
    2. Her pikseli en yakın baskın renge ata
    3. Yeni görüntü oluştur (sadece baskın renklerle)
    
    Args:
        goruntu: BGR formatında görüntü
        esik: Delta E eşiği (şu an kullanılmıyor, gelecekte grup için)
    
    Returns:
        np.ndarray: Gruplandırılmış renk haritası (posterize görüntü)
    """
    # LAB'a dönüştür (Delta E hesaplamaları için)
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Baskın renkleri bul (8 ana renk)
    dominant = dominant_renkler_bul(goruntu, k=8)
    
    # Sonuç için boş görüntü oluştur
    h, w = goruntu.shape[:2]  # Yükseklik ve genişlik
    sonuc = np.zeros_like(goruntu)  # Aynı boyutta boş görüntü
    
    # Her piksel için döngü (yavaş ama anlaşılır yöntem)
    for i in range(h):
        for j in range(w):
            piksel_lab = lab[i, j]  # Bu pikselin LAB değeri
            min_mesafe = float('inf')  # Sonsuz ile başla
            en_yakin = dominant[0]  # En yakın renk (şimdilik ilk renk)
            
            # Hangi baskın renge en yakın?
            for renk in dominant:
                mesafe = delta_e_cie76(piksel_lab, renk)  # Delta E hesapla
                if mesafe < min_mesafe:  # Daha yakın renk bulundu
                    min_mesafe = mesafe
                    en_yakin = renk
            
            # En yakın baskın rengi bu piksele ata
            # LAB'dan BGR'ye dönüştür (ekranda göstermek için)
            lab_piksel = np.uint8([[en_yakin]])
            bgr_piksel = cv2.cvtColor(lab_piksel.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)
            sonuc[i, j] = bgr_piksel[0, 0]
    
    return sonuc


# Bu modül direkt çalıştırılırsa (python renk_gruplama.py)
if __name__ == "__main__":
    print("Delta E Renk Gruplama Modülü")
    
    # Örnek: İki renk arasındaki fark
    lab1 = (50, 20, 30)  # İlk LAB rengi
    lab2 = (55, 22, 28)  # İkinci LAB rengi
    de = delta_e_cie76(lab1, lab2)
    print(f"LAB{lab1} ve LAB{lab2} arası Delta E: {de:.2f}")

