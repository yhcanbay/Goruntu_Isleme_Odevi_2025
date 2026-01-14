"""
Görüntü İşleme Ödevi - Renk Uzayları ve Segmentasyon
Proje 4: RGB, HSV, LAB renk uzayları ile renk tabanlı segmentasyon

Görevler:
1. RGB'den HSV ve LAB'a dönüşüm
2. HSV ile çoklu renk segmentasyonu
3. LAB Delta E ile renk gruplama
4. Morfolojik işlemlerle iyileştirme
"""

import cv2
import numpy as np
from segmentasyon import renk_segmentasyonu, maskeyi_uygula
from morfoloji import tam_iyilestirme, acma, kapama
from renk_gruplama import dominant_renkler_bul, delta_e_cie76, renkleri_grupla


def gorev1_renk_uzayi_donusumleri(goruntu: np.ndarray):
    """Görev 1: RGB'den HSV ve LAB'a dönüşüm"""
    
    # RGB'den HSV'ye dönüştürüyorum
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    # RGB'den LAB'a dönüştürüyorum
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    
    # Sonuçları gösteriyorum
    cv2.imshow('Orijinal Goruntu (RGB)', goruntu)
    cv2.imshow('HSV Goruntu', hsv)
    cv2.imshow('LAB Goruntu', lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gorev2_hsv_segmentasyon(goruntu: np.ndarray):
    """Görev 2: HSV ile çoklu renk segmentasyonu"""
    
    renkler = ['kirmizi', 'yesil', 'mavi']
    maskeler = {}
    segmente_nesneler = {}
    
    # Her renk için segmentasyon yapıyorum
    for renk in renkler:
        maske = renk_segmentasyonu(goruntu, renk)
        segmente_nesne = maskeyi_uygula(goruntu, maske)
        
        maskeler[renk] = maske
        segmente_nesneler[renk] = segmente_nesne
    
    # Sonuçları gösteriyorum
    for renk in renkler:
        # Maske ve segmente edilmiş görüntüyü yan yana koyuyorum
        maske_bgr = cv2.cvtColor(maskeler[renk], cv2.COLOR_GRAY2BGR)
        birlesik = np.hstack([maske_bgr, segmente_nesneler[renk]])
        
        # Ekrana sığması için boyutlandırıyorum
        yukseklik, genislik = birlesik.shape[:2]
        yeni_genislik = int(genislik * 0.5)
        yeni_yukseklik = int(yukseklik * 0.5)
        birlesik_kucuk = cv2.resize(birlesik, (yeni_genislik, yeni_yukseklik))
        
        cv2.imshow(f'{renk.capitalize()} - Maske | Segmente', birlesik_kucuk)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return maskeler



def gorev3_delta_e_gruplama(goruntu: np.ndarray):
    """Görev 3: LAB Delta E ile renk gruplama (GERÇEK KULLANIM)"""

    # 1️⃣ K-Means ile baskın renkleri bul
    k = 8
    dominant = dominant_renkler_bul(goruntu, k=k)  # LAB formatında

    # 2️⃣ Delta E ile renkleri grupla
    renkler_liste = [tuple(r) for r in dominant]
    gruplar = renkleri_grupla(renkler_liste, esik=20.0)

    print(f"\n=== Delta E Gruplama Sonuçları ===")
    print(f"Başlangıç renk sayısı (K-means): {k}")
    print(f"Delta E ile gruplanmış renk sayısı: {len(gruplar)}")
    
    # Her grubun detaylarını yazdır
    for grup_id, grup_renkleri in gruplar.items():
        print(f"Grup {grup_id}: {len(grup_renkleri)} renk")

    # 3️⃣ Her grubun ortalama rengini al (LAB)
    grup_renkleri = []
    for grup in gruplar.values():
        grup = np.array(grup)
        ortalama_lab = np.mean(grup, axis=0)
        grup_renkleri.append(ortalama_lab)

    # 4️⃣ Delta E ile gruplanmış renk paleti oluştur
    grup_sayisi = len(grup_renkleri)
    renk_paleti = np.zeros((100, grup_sayisi * 100, 3), dtype=np.uint8)

    for i, renk_lab in enumerate(grup_renkleri):
        lab_pixel = np.uint8([[renk_lab]])
        bgr_pixel = cv2.cvtColor(lab_pixel.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)
        renk_bgr = bgr_pixel[0, 0]
        renk_paleti[:, i*100:(i+1)*100] = renk_bgr

    # 5️⃣ Paleti göster
    cv2.imshow('Delta E ile Gruplanmis Renk Paleti', renk_paleti)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



def gorev4_morfolojik_iyilestirme(goruntu: np.ndarray, maskeler: dict):
    """Görev 4: Morfolojik işlemlerle iyileştirme"""
    
    for renk, maske in maskeler.items():
        # Morfolojik işlemler uyguluyorum
        maske_acma = acma(maske, kernel_boyut=3)  # Gürültüyü temizliyorum
        maske_kapama = kapama(maske_acma, kernel_boyut=7)  # Delikleri dolduruyorum
        maske_iyilestirilmis = tam_iyilestirme(maske, acma_boyut=3, kapama_boyut=7)
        
        # İyileştirilmiş maskeyi uyguluyorum
        nesne_iyilestirilmis = maskeyi_uygula(goruntu, maske_iyilestirilmis)
        
        # Ham maskeyi de uyguluyorum (karşılaştırma için)
        nesne_ham = maskeyi_uygula(goruntu, maske)
        
        # Tüm aşamaları gösteriyorum (2x3 grid)
        ust_satir = np.hstack([
            cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(maske_acma, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(maske_kapama, cv2.COLOR_GRAY2BGR)
        ])
        
        # Alt satır: Önce-sonra karşılaştırması
        alt_satir = np.hstack([
            cv2.cvtColor(maske_iyilestirilmis, cv2.COLOR_GRAY2BGR),
            nesne_ham,  # İyileştirme öncesi
            nesne_iyilestirilmis  # İyileştirme sonrası
        ])
        
        birlesik_goruntu = np.vstack([ust_satir, alt_satir])
        
        # Ekrana sığması için boyutlandırıyorum
        yukseklik, genislik = birlesik_goruntu.shape[:2]
        yeni_genislik = int(genislik * 0.5)
        yeni_yukseklik = int(yukseklik * 0.5)
        birlesik_goruntu_kucuk = cv2.resize(birlesik_goruntu, (yeni_genislik, yeni_yukseklik))
        
        cv2.imshow(f'{renk.capitalize()} - Morfolojik Asamalar', birlesik_goruntu_kucuk)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



def ana():
    """Ana fonksiyon - tüm görevleri çalıştır"""
    
    goruntu_yolu = "araba.jpeg"
    goruntu = cv2.imread(goruntu_yolu)
    
    if goruntu is None:
        return
    
    # Görevleri sırayla çalıştırıyorum
    gorev1_renk_uzayi_donusumleri(goruntu)
    maskeler = gorev2_hsv_segmentasyon(goruntu)
    gorev3_delta_e_gruplama(goruntu)
    gorev4_morfolojik_iyilestirme(goruntu, maskeler)


if __name__ == "__main__":
    ana()