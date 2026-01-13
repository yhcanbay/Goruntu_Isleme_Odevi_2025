"""
Renkli Görüntü İşleme ve Renk Uzayları - Ana Program
====================================================
Proje 4: RGB, HSV, LAB renk uzayları ve renk tabanlı segmentasyon

Görevler:
1. RGB → HSV ve LAB dönüşümleri
2. HSV ile kırmızı araba segmentasyonu
3. LAB Delta E ile renk gruplama
4. Morfolojik işlemlerle iyileştirme

Kitap Bölüm: 6 (Color Image Processing)
Kütüphaneler: OpenCV, NumPy
"""

import cv2
import numpy as np
from segmentasyon import renk_segmentasyonu, maskeyi_uygula
from morfoloji import tam_iyilestirme, acma, kapama
from renk_gruplama import dominant_renkler_bul, delta_e_cie76, renkleri_grupla


def gorev1_renk_uzayi_donusumleri(goruntu: np.ndarray, sonuclar_klasoru: str):
    """
    GÖREV 1: RGB → HSV ve LAB dönüşümleri
    """
    print("\n" + "="*70)
    print("GÖREV 1: RENK UZAYI DÖNÜŞÜMLERİ (RGB → HSV, LAB)")
    print("="*70)
    
    # RGB → HSV dönüşümü
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    print(f"✓ RGB → HSV dönüşümü tamamlandı")
    print(f"  - H (Hue) aralığı: {h.min()} - {h.max()}")
    print(f"  - S (Saturation) aralığı: {s.min()} - {s.max()}")
    print(f"  - V (Value) aralığı: {v.min()} - {v.max()}")
    
    # RGB → LAB dönüşümü
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    print(f"✓ RGB → LAB dönüşümü tamamlandı")
    print(f"  - L (Lightness) aralığı: {l.min()} - {l.max()}")
    print(f"  - a (Green-Red) aralığı: {a.min()} - {a.max()}")
    print(f"  - b (Blue-Yellow) aralığı: {b.min()} - {b.max()}")
    
    # Sonuçları kaydet (sadece ekranda gösterilenleri)
    cv2.imwrite(sonuclar_klasoru + "/1_orijinal.png", goruntu)
    cv2.imwrite(sonuclar_klasoru + "/1_hsv.png", hsv)
    cv2.imwrite(sonuclar_klasoru + "/1_lab.png", lab)
    
    print(f"✓ Sonuçlar '{sonuclar_klasoru}' klasörüne kaydedildi")
    
    # Görüntüleri ekranda göster
    cv2.imshow('Orijinal Goruntu (RGB)', goruntu)
    cv2.imshow('HSV Goruntu', hsv)
    cv2.imshow('LAB Goruntu', lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gorev2_hsv_segmentasyon(goruntu: np.ndarray, sonuclar_klasoru: str):
    """
    GÖREV 2: HSV ile çoklu renk segmentasyonu (Kırmızı, Yeşil, Mavi)
    """
    print("\n" + "="*70)
    print("GÖREV 2: HSV UZAYINDA ÇOKLU RENK SEGMENTASYONU")
    print("="*70)
    
    # Segmente edilecek renkler
    renkler = ['kirmizi', 'yesil', 'mavi']
    maskeler = {}
    segmente_nesneler = {}
    
    toplam_piksel = goruntu.shape[0] * goruntu.shape[1]
    
    # Her renk için segmentasyon yap
    for renk in renkler:
        print(f"\n--- {renk.upper()} Renk Segmentasyonu ---")
        
        # Renk segmentasyonu
        maske = renk_segmentasyonu(goruntu, renk)
        
        beyaz_piksel = np.sum(maske > 0)
        oran = (beyaz_piksel / toplam_piksel) * 100
        
        print(f"✓ HSV eşikleme tamamlandı")
        print(f"  - Segmente edilen piksel sayısı: {beyaz_piksel:,}")
        print(f"  - Segmentasyon oranı: {oran:.2f}%")
        
        # Maskeyi uygula
        segmente_nesne = maskeyi_uygula(goruntu, maske)
        
        # Sonuçları sakla
        maskeler[renk] = maske
        segmente_nesneler[renk] = segmente_nesne
    
    print(f"\n✓ Tüm segmentasyon sonuçları kaydedildi")
    
    # Her renk için görüntüleri yan yana birleştir ve göster
    for renk in renkler:
        # Maske ve segmente görüntüyü yan yana birleştir
        maske_bgr = cv2.cvtColor(maskeler[renk], cv2.COLOR_GRAY2BGR)
        birlesik = np.hstack([maske_bgr, segmente_nesneler[renk]])
        
        # Görüntüyü ekrana sığacak şekilde küçült (ölçek: %50)
        yukseklik, genislik = birlesik.shape[:2]
        yeni_genislik = int(genislik * 0.5)
        yeni_yukseklik = int(yukseklik * 0.5)
        birlesik_kucuk = cv2.resize(birlesik, (yeni_genislik, yeni_yukseklik))
        
        # Birleşik görüntüyü kaydet ve göster
        cv2.imwrite(sonuclar_klasoru + f"/2_{renk}_birlesik.png", birlesik)
        cv2.imshow(f'{renk.capitalize()} - Maske | Segmente', birlesik_kucuk)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return maskeler



def gorev3_delta_e_gruplama(goruntu: np.ndarray, sonuclar_klasoru: str):
    """
    GÖREV 3: LAB Delta E ile renk gruplama
    """
    print("\n" + "="*70)
    print("GÖREV 3: LAB UZAYINDA DELTA E İLE RENK GRUPLAMA")
    print("="*70)
    
    # Baskın renkleri bul
    k = 8
    dominant = dominant_renkler_bul(goruntu, k=k)
    
    print(f"✓ K-Means ile {k} baskın renk bulundu (LAB formatında):")
    for i, renk in enumerate(dominant):
        print(f"  {i+1}. L={renk[0]:6.2f}, a={renk[1]:6.2f}, b={renk[2]:6.2f}")
    
    # Delta E mesafelerini hesapla
    print(f"\n✓ Renkler arası Delta E (CIE76) mesafeleri:")
    for i in range(len(dominant)):
        for j in range(i+1, len(dominant)):
            de = delta_e_cie76(dominant[i], dominant[j])
            if de < 30:  # Benzer renkler
                print(f"  Renk {i+1} ↔ Renk {j+1}: ΔE = {de:6.2f} ★ (Benzer)")
            else:
                print(f"  Renk {i+1} ↔ Renk {j+1}: ΔE = {de:6.2f}")
    
    # Renkleri grupla (Delta E < 20 olanlar aynı grupta)
    renkler_liste = [tuple(r) for r in dominant]
    gruplar = renkleri_grupla(renkler_liste, esik=20.0)
    
    print(f"\n✓ Renk gruplama sonucu (ΔE < 20):")
    print(f"  - Toplam grup sayısı: {len(gruplar)}")
    for grup_id, renkler in gruplar.items():
        print(f"  - Grup {grup_id + 1}: {len(renkler)} renk")
    
    # Baskın renkleri görselleştir
    renk_paleti = np.zeros((100, k * 100, 3), dtype=np.uint8)
    for i, renk_lab in enumerate(dominant):
        # LAB → BGR dönüşümü
        lab_pixel = np.uint8([[renk_lab]])
        bgr_pixel = cv2.cvtColor(lab_pixel.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)
        renk_bgr = bgr_pixel[0, 0]
        renk_paleti[:, i*100:(i+1)*100] = renk_bgr
    
    cv2.imwrite(sonuclar_klasoru + "/3_renk_paleti.png", renk_paleti)
    print(f"✓ Renk paleti kaydedildi")
    
    # Renk paletini ekranda göster
    cv2.imshow('Baskin Renk Paleti (LAB)', renk_paleti)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



def gorev4_morfolojik_iyilestirme(goruntu: np.ndarray, maskeler: dict, sonuclar_klasoru: str):
    """
    GÖREV 4: Morfolojik işlemlerle çoklu renk segmentasyon iyileştirme
    """
    print("\n" + "="*70)
    print("GÖREV 4: MORFOLOJİK İŞLEMLERLE İYİLEŞTİRME (KIRMIZI, YEŞİL, MAVİ)")
    print("="*70)
    
    maskeler_iyilestirilmis = {}
    nesneler_iyilestirilmis = {}
    
    # Her renk için morfolojik işlemler
    for renk, maske in maskeler.items():
        print(f"\n--- {renk.upper()} Renk İyileştirme ---")
        
        # Ham maske istatistikleri
        ham_beyaz = np.sum(maske > 0)
        print(f"Ham maske - Beyaz piksel: {ham_beyaz:,}")
        
        # 1. Açma işlemi (gürültü temizleme)
        maske_acma = acma(maske, kernel_boyut=3)
        acma_beyaz = np.sum(maske_acma > 0)
        print(f"✓ Açma (Opening) işlemi tamamlandı")
        print(f"  - Beyaz piksel: {acma_beyaz:,} (Fark: {ham_beyaz - acma_beyaz:,})")
        
        # 2. Kapama işlemi (delik doldurma)
        maske_kapama = kapama(maske_acma, kernel_boyut=7)
        kapama_beyaz = np.sum(maske_kapama > 0)
        print(f"✓ Kapama (Closing) işlemi tamamlandı")
        print(f"  - Beyaz piksel: {kapama_beyaz:,} (Fark: +{kapama_beyaz - acma_beyaz:,})")
        
        # 3. Tam iyileştirme (açma + kapama)
        maske_iyilestirilmis = tam_iyilestirme(maske, acma_boyut=3, kapama_boyut=7)
        iyi_beyaz = np.sum(maske_iyilestirilmis > 0)
        print(f"✓ Tam iyileştirme tamamlandı")
        print(f"  - Beyaz piksel: {iyi_beyaz:,}")
        print(f"  - Net değişim: {iyi_beyaz - ham_beyaz:,} piksel")
        
        # İyileştirilmiş maskeyi uygula
        nesne_iyilestirilmis = maskeyi_uygula(goruntu, maske_iyilestirilmis)
        
        # Sonuçları sakla
        maskeler_iyilestirilmis[renk] = maske_iyilestirilmis
        nesneler_iyilestirilmis[renk] = nesne_iyilestirilmis
        
        # Tüm aşamaları tek görüntüde birleştir (2 satır x 3 sütun)
        # Üst satır: Ham, Açma, Kapama maskeleri
        ust_satir = np.hstack([
            cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(maske_acma, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(maske_kapama, cv2.COLOR_GRAY2BGR)
        ])
        
        # Alt satır: İyileştirilmiş maske, İyileştirilmiş nesne, Boş (veya tekrar iyileştirilmiş nesne)
        alt_satir = np.hstack([
            cv2.cvtColor(maske_iyilestirilmis, cv2.COLOR_GRAY2BGR),
            nesne_iyilestirilmis,
            nesne_iyilestirilmis  # Son sütunu da nesne ile doldur
        ])
        
        # İki satırı birleştir
        birlesik_goruntu = np.vstack([ust_satir, alt_satir])
        
        # Görüntüyü ekrana sığacak şekilde küçült (ölçek: %50)
        yukseklik, genislik = birlesik_goruntu.shape[:2]
        yeni_genislik = int(genislik * 0.5)
        yeni_yukseklik = int(yukseklik * 0.5)
        birlesik_goruntu_kucuk = cv2.resize(birlesik_goruntu, (yeni_genislik, yeni_yukseklik))
        
        # Birleşik görüntüyü kaydet ve göster
        cv2.imwrite(sonuclar_klasoru + f"/4_{renk}_tum_asamalar.png", birlesik_goruntu)
        cv2.imshow(f'{renk.capitalize()} - Morfolojik Asamalar', birlesik_goruntu_kucuk)
    
    print(f"\n✓ Tüm morfolojik işlem sonuçları kaydedildi")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



def ana():
    """Ana program fonksiyonu"""
    print("\n" + "="*70)
    print(" " * 10 + "RENKLİ GÖRÜNTÜ İŞLEME VE RENK UZAYLARI")
    print(" " * 20 + "Proje 4 - Ana Program")
    print("="*70)
    
    # Görüntüyü yükle
    goruntu_yolu = "araba.jpeg"
    goruntu = cv2.imread(goruntu_yolu)
    
    if goruntu is None:
        print(f"❌ HATA: '{goruntu_yolu}' bulunamadı!")
        return
    
    print(f"\n✓ Görüntü yüklendi: {goruntu_yolu}")
    print(f"  - Boyut: {goruntu.shape[1]} x {goruntu.shape[0]} piksel")
    print(f"  - Renk kanalları: {goruntu.shape[2]}")
    
    # Sonuçlar klasörü
    sonuclar_klasoru = "sonuclar"
    print(f"✓ Sonuçlar klasörü: {sonuclar_klasoru}/")
    
    # GÖREV 1: Renk uzayı dönüşümleri
    gorev1_renk_uzayi_donusumleri(goruntu, sonuclar_klasoru)
    
    # GÖREV 2: HSV segmentasyon
    maskeler = gorev2_hsv_segmentasyon(goruntu, sonuclar_klasoru)
    
    # GÖREV 3: Delta E renk gruplama
    gorev3_delta_e_gruplama(goruntu, sonuclar_klasoru)
    
    # GÖREV 4: Morfolojik iyileştirme (tüm renkler için)
    gorev4_morfolojik_iyilestirme(goruntu, maskeler, sonuclar_klasoru)
    
    # Özet rapor
    print("\n" + "="*70)
    print(" " * 25 + "PROJE TAMAMLANDI")
    print("="*70)
    print(f"\n✓ Tüm görevler başarıyla tamamlandı!")
    print(f"✓ Sonuçlar '{sonuclar_klasoru}/' klasörüne kaydedildi")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    ana()