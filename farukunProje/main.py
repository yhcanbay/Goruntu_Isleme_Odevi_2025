"""
Renkli Görüntü İşleme ve Renk Uzayları - Ana Program (Otomatik Mod)
===================================================================
cv2.imshow() olmadan çalışır - sadece dosyalara kaydeder
"""

import cv2
import numpy as np
from segmentasyon import renk_segmentasyonu, maskeyi_uygula
from morfoloji import tam_iyilestirme, acma, kapama
from renk_gruplama import dominant_renkler_bul, delta_e_cie76, renkleri_grupla


def gorev1_renk_uzayi_donusumleri(goruntu, sonuclar_klasoru):
    """GÖREV 1: RGB → HSV ve LAB dönüşümleri"""
    print("\n" + "="*70)
    print("GÖREV 1: RENK UZAYI DÖNÜŞÜMLERİ (RGB → HSV, LAB)")
    print("="*70)
    
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    print(f"✓ RGB → HSV dönüşümü tamamlandı")
    print(f"  - H aralığı: {h.min()} - {h.max()}, S: {s.min()} - {s.max()}, V: {v.min()} - {v.max()}")
    
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    print(f"✓ RGB → LAB dönüşümü tamamlandı")
    print(f"  - L aralığı: {l.min()} - {l.max()}, a: {a.min()} - {a.max()}, b: {b.min()} - {b.max()}")
    
    cv2.imwrite(f"{sonuclar_klasoru}/1_orijinal.png", goruntu)
    cv2.imwrite(f"{sonuclar_klasoru}/1_hsv_h.png", h)
    cv2.imwrite(f"{sonuclar_klasoru}/1_hsv_s.png", s)
    cv2.imwrite(f"{sonuclar_klasoru}/1_hsv_v.png", v)
    cv2.imwrite(f"{sonuclar_klasoru}/1_lab_l.png", l)
    cv2.imwrite(f"{sonuclar_klasoru}/1_lab_a.png", a)
    cv2.imwrite(f"{sonuclar_klasoru}/1_lab_b.png", b)
    print(f"✓ Kanal görüntüleri kaydedildi")
    
    return {'hsv': hsv, 'lab': lab}


def gorev2_hsv_segmentasyon(goruntu, sonuclar_klasoru):
    """GÖREV 2: HSV ile çoklu renk segmentasyonu"""
    print("\n" + "="*70)
    print("GÖREV 2: HSV UZAYINDA RENK SEGMENTASYONU")
    print("="*70)
    
    renkler = ['kirmizi', 'yesil', 'mavi']
    maskeler = {}
    toplam_piksel = goruntu.shape[0] * goruntu.shape[1]
    
    for renk in renkler:
        maske = renk_segmentasyonu(goruntu, renk)
        segmente = maskeyi_uygula(goruntu, maske)
        maskeler[renk] = maske
        
        beyaz = np.sum(maske > 0)
        oran = (beyaz / toplam_piksel) * 100
        print(f"✓ {renk.upper()}: {beyaz:,} piksel ({oran:.2f}%)")
        
        cv2.imwrite(f"{sonuclar_klasoru}/2_{renk}_maske.png", maske)
        cv2.imwrite(f"{sonuclar_klasoru}/2_{renk}_segmente.png", segmente)
    
    return maskeler


def gorev3_delta_e_gruplama(goruntu, sonuclar_klasoru):
    """GÖREV 3: LAB Delta E ile renk gruplama"""
    print("\n" + "="*70)
    print("GÖREV 3: LAB DELTA E İLE RENK GRUPLAMA")
    print("="*70)
    
    k = 8
    dominant = dominant_renkler_bul(goruntu, k=k)
    print(f"✓ K-Means ile {k} baskın renk bulundu:")
    for i, renk in enumerate(dominant):
        print(f"  {i+1}. L={renk[0]:.1f}, a={renk[1]:.1f}, b={renk[2]:.1f}")
    
    # Renk paleti oluştur
    palet = np.zeros((100, k * 100, 3), dtype=np.uint8)
    for i, renk_lab in enumerate(dominant):
        lab_pixel = np.uint8([[renk_lab]])
        bgr = cv2.cvtColor(lab_pixel.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
        palet[:, i*100:(i+1)*100] = bgr
    
    cv2.imwrite(f"{sonuclar_klasoru}/3_renk_paleti.png", palet)
    print(f"✓ Renk paleti kaydedildi")
    
    return dominant


def gorev4_morfolojik_iyilestirme(goruntu, maskeler, sonuclar_klasoru):
    """GÖREV 4: Morfolojik işlemlerle iyileştirme"""
    print("\n" + "="*70)
    print("GÖREV 4: MORFOLOJİK İYİLEŞTİRME")
    print("="*70)
    
    for renk, maske in maskeler.items():
        ham = np.sum(maske > 0)
        maske_acma = acma(maske, 3)
        maske_kapama = kapama(maske_acma, 7)
        maske_iyi = tam_iyilestirme(maske, 3, 7)
        iyi = np.sum(maske_iyi > 0)
        
        print(f"✓ {renk.upper()}: {ham:,} → {iyi:,} piksel (fark: {iyi-ham:+,})")
        
        nesne_iyi = maskeyi_uygula(goruntu, maske_iyi)
        cv2.imwrite(f"{sonuclar_klasoru}/4_{renk}_iyilestirilmis.png", nesne_iyi)
        
        # Karşılaştırma görüntüsü
        karsilastirma = np.hstack([
            cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(maske_iyi, cv2.COLOR_GRAY2BGR),
            nesne_iyi
        ])
        cv2.imwrite(f"{sonuclar_klasoru}/4_{renk}_karsilastirma.png", karsilastirma)


def ana():
    """Ana program"""
    print("\n" + "="*70)
    print("      RENKLİ GÖRÜNTÜ İŞLEME VE RENK UZAYLARI - PROJE 4")
    print("="*70)
    
    goruntu = cv2.imread("araba.jpeg")
    if goruntu is None:
        print("❌ HATA: 'araba.jpeg' bulunamadı!")
        return
    
    print(f"\n✓ Görüntü yüklendi: {goruntu.shape[1]}x{goruntu.shape[0]} piksel")
    
    sonuclar = "farukunProje/sonuclar"
    
    gorev1_renk_uzayi_donusumleri(goruntu, sonuclar)
    maskeler = gorev2_hsv_segmentasyon(goruntu, sonuclar)
    gorev3_delta_e_gruplama(goruntu, sonuclar)
    gorev4_morfolojik_iyilestirme(goruntu, maskeler, sonuclar)
    
    print("\n" + "="*70)
    print("✅ PROJE TAMAMLANDI - Sonuçlar 'sonuclar/' klasöründe")
    print("="*70 + "\n")


if __name__ == "__main__":
    ana()
