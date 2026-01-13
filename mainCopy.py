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

# Gerekli kütüphaneleri içe aktar
import cv2  # OpenCV - Görüntü işleme kütüphanesi
import numpy as np  # NumPy - Sayısal işlemler ve matris hesaplamaları için
from segmentasyon import renk_segmentasyonu, maskeyi_uygula  # Renk tabanlı segmentasyon fonksiyonları
from morfoloji import tam_iyilestirme, acma, kapama  # Morfolojik işlem fonksiyonları
from renk_gruplama import dominant_renkler_bul, delta_e_cie76, renkleri_grupla  # Renk analizi fonksiyonları

def gorev1_renk_uzayi_donusumleri(goruntu: np.ndarray, sonuclar_klasoru: str):
    """
    GÖREV 1: RGB → HSV ve LAB dönüşümleri
    
    Amaç: Farklı renk uzaylarında görüntüyü temsil etmek
    - HSV: Hue (Renk Tonu), Saturation (Doygunluk), Value (Parlaklık)
    - LAB: L (Parlaklık), a (Yeşil-Kırmızı), b (Mavi-Sarı)
    """
    print("\n" + "="*70)
    print("GÖREV 1: RENK UZAYI DÖNÜŞÜMLERİ (RGB → HSV, LAB)")
    print("="*70)
    
    # RGB → HSV dönüşümü
    # HSV renk uzayı renk tabanlı segmentasyon için daha uygundur
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)  # OpenCV BGR formatından HSV'ye çevirir
    h, s, v = cv2.split(hsv)  # 3 kanala ayır (Hue, Saturation, Value)
    
    print(f"✓ RGB → HSV dönüşümü tamamlandı")
    print(f"  - H (Hue) aralığı: {h.min()} - {h.max()}")
      # Hue: 0-179 arası (OpenCV'de)
    print(f"  - S (Saturation) aralığı: {s.min()} - {s.max()}")  # Saturation: 0-255
    print(f"  - V (Value) aralığı: {v.min()} - {v.max()}")  # Value: 0-255
    
    # RGB → LAB dönüşümü
    # LAB renk uzayı insan algısına daha yakın, renk mesafesi hesaplamak için ideal
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)  # BGR formatından LAB'a çevir
    l, a, b = cv2.split(lab)  # 3 kanala ayır (L, a, b)
    
    print(f"✓ RGB → LAB dönüşümü tamamlandı")
    print(f"  - L (Lightness) aralığı: {l.min()} - {l.max()}")  # L: 0-255 (Parlaklık)
    print(f"  - a (Green-Red) aralığı: {a.min()} - {a.max()}")  # a: Yeşil(-) ↔ Kırmızı(+)
    print(f"  - b (Blue-Yellow) aralığı: {b.min()} - {b.max()}")  # b: Mavi(-) ↔ Sarı(+)
    
    # Sonuçları dosyaya kaydet
    cv2.imwrite(sonuclar_klasoru + "/1_orijinal.png", goruntu)  # Orijinal görüntü
    cv2.imwrite(sonuclar_klasoru + "/1_hsv.png", hsv)  # HSV formatında görüntü
    cv2.imwrite(sonuclar_klasoru + "/1_lab.png", lab)  # LAB formatında görüntü
    
    print(f"✓ Sonuçlar '{sonuclar_klasoru}' klasörüne kaydedildi")
    
    # Görüntüleri ekranda göster
    cv2.imshow('Orijinal Goruntu (RGB)', goruntu)
    cv2.imshow('HSV Goruntu', hsv)
    cv2.imshow('LAB Goruntu', lab)
    cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar bekle
    cv2.destroyAllWindows()  # Tüm pencereleri kapat


def gorev2_hsv_segmentasyon(goruntu: np.ndarray, sonuclar_klasoru: str):
    """
    GÖREV 2: HSV ile çoklu renk segmentasyonu (Kırmızı, Yeşil, Mavi)
    
    Amaç: HSV renk uzayında belirli renkleri ayırmak (segmentasyon)
    HSV kullanma nedeni: Aydınlatma değişimlerine daha dayanıklı
    """
    print("\n" + "="*70)
    print("GÖREV 2: HSV UZAYINDA ÇOKLU RENK SEGMENTASYONU")
    print("="*70)
    
    # Segmente edilecek renkler
    renkler = ['kirmizi', 'yesil', 'mavi']  # Bu renkler ayrı ayrı tespit edilecek
    maskeler = {}  # Her renk için maske saklamak için sözlük
    segmente_nesneler = {}  # Her renk için segmente edilmiş görüntüleri sakla
    
    toplam_piksel = goruntu.shape[0] * goruntu.shape[1]  # Toplam piksel sayısı (yükseklik x genişlik)
    
    # Her renk için segmentasyon yap
    for renk in renkler:
        print(f"\n--- {renk.upper()} Renk Segmentasyonu ---")
        
        # Renk segmentasyonu (HSV eşikleme ile belirli renkteki pikselleri bul)
        maske = renk_segmentasyonu(goruntu, renk)  # İkili maske döner (0: siyah, 255: beyaz)
        
        beyaz_piksel = np.sum(maske > 0)  # Beyaz pikselleri say (segmente edilen bölge)
        oran = (beyaz_piksel / toplam_piksel) * 100  # Yüzde olarak hesapla
        
        print(f"✓ HSV eşikleme tamamlandı")
        print(f"  - Segmente edilen piksel sayısı: {beyaz_piksel:,}")
        print(f"  - Segmentasyon oranı: {oran:.2f}%")
        
        # Maskeyi orijinal görüntüye uygula (sadece segmente edilen bölgeyi göster)
        segmente_nesne = maskeyi_uygula(goruntu, maske)  # Maske beyaz olan yerleri gösterir
        
        # Sonuçları sakla
        maskeler[renk] = maske
        segmente_nesneler[renk] = segmente_nesne
    
    print(f"\n✓ Tüm segmentasyon sonuçları kaydedildi")
    
    # Her renk için görüntüleri yan yana birleştir ve göster
    for renk in renkler:
        # Maske ve segmente görüntüyü yan yana birleştir
        maske_bgr = cv2.cvtColor(maskeler[renk], cv2.COLOR_GRAY2BGR)  # Gri maskeyi BGR'ye çevir
        birlesik = np.hstack([maske_bgr, segmente_nesneler[renk]])  # Yatay olarak birleştir
        
        # Görüntüyü ekrana sığacak şekilde küçült (ölçek: %50)
        yukseklik, genislik = birlesik.shape[:2]  # Mevcut boyutları al
        yeni_genislik = int(genislik * 0.5)  # %50 küçült
        yeni_yukseklik = int(yukseklik * 0.5)
        birlesik_kucuk = cv2.resize(birlesik, (yeni_genislik, yeni_yukseklik))  # Yeniden boyutlandır
        
        # Birleşik görüntüyü kaydet ve göster
        cv2.imwrite(sonuclar_klasoru + f"/2_{renk}_birlesik.png", birlesik)
        cv2.imshow(f'{renk.capitalize()} - Maske | Segmente', birlesik_kucuk)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return maskeler  # Görev 4'te kullanmak üzere maskeleri döndür



def gorev3_delta_e_gruplama(goruntu: np.ndarray, sonuclar_klasoru: str):
    """
    GÖREV 3: LAB Delta E ile renk gruplama
    
    Amaç: Görüntüdeki baskın renkleri bul ve benzer renkleri grupla
    - K-Means: Pikselleri K gruba ayıran kümeleme algoritması
    - Delta E: LAB uzayında iki renk arasındaki algısal mesafe
    """
    print("\n" + "="*70)
    print("GÖREV 3: LAB UZAYINDA DELTA E İLE RENK GRUPLAMA")
    print("="*70)
    
    # Baskın renkleri bul (K-Means kümeleme ile)
    k = 8  # 8 ana renk grubu bulacağız
    dominant = dominant_renkler_bul(goruntu, k=k)  # K-Means algoritması ile baskın renkleri al
    
    print(f"✓ K-Means ile {k} baskın renk bulundu (LAB formatında):")
    for i, renk in enumerate(dominant):
        print(f"  {i+1}. L={renk[0]:6.2f}, a={renk[1]:6.2f}, b={renk[2]:6.2f}")
    
    # Delta E mesafelerini hesapla (renkler arasındaki farkı ölç)
    # Delta E < 1: Göz farkı göremez, Delta E < 20: Benzer renkler, Delta E > 50: Farklı renkler
    print(f"\n✓ Renkler arası Delta E (CIE76) mesafeleri:")
    for i in range(len(dominant)):
        for j in range(i+1, len(dominant)):
            de = delta_e_cie76(dominant[i], dominant[j])  # İki renk arasındaki Delta E'yi hesapla
            if de < 30:  # Benzer renkler (küçük Delta E)
                print(f"  Renk {i+1} ↔ Renk {j+1}: ΔE = {de:6.2f} ★ (Benzer)")
            else:
                print(f"  Renk {i+1} ↔ Renk {j+1}: ΔE = {de:6.2f}")
    
    # Renkleri grupla (Delta E < 20 olanlar aynı grupta)
    # Benzer renkleri tek bir renk olarak kabul eder
    renkler_liste = [tuple(r) for r in dominant]  # Liste olarak hazırla
    gruplar = renkleri_grupla(renkler_liste, esik=20.0)  # Eşik değerinden küçük olanları grupla
    
    print(f"\n✓ Renk gruplama sonucu (ΔE < 20):")
    print(f"  - Toplam grup sayısı: {len(gruplar)}")
    for grup_id, renkler in gruplar.items():
        print(f"  - Grup {grup_id + 1}: {len(renkler)} renk")
    
    # Baskın renkleri görselleştir (renk paleti oluştur)
    renk_paleti = np.zeros((100, k * 100, 3), dtype=np.uint8)  # Boş görüntü (100 piksel yükseklik)
    for i, renk_lab in enumerate(dominant):
        # LAB → BGR dönüşümü (ekranda göstermek için BGR formatına çevir)
        lab_pixel = np.uint8([[renk_lab]])  # Tek piksel LAB rengi
        bgr_pixel = cv2.cvtColor(lab_pixel.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)  # LAB'dan BGR'ye
        renk_bgr = bgr_pixel[0, 0]
        renk_paleti[:, i*100:(i+1)*100] = renk_bgr  # Her renk için 100 piksel genişliğinde alan
    
    cv2.imwrite(sonuclar_klasoru + "/3_renk_paleti.png", renk_paleti)
    print(f"✓ Renk paleti kaydedildi")
    
    # Renk paletini ekranda göster
    cv2.imshow('Baskin Renk Paleti (LAB)', renk_paleti)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



def gorev4_morfolojik_iyilestirme(goruntu: np.ndarray, maskeler: dict, sonuclar_klasoru: str):
    """
    GÖREV 4: Morfolojik işlemlerle çoklu renk segmentasyon iyileştirme
    
    Amaç: Segmentasyon maskelerindeki gürültüyü temizle ve delikleri doldur
    - Açma (Opening): Küçük gürültüleri/noktaları kaldırır (Erozyon → Dilatasyon)
    - Kapama (Closing): Nesnelerdeki küçük delikleri doldurur (Dilatasyon → Erozyon)
    """
    print("\n" + "="*70)
    print("GÖREV 4: MORFOLOJİK İŞLEMLERLE İYİLEŞTİRME (KIRMIZI, YEŞİL, MAVİ)")
    print("="*70)
    
    maskeler_iyilestirilmis = {}  # İyileştirilmiş maskeleri sakla
    nesneler_iyilestirilmis = {}  # İyileştirilmiş nesneleri sakla
    
    # Her renk için morfolojik işlemler
    for renk, maske in maskeler.items():
        print(f"\n--- {renk.upper()} Renk İyileştirme ---")
        
        # Ham maske istatistikleri
        ham_beyaz = np.sum(maske > 0)  # Ham maskedeki beyaz piksel sayısı
        print(f"Ham maske - Beyaz piksel: {ham_beyaz:,}")
        
        # 1. Açma işlemi (gürültü temizleme)
        # Küçük beyaz noktaları (gürültü) kaldırır
        maske_acma = acma(maske, kernel_boyut=3)  # 3x3 kernel ile açma
        acma_beyaz = np.sum(maske_acma > 0)
        print(f"✓ Açma (Opening) işlemi tamamlandı")
        print(f"  - Beyaz piksel: {acma_beyaz:,} (Fark: {ham_beyaz - acma_beyaz:,})")
        
        # 2. Kapama işlemi (delik doldurma)
        # Nesnelerin içindeki küçük siyah delikleri doldurur
        maske_kapama = kapama(maske_acma, kernel_boyut=7)  # 7x7 kernel ile kapama
        kapama_beyaz = np.sum(maske_kapama > 0)
        print(f"✓ Kapama (Closing) işlemi tamamlandı")
        print(f"  - Beyaz piksel: {kapama_beyaz:,} (Fark: +{kapama_beyaz - acma_beyaz:,})")
        
        # 3. Tam iyileştirme (açma + kapama birlikte)
        # Önce gürültüyü temizle, sonra delikleri doldur
        maske_iyilestirilmis = tam_iyilestirme(maske, acma_boyut=3, kapama_boyut=7)
        iyi_beyaz = np.sum(maske_iyilestirilmis > 0)
        print(f"✓ Tam iyileştirme tamamlandı")
        print(f"  - Beyaz piksel: {iyi_beyaz:,}")
        print(f"  - Net değişim: {iyi_beyaz - ham_beyaz:,} piksel")
        
        # İyileştirilmiş maskeyi orijinal görüntüye uygula
        nesne_iyilestirilmis = maskeyi_uygula(goruntu, maske_iyilestirilmis)
        
        # Sonuçları sakla
        maskeler_iyilestirilmis[renk] = maske_iyilestirilmis
        nesneler_iyilestirilmis[renk] = nesne_iyilestirilmis
        
        # Tüm aşamaları tek görüntüde birleştir (2 satır x 3 sütun)
        # Üst satır: Ham, Açma, Kapama maskeleri
        ust_satir = np.hstack([
            cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR),  # Ham maske
            cv2.cvtColor(maske_acma, cv2.COLOR_GRAY2BGR),  # Açma sonrası
            cv2.cvtColor(maske_kapama, cv2.COLOR_GRAY2BGR)  # Kapama sonrası
        ])
        
        # Alt satır: İyileştirilmiş maske, İyileştirilmiş nesne, İyileştirilmiş nesne (tekrar)
        alt_satir = np.hstack([
            cv2.cvtColor(maske_iyilestirilmis, cv2.COLOR_GRAY2BGR),  # İyileştirilmiş maske
            nesne_iyilestirilmis,  # İyileştirilmiş nesne
            nesne_iyilestirilmis  # Son sütunu da nesne ile doldur
        ])
        
        # İki satırı birleştir (üst + alt)
        birlesik_goruntu = np.vstack([ust_satir, alt_satir])  # Dikey olarak birleştir
        
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
    """Ana program fonksiyonu - Tüm görevleri sırayla çalıştırır"""
    print("\n" + "="*70)
    print(" " * 10 + "RENKLİ GÖRÜNTÜ İŞLEME VE RENK UZAYLARI")
    print(" " * 20 + "Proje 4 - Ana Program")
    print("="*70)
    
    # Görüntüyü dosyadan yükle
    goruntu_yolu = "araba.jpeg"
    goruntu = cv2.imread(goruntu_yolu)  # OpenCV ile görüntüyü oku (BGR formatında)
    
    # Görüntü yüklenemezse hata mesajı ver
    if goruntu is None:
        print(f"❌ HATA: '{goruntu_yolu}' bulunamadı!")
        return
    
    # Görüntü bilgilerini ekrana yazdır
    print(f"\n✓ Görüntü yüklendi: {goruntu_yolu}")
    print(f"  - Boyut: {goruntu.shape[1]} x {goruntu.shape[0]} piksel")  # Genişlik x Yükseklik
    print(f"  - Renk kanalları: {goruntu.shape[2]}")  # 3 kanal: BGR
    
    # Sonuçların kaydedileceği klasör
    sonuclar_klasoru = "sonuclar"
    print(f"✓ Sonuçlar klasörü: {sonuclar_klasoru}/")
    
    # GÖREV 1: Renk uzayı dönüşümleri (RGB → HSV, LAB)
    gorev1_renk_uzayi_donusumleri(goruntu, sonuclar_klasoru)
    
    # GÖREV 2: HSV segmentasyon (Kırmızı, Yeşil, Mavi renkleri ayır)
    maskeler = gorev2_hsv_segmentasyon(goruntu, sonuclar_klasoru)
    
    # GÖREV 3: Delta E renk gruplama (Baskın renkleri bul ve grupla)
    gorev3_delta_e_gruplama(goruntu, sonuclar_klasoru)
    
    # GÖREV 4: Morfolojik iyileştirme (Maskeleri temizle ve düzelt)
    gorev4_morfolojik_iyilestirme(goruntu, maskeler, sonuclar_klasoru)
    
    # Özet rapor
    print("\n" + "="*70)
    print(" " * 25 + "PROJE TAMAMLANDI")
    print("="*70)
    print(f"\n✓ Tüm görevler başarıyla tamamlandı!")
    print(f"✓ Sonuçlar '{sonuclar_klasoru}/' klasörüne kaydedildi")
    print("\n" + "="*70 + "\n")


# Program buradan başlar
if __name__ == "__main__":
    ana()  # Ana fonksiyonu çalıştır

