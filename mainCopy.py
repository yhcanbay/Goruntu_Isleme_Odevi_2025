"""
Görüntü İşleme Ödevi - Renk Uzayları ve Segmentasyon
Proje 4: RGB, HSV, LAB renk uzayları ile renk tabanlı segmentasyon

Görevler:
1. RGB'den HSV ve LAB'a dönüşüm
2. HSV ile çoklu renk segmentasyonu
3. LAB Delta E ile renk gruplama
4. Morfolojik işlemlerle iyileştirme
"""

# Gerekli kütüphaneleri içe aktarıyorum
import cv2  # OpenCV - Görüntü işleme kütüphanesi
import numpy as np  # NumPy - Sayısal işlemler ve matris hesaplamaları için
from segmentasyon import renk_segmentasyonu, maskeyi_uygula  # Renk tabanlı segmentasyon fonksiyonları
from morfoloji import tam_iyilestirme, acma, kapama  # Morfolojik işlem fonksiyonları
from renk_gruplama import dominant_renkler_bul, delta_e_cie76, renkleri_grupla  # Renk analizi fonksiyonları


def gorev1_renk_uzayi_donusumleri(goruntu: np.ndarray):
    """
    Görev 1: RGB'den HSV ve LAB'a dönüşüm
    
    Neden farklı renk uzayları kullanıyoruz?
    - RGB: Kameralar ve ekranlar için doğal format ama renk segmentasyonu için zor
    - HSV: Renk tonunu (Hue) aydınlatmadan (Value) ayırdığı için renk tespiti çok kolay
    - LAB: İnsan gözünün renk algısına en yakın format, renk mesafesi hesaplamak için ideal
    """
    
    # RGB'den HSV'ye dönüştürüyorum
    # HSV = Hue (Renk Tonu: 0-179), Saturation (Doygunluk: 0-255), Value (Parlaklık: 0-255)
    # Örnek: Kırmızı bir araba gölgede de güneşte de aynı Hue değerine sahip olur
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    # RGB'den LAB'a dönüştürüyorum
    # LAB = L (Parlaklık: 0-255), a (Yeşil-Kırmızı eksen), b (Mavi-Sarı eksen)
    # Delta E hesaplaması için LAB kullanıyoruz çünkü insan gözü gibi çalışıyor
    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    
    # Sonuçları ekranda gösteriyorum
    cv2.imshow('Orijinal Goruntu (RGB)', goruntu)
    cv2.imshow('HSV Goruntu', hsv)
    cv2.imshow('LAB Goruntu', lab)
    cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar bekliyorum
    cv2.destroyAllWindows()  # Tüm pencereleri kapatıyorum


def gorev2_hsv_segmentasyon(goruntu: np.ndarray):
    """
    Görev 2: HSV ile çoklu renk segmentasyonu
    
    Segmentasyon nedir?
    - Görüntüdeki belirli renkteki pikselleri ayırma işlemi
    - Örnek: Kırmızı arabayı arka plandan ayırmak
    
    Neden HSV kullanıyoruz?
    - RGB'de kırmızı renk birçok farklı değer alabilir (gölge, ışık vb.)
    - HSV'de sadece Hue (renk tonu) değerine bakarak rengi tespit edebiliriz
    - Aydınlatma değişse bile renk tonu aynı kalır
    """
    
    # Tespit edeceğim renkler
    renkler = ['kirmizi', 'yesil', 'mavi']
    maskeler = {}  # Her renk için maske saklayacağım (maske = siyah-beyaz görüntü)
    segmente_nesneler = {}  # Her renk için segmente edilmiş renkli görüntü
    
    # Her renk için ayrı ayrı segmentasyon yapıyorum
    for renk in renkler:
        # HSV eşikleme ile belirli renkteki pikselleri buluyorum
        # Maske: Beyaz (255) = renk bulundu, Siyah (0) = renk yok
        maske = renk_segmentasyonu(goruntu, renk)
        
        # Maskeyi orijinal görüntüye uyguluyorum
        # Sadece beyaz olan yerlerdeki pikselleri gösteriyor, geri kalanı siyah yapıyor
        segmente_nesne = maskeyi_uygula(goruntu, maske)
        
        # Sonuçları saklıyorum (Görev 4'te kullanacağım)
        maskeler[renk] = maske
        segmente_nesneler[renk] = segmente_nesne
    
    # Sonuçları ekranda gösteriyorum
    for renk in renkler:
        # Maskeyi BGR formatına çeviriyorum (yan yana koymak için aynı formatta olmalı)
        maske_bgr = cv2.cvtColor(maskeler[renk], cv2.COLOR_GRAY2BGR)
        
        # Maske ve segmente edilmiş görüntüyü yan yana koyuyorum
        # Sol: Maske (siyah-beyaz), Sağ: Segmente edilmiş nesne (renkli)
        birlesik = np.hstack([maske_bgr, segmente_nesneler[renk]])
        
        # Ekrana sığması için %50 küçültüyorum
        yukseklik, genislik = birlesik.shape[:2]
        yeni_genislik = int(genislik * 0.5)
        yeni_yukseklik = int(yukseklik * 0.5)
        birlesik_kucuk = cv2.resize(birlesik, (yeni_genislik, yeni_yukseklik))
        
        cv2.imshow(f'{renk.capitalize()} - Maske | Segmente', birlesik_kucuk)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return maskeler  # Görev 4'te morfolojik işlemler için maskeleri döndürüyorum


def gorev3_delta_e_gruplama(goruntu: np.ndarray):
    """
    Görev 3: LAB Delta E ile renk gruplama
    
    Bu görevde ne yapıyoruz?
    1. K-Means ile görüntüdeki en yaygın 8 rengi buluyoruz
    2. Delta E ile bu renkler arasındaki "algısal mesafeyi" ölçüyoruz
    3. Benzer renkleri grupluyoruz
    
    Delta E nedir?
    - İki renk arasında insan gözünün algıladığı fark
    - Delta E < 1: Göz farkı göremez
    - Delta E < 20: Benzer renkler
    - Delta E > 50: Tamamen farklı renkler
    
    Neden LAB kullanıyoruz?
    - RGB'de Öklid mesafesi insan algısına uymuyor
    - LAB'da Delta E hesabı insan gözü gibi çalışıyor
    """
    
    # K-Means kümeleme algoritması ile 8 baskın rengi buluyorum
    # K-Means: Tüm pikselleri 8 gruba ayırıyor, her grubun merkezi = baskın renk
    k = 8
    dominant = dominant_renkler_bul(goruntu, k=k)  # LAB formatında 8 renk döndürüyor
    
    # Benzer renkleri grupluyorum
    # Eşik = 20: Delta E < 20 olan renkler aynı grupta sayılıyor
    renkler_liste = [tuple(r) for r in dominant]
    gruplar = renkleri_grupla(renkler_liste, esik=20.0)
    
    # Baskın renkleri görselleştirmek için renk paleti oluşturuyorum
    # Her renk için 100 piksel genişliğinde bir alan ayırıyorum
    renk_paleti = np.zeros((100, k * 100, 3), dtype=np.uint8)
    
    for i, renk_lab in enumerate(dominant):
        # LAB rengini BGR'ye çeviriyorum (ekranda göstermek için)
        # OpenCV ekranda BGR formatı bekliyor
        lab_pixel = np.uint8([[renk_lab]])
        bgr_pixel = cv2.cvtColor(lab_pixel.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)
        renk_bgr = bgr_pixel[0, 0]
        
        # Paletteki ilgili alana bu rengi yerleştiriyorum
        renk_paleti[:, i*100:(i+1)*100] = renk_bgr
    
    # Renk paletini ekranda gösteriyorum
    cv2.imshow('Baskin Renk Paleti (LAB)', renk_paleti)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gorev4_morfolojik_iyilestirme(goruntu: np.ndarray, maskeler: dict):
    """
    Görev 4: Morfolojik işlemlerle iyileştirme
    
    Segmentasyon maskeleri neden gürültülü olur?
    - Aydınlatma farklılıkları
    - Yansımalar
    - Gölgeler
    - Benzer renkli arka plan nesneleri
    
    Morfolojik işlemler nedir?
    - Açma (Opening): Önce Erozyon, sonra Dilatasyon
      * Küçük beyaz noktaları (gürültü) kaldırır
      * Nesnenin kenarlarını biraz inceltir ama sonra geri genişletir
    
    - Kapama (Closing): Önce Dilatasyon, sonra Erozyon
      * Nesnelerin içindeki küçük siyah delikleri doldurur
      * Nesnenin kenarlarını biraz genişletir ama sonra geri inceltir
    
    Neden bu sırayla yapıyoruz?
    1. Önce Açma: Gürültüyü temizle
    2. Sonra Kapama: Delikleri doldur
    """
    
    # Her renk için morfolojik işlemler uyguluyorum
    for renk, maske in maskeler.items():
        # 1. Açma işlemi - Gürültüyü temizliyorum
        # 3x3 kernel: Küçük bir pencere ile tarama yapıyor
        # Küçük beyaz noktaları (gürültü) kaldırıyor
        maske_acma = acma(maske, kernel_boyut=3)
        
        # 2. Kapama işlemi - Delikleri dolduruyorum
        # 7x7 kernel: Daha büyük pencere ile tarama yapıyor
        # Nesnelerin içindeki küçük siyah delikleri beyaz yapıyor
        maske_kapama = kapama(maske_acma, kernel_boyut=7)
        
        # 3. Tam iyileştirme - Açma + Kapama birlikte
        # En temiz sonucu almak için ikisini birlikte uyguluyorum
        maske_iyilestirilmis = tam_iyilestirme(maske, acma_boyut=3, kapama_boyut=7)
        
        # İyileştirilmiş maskeyi orijinal görüntüye uyguluyorum
        nesne_iyilestirilmis = maskeyi_uygula(goruntu, maske_iyilestirilmis)
        
        # Ham maskeyi de uyguluyorum (karşılaştırma için)
        nesne_ham = maskeyi_uygula(goruntu, maske)
        
        # Tüm aşamaları 2x3 grid halinde gösteriyorum
        # Üst satır: Ham maske → Açma → Kapama
        ust_satir = np.hstack([
            cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR),  # Orijinal (gürültülü) maske
            cv2.cvtColor(maske_acma, cv2.COLOR_GRAY2BGR),  # Gürültü temizlenmiş
            cv2.cvtColor(maske_kapama, cv2.COLOR_GRAY2BGR)  # Delikler doldurulmuş
        ])
        
        # Alt satır: İyileştirilmiş maske → Ham nesne → İyileştirilmiş nesne
        # Böylece önce-sonra karşılaştırması yapabilirsin
        alt_satir = np.hstack([
            cv2.cvtColor(maske_iyilestirilmis, cv2.COLOR_GRAY2BGR),  # Final maske
            nesne_ham,  # İyileştirme öncesi (gürültülü)
            nesne_iyilestirilmis  # İyileştirme sonrası (temiz)
        ])
        
        # İki satırı birleştiriyorum (üst + alt)
        birlesik_goruntu = np.vstack([ust_satir, alt_satir])
        
        # Ekrana sığması için %50 küçültüyorum
        yukseklik, genislik = birlesik_goruntu.shape[:2]
        yeni_genislik = int(genislik * 0.5)
        yeni_yukseklik = int(yukseklik * 0.5)
        birlesik_goruntu_kucuk = cv2.resize(birlesik_goruntu, (yeni_genislik, yeni_yukseklik))
        
        cv2.imshow(f'{renk.capitalize()} - Morfolojik Asamalar', birlesik_goruntu_kucuk)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ana():
    """
    Ana fonksiyon - Tüm görevleri sırayla çalıştırıyorum
    
    Program akışı:
    1. Görüntüyü yükle
    2. Renk uzayı dönüşümleri yap (RGB → HSV, LAB)
    3. HSV ile renk segmentasyonu yap
    4. LAB ile renk gruplama yap
    5. Morfolojik işlemlerle maskeleri iyileştir
    """
    
    # Görüntüyü dosyadan yüklüyorum
    goruntu_yolu = "araba.jpeg"
    goruntu = cv2.imread(goruntu_yolu)  # BGR formatında okuyor (OpenCV standardı)
    
    # Görüntü bulunamazsa programı sonlandırıyorum
    if goruntu is None:
        return
    
    # Görevleri sırayla çalıştırıyorum
    gorev1_renk_uzayi_donusumleri(goruntu)  # RGB → HSV, LAB dönüşümleri
    maskeler = gorev2_hsv_segmentasyon(goruntu)  # Kırmızı, yeşil, mavi segmentasyonu
    gorev3_delta_e_gruplama(goruntu)  # Baskın renkleri bul ve grupla
    gorev4_morfolojik_iyilestirme(goruntu, maskeler)  # Maskeleri temizle ve iyileştir


# Program buradan başlıyor
# Bu kontrol sayesinde dosya import edildiğinde ana() çalışmıyor
# Sadece doğrudan çalıştırıldığında (python mainCopy.py) ana() çalışıyor
if __name__ == "__main__":
    ana()
