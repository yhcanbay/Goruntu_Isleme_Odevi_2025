"""
Morfolojik İşlemler
===================
Segmentasyon sonuçlarını iyileştirmek için açma/kapama işlemleri.

Morfolojik işlemler binary (ikili) görüntülerde şekil analizi yapar:
- Erozyon: Nesneleri küçültür, ince bağlantıları koparır
- Dilatasyon (Genişleme): Nesneleri büyütür, delikleri doldurur
- Açma: Erozyon + Dilatasyon (gürültü temizler)
- Kapama: Dilatasyon + Erozyon (delikleri doldurur)
"""

import cv2  # OpenCV - Görüntü işleme kütüphanesi
import numpy as np  # NumPy - Sayısal işlemler için


def kernel_olustur(boyut: int = 5, sekil: str = 'kare') -> np.ndarray:
    """
    Morfolojik işlemler için yapısal element (kernel) oluşturur.
    
    Kernel: Morfolojik işlemlerde kullanılan küçük matris (yapısal element)
    Kernel'in şekli ve boyutu işlemin etkisini belirler:
    - Kare: Genel amaçlı, tüm yönlerde eşit etki
    - Daire: Daha yumuşak, dairesel nesneler için
    - Çapraz: Sadece yatay ve dikey yönlerde etki
    
    Args:
        boyut: Kernel boyutu (tek sayı olmalı, örn: 3, 5, 7)
        sekil: 'kare', 'daire', veya 'capraz'
    
    Returns:
        np.ndarray: Yapısal element (kernel matrisi)
    """
    if sekil == 'kare':
        # Dikdörtgen şeklinde kernel (MORPH_RECT)
        # Örnek 3x3: [[1,1,1], [1,1,1], [1,1,1]]
        return cv2.getStructuringElement(cv2.MORPH_RECT, (boyut, boyut))
    elif sekil == 'daire':
        # Elips/daire şeklinde kernel (MORPH_ELLIPSE)
        # Köşeler yuvarlatılmış, daha yumuşak işlem
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boyut, boyut))
    elif sekil == 'capraz':
        # Çapraz şeklinde kernel (MORPH_CROSS)
        # Sadece merkez, üst, alt, sağ, sol (+ şeklinde)
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (boyut, boyut))
    else:
        raise ValueError(f"Bilinmeyen şekil: {sekil}")


def acma(maske: np.ndarray, kernel_boyut: int = 5) -> np.ndarray:
    """
    Açma (Opening) işlemi: Erozyon + Genişleme (Dilation)
    
    Amaç: Küçük gürültüleri (beyaz noktaları) temizler
    Çalışma prensibi:
    1. Erozyon: Küçük beyaz noktaları yok eder
    2. Dilatasyon: Kalan nesneleri orijinal boyutlarına yaklaştırır
    
    Kullanım alanları:
    - Segmentasyondaki küçük gürültüleri temizleme
    - İnce bağlantıları koparma
    - Küçük nesneleri çıkarma
    
    Args:
        maske: Binary maske (0 veya 255)
        kernel_boyut: Yapısal element boyutu (büyük değer = daha güçlü temizleme)
    
    Returns:
        np.ndarray: İşlenmiş maske (gürültüsüz)
    """
    kernel = kernel_olustur(kernel_boyut, 'kare')  # Kare şeklinde kernel oluştur
    # morphologyEx ile MORPH_OPEN işlemi: Erozyon sonra dilatasyon
    return cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel)


def kapama(maske: np.ndarray, kernel_boyut: int = 5) -> np.ndarray:
    """
    Kapama (Closing) işlemi: Genişleme (Dilation) + Erozyon
    
    Amaç: Küçük delikleri (siyah noktaları) doldurur
    Çalışma prensibi:
    1. Dilatasyon: Nesneleri büyütür, delikleri doldurur
    2. Erozyon: Nesneleri orijinal boyutlarına yaklaştırır
    
    Kullanım alanları:
    - Nesnelerdeki küçük delikleri doldurma
    - Kopuk bölgeleri birleştirme
    - Nesne konturlarını düzleştirme
    
    Args:
        maske: Binary maske (0 veya 255)
        kernel_boyut: Yapısal element boyutu (büyük değer = daha büyük delikleri doldurur)
    
    Returns:
        np.ndarray: İşlenmiş maske (delikler doldurulmuş)
    """
    kernel = kernel_olustur(kernel_boyut, 'kare')  # Kare şeklinde kernel oluştur
    # morphologyEx ile MORPH_CLOSE işlemi: Dilatasyon sonra erozyon
    return cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)


def erozyon(maske: np.ndarray, kernel_boyut: int = 3) -> np.ndarray:
    """
    Erozyon: Nesneyi küçültür, kenarları aşındırır.
    
    Çalışma prensibi:
    - Kernel'i her piksel üzerinde kaydır
    - Kernel'in tüm elemanları beyaz üzerindeyse pikseli beyaz yap, değilse siyah
    - Sonuç: Nesneler küçülür, ince bölgeler kaybolur
    
    Kullanım alanları:
    - Nesneleri küçültme
    - İnce bağlantıları koparma
    - Gürültü temizleme
    
    Args:
        maske: Binary maske
        kernel_boyut: Yapısal element boyutu
    
    Returns:
        np.ndarray: Erozyon uygulanmış maske
    """
    kernel = kernel_olustur(kernel_boyut, 'kare')
    # cv2.erode: Erozyon işlemi, iterations: Kaç kez tekrarlanacağı
    return cv2.erode(maske, kernel, iterations=1)


def genisleme(maske: np.ndarray, kernel_boyut: int = 3) -> np.ndarray:
    """
    Genişleme (Dilation): Nesneyi büyütür.
    
    Çalışma prensibi:
    - Kernel'i her piksel üzerinde kaydır
    - Kernel'in herhangi bir elemanı beyaz üzerindeyse pikseli beyaz yap
    - Sonuç: Nesneler büyür, delikler dolar
    
    Kullanım alanları:
    - Nesneleri büyütme
    - Delikleri doldurma
    - Kopuk parçaları birleştirme
    
    Args:
        maske: Binary maske
        kernel_boyut: Yapısal element boyutu
    
    Returns:
        np.ndarray: Dilatasyon uygulanmış maske
    """
    kernel = kernel_olustur(kernel_boyut, 'kare')
    # cv2.dilate: Dilatasyon işlemi, iterations: Kaç kez tekrarlanacağı
    return cv2.dilate(maske, kernel, iterations=1)


def tam_iyilestirme(maske: np.ndarray, 
                    acma_boyut: int = 3, 
                    kapama_boyut: int = 5) -> np.ndarray:
    """
    Tam iyileştirme: Önce açma (gürültü temizle), sonra kapama (delikleri doldur).
    
    Bu işlem segmentasyon sonuçlarını optimize eder:
    1. Açma ile küçük gürültüler (yanlış tespit edilen noktalar) temizlenir
    2. Kapama ile nesnelerdeki delikler (eksik tespit edilen bölgeler) doldurulur
    
    Sonuç: Daha temiz ve daha bütün bir maske
    
    Args:
        maske: Binary maske (segmentasyon sonucu)
        acma_boyut: Açma kernel boyutu (küçük = hafif temizlik, büyük = agresif)
        kapama_boyut: Kapama kernel boyutu (büyük = daha büyük delikleri doldur)
    
    Returns:
        np.ndarray: İyileştirilmiş maske (temiz ve bütün)
    """
    # Önce açma ile küçük gürültüleri temizle
    sonuc = acma(maske, acma_boyut)
    # Sonra kapama ile delikleri doldur
    sonuc = kapama(sonuc, kapama_boyut)
    return sonuc


# Bu modül direkt çalıştırılırsa (python morfoloji.py)
if __name__ == "__main__":
    print("Morfolojik İşlemler Modülü")
    print("Fonksiyonlar: acma(), kapama(), erozyon(), genisleme(), tam_iyilestirme()")

