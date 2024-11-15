import pandas as pd
import numpy as np

dosyol = 'DataForPerceptron.xlsx' # excel dosyasından eğitim ve test verilerini yükle
ever = pd.read_excel(dosyol, sheet_name='TRAINData') # eğitim 
tver = pd.read_excel(dosyol, sheet_name='TESTData') # test 

X_egitim = ever.iloc[:, :-1].values # eğitim verileri için özellikleri ayır
y_egitim = ever.iloc[:, -1].values # etiketleri ayır

y_egitim = np.where(y_egitim == 2, -1, 1) # etiketleri -1 ve 1 olarak dönüştür

X_test = tver.iloc[:, :-1].values # test verilerinde özellikleri ayır

class Perceptron: # perceptron öğrenme algoritması
    def __init__(self, ogrenme_orani=0.1, max_iter=1000):
        self.ogrenme_orani = ogrenme_orani # öğrenme oranı
        self.max_iter = max_iter # maksimum iterasyon sayısı
        self.agirliklar = None # ağırlıklar (başlangıçta boş)

    def egit(self, X, y):
        ornek_sayisi, ozellik_sayisi = X.shape # örnek ve özellik sayısını al
        self.agirliklar = np.zeros(ozellik_sayisi + 1) # bias için ek ağırlıkla başlat
        X_augmented = np.c_[np.ones(ornek_sayisi), X] # bias terimini ekle

        for _ in range(self.max_iter): # iterasyonları döngüye al
            for i, x_i in enumerate(X_augmented): # her örneği sırayla kontrol et
                if y[i] * (np.dot(x_i, self.agirliklar)) <= 0: # yanlış sınıflandırma kontrolü
                    self.agirliklar += self.ogrenme_orani * y[i] * x_i # ağırlıkları güncelle

    def tahmin_et(self, X):
        ornek_sayisi = X.shape[0] # test örnek sayısını al
        X_augmented = np.c_[np.ones(ornek_sayisi), X] # bias terimini ekle
        linear_output = np.dot(X_augmented, self.agirliklar) # doğrusal çıkış hesapla
        return np.where(linear_output >= 0, 4, 2) # tahmin edilen sınıfları 2 veya 4 olarak döndür

perceptron_model = Perceptron(ogrenme_orani=0.1, max_iter=1000) # perceptron modelini oluştur
perceptron_model.egit(X_egitim, y_egitim) # modeli eğitim verileri ile eğit

y_tahmin = perceptron_model.tahmin_et(X_test) # test verileri üzerinde tahmin yap

print('Test verileri icin tahmin edilen siniflar:\n', y_tahmin) # tahmin edilen sınıfları yazdır
