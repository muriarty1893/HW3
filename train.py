import pandas as pd
import numpy as np

file_path = 'DataForPerceptron.xlsx' # excel dosyasından eğitim ve test verilerini yükle.
train_data = pd.read_excel(file_path, sheet_name='TRAINData')
test_data = pd.read_excel(file_path, sheet_name='TESTData')

X_train = train_data.iloc[:, :-1].values # eğitim verileri için özellikleri ve etiketleri ayır.
y_train = train_data.iloc[:, -1].values
y_train = np.where(y_train == 0, -1, 1)

X_test = test_data.iloc[:, :-1].values # test verileri için özellikleri ve etiketleri ayır.
y_test = test_data.iloc[:, -1].values

if pd.isna(y_test).all():
    y_test = None
else:
    y_test = np.where(y_test == 0, -1, 1)

class Perceptron: # perceptron öğrenme algoritması.
    def __init__(self, learning_rate=0.1, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)  # for the bias term (w0) or (tetha0).
        X_augmented = np.c_[np.ones(n_samples), X]

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X_augmented):
                condition = y[idx] * (np.dot(x_i, self.weights)) <= 0
                if condition:
                    self.weights += self.learning_rate * y[idx] * x_i

    def predict(self, X):
        n_samples = X.shape[0]
        X_augmented = np.c_[np.ones(n_samples), X]
        linear_output = np.dot(X_augmented, self.weights)
        return np.where(linear_output >= 0, 1, -1)

perceptron_model = Perceptron(learning_rate=0.1, max_iter=1000) # perceptron'u eğit.
perceptron_model.fit(X_train, y_train)

y_pred = perceptron_model.predict(X_test) # test verileri üzerinde prediction yap.

y_train_pred = perceptron_model.predict(X_train) # eğitim verileri üzerinde tahmin yap.

if y_test is not None: # test verileri üzerinde doğruluğu hesapla (etiketler mevcutsa).
    test_accuracy = np.mean(y_pred == y_test)
    print(f'test doğruluğu: {test_accuracy * 100:.2f}%')
else:
    print('test doğruluğu: n/a (test etiketleri mevcut değil)')

train_accuracy = np.mean(y_train_pred == y_train) # eğitim verileri üzerinde doğruluğu hesapla.
print(f'eğitim doğruluğu: {train_accuracy * 100:.2f}%') # eğitim işlemi doğruluğunu göster

print('test tahminleri:\n', y_pred) # tahminleri göster (1 doğru).
