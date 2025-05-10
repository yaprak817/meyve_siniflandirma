# FRUITS DATASETİ İLE MEYVE TAHMİNİ

Bu proje, görüntü tabanlı bir derin öğrenme modeli ile elma, muz, çilek, portakal ve ananas gibi meyveleri sınıflandırmak üzere geliştirilmiştir. PyTorch ile eğitilen model, kullanıcıların kendi görsellerini yükleyerek sınıf tahmini yapabilecekleri interaktif bir web arayüzü (Streamlit) ile sunulmaktadır.

## Proje Hakkında

Proje, 5 farklı meyve kategorisini sınıflandırmak üzere sıfırdan eğitilmiş bir **ResNet18 CNN** modelini içerir:

## Veri Seti

Fruits Image Classification veri seti şu kategorileri içerir:
- Apple (Elma)  
- Banana (Muz)  
- Orange (Portakal)  
- Strawberry (Çilek)  
- Pineapple (Ananas)

Veri seti eğitim ve test olmak üzere iki alt kümeden oluşmaktadır.

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri setini indirin ve aşağıdaki dizin yapısını oluşturun:
```
classifier2/
├── data/
│   └── train/
└── data/
    └── test/
```

## Kullanım

### Model Eğitimi

Modeli eğitmek için:
```bash
python train_model.py
```

Bu komut:

- Görselleri yeniden boyutlandırır (100x100)
- Normalize eder
- Modeli sıfırdan eğitir (10 epoch)
- Her epoch sonunda accuracy, precision, recall, f1-score metriklerini hesaplar
- Kayıp (loss) değerlerini çizerek görselleştirir
- Eğitilen modeli model/classifier.pth dosyasına kaydeder

### Web Arayüzü

Web arayüzünü başlatmak için:
```bash
streamlit run app.py
```

Arayüz şu özellikleri sunar:

- Görsel yükleme ve önizleme
- Seçilen görsel için meyve sınıf tahmini
- Gerçek zamanlı sınıflandırma sonucu
- Kullanımı kolay ve sade bir arayüz

## Model Mimarisi

- Model olarak torchvision.models.resnet18 kullanılmıştır:
- Pretrained = False (sıfırdan eğitildi)
- Son katmanı (fc) değiştirilerek 5 sınıfa uygun hale getirildi: model.fc = nn.Linear(..., 5)

##  Eğitim Sonuçları (Örnek)

Epoch 1/10 - Loss: 1.4829, Accuracy: 0.52, Precision: 0.53, Recall: 0.50, F1 Score: 0.51

Epoch 10/10 - Loss: 0.2247, Accuracy: 0.94, Precision: 0.94, Recall: 0.93, F1 Score: 0.94


## Veri Ön İşleme
- Görüntüler aşağıdaki işlemlerden geçmektedir:
- Yeniden boyutlandırma: 100x100
- Tensor dönüşümü
- Normalize: (0.5, 0.5) ortalama ve standart sapma ile
  

## Sınıf İsimleri

class_names = ['Apple', 'Banana', 'Orange', 'Strawberry', 'Pineapple']


##  Özellikler
- 5 farklı meyve sınıfı
- Kendi görüntünüzü yükleyerek sınıf tahmini yapma
- %90+ doğruluk oranı
- Web tabanlı kullanıcı arayüzü
- Eğitim sırasında başarı metriklerinin izlenmesi


## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Dalınıza push yapın (`git push origin yeni-ozellik`)
5. Bir Pull Request oluşturun

## İletişim

Sorularınız veya önerileriniz için GitHub üzerinden bir Issue açabilirsiniz. 
