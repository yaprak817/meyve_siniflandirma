import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

'''
modelin eğitilmesi için torch ve torchvison kütüphanelerini kullandım
başarı metriklerini (accuarcy,precision,recall ve f1-score) ölçmek için
scikit-learn kütüphanesini kullanıdm. loss değerlerini grafiksel olarak 
göstermek için de matplotlib kütüphanesini kullandım.
'''


# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
ön işleme ksımında da ilk olarak farklı boyuttaki görüntüleri aynı boyuta
getirerek analizi daha rahat yapmaktı dah sonra görüntü piksellerin de 
aynı olması adına onu da 0,5 olarak normalize ettim.
'''

# Veri ön işleme
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Veri seti
train_dataset = datasets.ImageFolder(root="data/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

'''
modeli eğitirken meyce görseller için sadece göstermek adına 5 tane 
meyceyi tek kullandım. bunlar :
    -apple, -pineapple, -strawberry, -banane, -orange
    '''
    
# Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 sınıf
model = model.to(device)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

'''
görüntüleri sürekli taraması için döngü adımını 10 olarak belirledim.
'''

# Eğitim
epochs = 10
loss_history = []

for epoch in range(epochs):
    running_loss = 0.0
    all_preds = []
    all_labels = []

    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Tahminleri ve gerçek etiketleri topla
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        '''
        bu kısımda da başarı metrikleri olan accuarcy, recall, precision
        ve f1 score sonuçlarını gösterdim ama bunu her adımın sonunda 
        gösterdim ki hangi döngüde ne kadar kayıp olmuş her döngünün
        jayıp maliyeti aynı mı diye bakmak istedim.
        '''
        

    # Epoch sonunda metrik hesapla
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

# Modeli kaydet
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "classifier.pth")
torch.save(model.state_dict(), model_path)
print(f"Model kaydedildi: {model_path}")

# Loss grafiği
plt.plot(range(1, epochs+1), loss_history, marker='o')
plt.title('Eğitim Kayıp Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()
