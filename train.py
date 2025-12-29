import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

"""
train.py

CIFAR-10 veri seti üzerinde ResNet18 modelini eğitir.
Eğitim sonrası test setinde:
- Accuracy
- Precision (macro)
- Recall (macro)
metriklerini hesaplar ve ekrana yazdırır.

Eğitilmiş model 'models/model.pth' dosyasına kaydedilir.
"""


# CIFAR-10 sınıf isimleri
CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# Ön işleme
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Dataset otomatik indirir
train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tf)
test_ds  = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

# Model (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

# Test
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        out = model(imgs)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

print("\n=== Test Sonuçları ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)

# Modeli kaydet
os.makedirs("models", exist_ok=True)
torch.save({"model": model.state_dict(), "classes": CLASSES}, "models/model.pth")
print("\nModel kaydedildi: models/model.pth")
