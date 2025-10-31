# mnist_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
import numpy as np
from LeNet import LeNet
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================
# ⚙️ Cấu hình thiết bị
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================
# 🧩 Collate Function (gom batch)
# ============================================
def collate_fn(items: list) -> dict[torch.Tensor]:
    images, labels = zip(*items)  # unpack tuple (image, label)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"image": images, "label": labels}

# ============================================
# 📦 Dataset class cho MNIST
# ============================================
class MNISTDataset(Dataset):
    def __init__(self, train: bool = True):
        transform = transforms.Compose([
            transforms.ToTensor(),                      # convert [0,255] → [0,1]
            transforms.Normalize((0.1307,), (0.3081,))  # normalize mean/std MNIST
        ])
        self.dataset = datasets.MNIST(
            root="./data",
            train=train,
            transform=transform,
            download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label  # trả tuple để collate_fn xử lý

# ============================================
# 🧠 Chọn mô hình
# ============================================
print("Hay nhap loai mo hinh (1 hoac 3): ")
model_type = input().strip()
if model_type == "1":
    model = LeNet(image_size=(28, 28), num_labels=10).to(device)
elif model_type == "3":
    raise NotImplementedError("Model type 3 chưa được định nghĩa.")
else:
    raise ValueError("Chỉ chấp nhận loại mô hình 1 hoặc 3.")

# ============================================
# ⚙️ Cấu hình huấn luyện
# ============================================
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================
# 📚 Tạo Dataset và DataLoader
# ============================================
train_dataset = MNISTDataset(train=True)
test_dataset = MNISTDataset(train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)

# ============================================
# 📊 Hàm đánh giá mô hình
# ============================================
def evaluate(model, dataloader):
    model.eval()
    outputs = []
    trues = []
    with torch.no_grad():
        for item in dataloader:
            image = item["image"].to(device)
            label = item["label"].to(device)
            output = model(image)
            predictions = torch.argmax(output, dim=-1)

            outputs.extend(predictions.tolist())
            trues.extend(label.tolist())

    return {
        "recall": recall_score(trues, outputs, average="macro"),
        "precision": precision_score(trues, outputs, average="macro"),
        "f1": f1_score(trues, outputs, average="macro"),
    }

# ============================================
# 🚀 Vòng lặp huấn luyện
# ============================================
EPOCHS = 10
for epoch in range(EPOCHS):
    print(f"\nEpoch: {epoch+1}/{EPOCHS}")
    losses = []

    model.train()
    for item in train_dataloader:
        image = item["image"].to(device)
        label = item["label"].to(device)

        # Forward
        output = model(image)
        loss = loss_fn(output, label.long())
        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    print(f"Training Loss: {avg_loss:.4f}")

    # Đánh giá trên tập test
    metrics = evaluate(model, test_dataloader)
    for metric in metrics:
        print(f"{metric}: {metrics[metric]:.4f}")

