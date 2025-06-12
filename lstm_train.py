import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class LSTMClassifier(nn.Module):
    """Простая LSTM-классификация."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10,
          lr: float = 0.001):
    """Обучает модель и возвращает историю ошибок и точности."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                output = model(xb)
                loss = criterion(output, yb)
                val_loss += loss.item() * xb.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == yb).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}: "
              f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    return history


def plot_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(history["val_acc"], label="val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy.png")


def main():
    torch.manual_seed(0)
    seq_len = 20
    input_size = 10
    num_classes = 3

    X = torch.randn(200, seq_len, input_size)
    y = torch.randint(0, num_classes, (200,))

    X_train, y_train = X[:160], y[:160]
    X_val, y_val = X[160:], y[160:]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = LSTMClassifier(input_size=input_size, num_classes=num_classes)

    history = train(model, train_loader, val_loader, epochs=10)

    with open("val_accuracy.txt", "w") as f:
        for acc in history["val_acc"]:
            f.write(f"{acc}\n")

    plot_history(history)


if __name__ == "__main__":
    main()
