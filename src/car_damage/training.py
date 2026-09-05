"""Bucle de entrenamiento genérico, compartido entre el CNN desde cero y ResNet18."""
import torch
from sklearn.metrics import accuracy_score


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    return avg_loss, acc


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler,
                 epochs, patience, tag, device, ckpt_path=None):
    best_val_loss, patience_counter = float("inf"), 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    ckpt_path = ckpt_path or f"/tmp/best_{tag}.pt"

    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"[{tag}] Epoch {epoch + 1}/{epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{tag}] Early stopping en epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(ckpt_path))
    return history, ckpt_path
