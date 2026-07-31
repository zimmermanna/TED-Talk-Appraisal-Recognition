import copy

import torch
from sklearn.metrics import balanced_accuracy_score

from src.training.evaluation import evaluate_one_epoch


def train_one_epoch(model, train_loader, device, optimizer, criterion, config):
    threshold = config["threshold"]
    model.train()

    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    for X, y, mask, batch_clip_ids in train_loader:
        X = X.to(device)
        y = y.to(device).float()
        mask = mask.to(device)

        optimizer.zero_grad()

        logits, attention_weights = model(X, mask)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()

        all_y_true.extend(y.cpu().numpy())
        all_y_pred.extend(preds.cpu().numpy())

        total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    balanced_acc = balanced_accuracy_score(all_y_true,all_y_pred)

    return avg_loss, balanced_acc

def full_training_per_split(
    num_epochs,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    config,
    writer=None,
    log=True
):
    best_val_acc = float("-inf")
    best_model_state = None

    for epoch in range(num_epochs):
        train_avg_loss, train_bal_acc = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
            config
        )

        val_avg_loss, val_bal_acc, _, _, _ = evaluate_one_epoch(
            model,
            val_loader,
            device,
            criterion,
            config
        )

        # Checkpointing
        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            best_model_state = copy.deepcopy(model.state_dict())

        # TensorBoard logging
        if writer is not None:
            writer.add_scalars(
                "Loss",
                {
                    "Train": train_avg_loss,
                    "Validation": val_avg_loss
                },
                epoch + 1
            )

            writer.add_scalars(
                "Balanced Accuracy",
                {
                    "Train": train_bal_acc,
                    "Validation": val_bal_acc
                },
                epoch + 1
            )
            writer.flush()

        if log:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Avg. Loss: {train_avg_loss:.4f} | "
                f"Val Avg. Loss: {val_avg_loss:.4f}"
            )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Bal. Acc: {train_bal_acc:.4f} | "
                f"Val Bal. Acc: {val_bal_acc:.4f}"
            )

            print("-" * 20)

    return best_model_state


