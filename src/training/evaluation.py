import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, log_loss



def evaluate_one_epoch(model, data_loader, device, criterion, config):
    threshold = config["threshold"]
    model.eval()

    total_loss = 0.0
    all_y_true = []
    all_y_pred = []
    attention_results = []

    with torch.no_grad():
        for X, y, mask, batch_clip_ids in data_loader:
            X = X.to(device)
            y = y.to(device).float()
            mask = mask.to(device)

            logits, attention_weights = model(X, mask)

            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())

            # Saving Attention Metrics for visualization
            if attention_weights.dim() == 3:
                attention_weights = attention_weights.squeeze(-1)

            for i in range(X.size(0)):
                valid_len = mask[i].sum().item()

                x_i = X[i, :valid_len]
                att_i = attention_weights[i, :valid_len]

                attention_results.append({
                    "clip_id": batch_clip_ids[i],
                    "true_label": int(y[i].item()),
                    "pred_prob": float(probs[i].item()),
                    "pred_label": int(preds[i].item()),
                    "attention": att_i.cpu().numpy(),
                    "X": x_i.cpu().numpy()
                })

    balanced_acc = balanced_accuracy_score(all_y_true, all_y_pred)
    avg_loss = total_loss / len(data_loader.dataset)

    all_y_true = np.array(all_y_true).astype(int)
    all_y_pred = np.array(all_y_pred).astype(int)

    return avg_loss, balanced_acc, all_y_true, all_y_pred, attention_results

def full_test_evaluation_per_split(best_model_state, model, test_loader, test_source, criterion, device, config, single_signal=None, results=None):
    model.load_state_dict(best_model_state)

    test_avg_loss, test_bal_acc, test_all_y_true, test_all_y_pred, attention_results = evaluate_one_epoch(
        model,
        test_loader,
        device,
        criterion,
        config
    )
    test_report = classification_report(
        test_all_y_true,
        test_all_y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )
    print(f"Bal. Accuracy: {test_bal_acc}")
    print(f"Avg. Loss:  {test_avg_loss}")
    print(confusion_matrix(test_all_y_true, test_all_y_pred, labels=[0, 1]))

    if not single_signal and results:
        results.append({
            "test_source": test_source,
            "loss": test_avg_loss,
            "bal. accuracy": test_bal_acc,
            "precision_0": test_report["0"]["precision"],
            "precision_1": test_report["1"]["precision"],
            "recall_0": test_report["0"]["recall"],
            "recall_1": test_report["1"]["recall"],
            "f1_0": test_report["0"]["f1-score"],
            "f1_1": test_report["1"]["f1-score"]
        })

    return test_avg_loss, test_all_y_true, test_all_y_pred, attention_results
