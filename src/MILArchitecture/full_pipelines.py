from src.configuration.config import SEED, set_seed, LOCKED_CONFIG
from src.training.evaluation import full_test_evaluation_per_split
from src.training.training import full_training_per_split
from src.MILArchitecture.AttentionMIL import AttentionMIL
import torch
import torch.nn as nn
import pandas as pd
from src.training.loaders import initialize_loaders_per_split
from src.utils.data import sources
from src.MILArchitecture.windows import initialize_all_bags
from src.visualization.confusion_matrix import heatmap_cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score, classification_report
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def attention_based_mil(df, au_cols, tensorboard=False):
    loss = 0.0
    test_all_y_true = []
    test_all_y_pred = []
    all_attention_results = []

    bags, labels, clip_ids, source_ids, metadata = initialize_all_bags(
        df=df,
        au_cols=au_cols,
        config=LOCKED_CONFIG
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for fold_idx, test_source in enumerate(sources):
        # print()
        # print("-" * 20)
        # print(f"Test Source: {test_source}")
        # print("-" * 20)

        fold_seed = SEED + fold_idx
        set_seed(fold_seed)

        train_loader, val_loader, test_loader = initialize_loaders_per_split(
            test_source=test_source,
            sources=sources,
            metadata=metadata,
            bags=bags,
            labels=labels,
            clip_ids=clip_ids,
            config=LOCKED_CONFIG,
            seed=fold_seed
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = AttentionMIL(
            input_dim=len(au_cols), dropout=LOCKED_CONFIG["dropout"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LOCKED_CONFIG["learning_rate"],
            weight_decay=LOCKED_CONFIG["weight_decay"]
        )

        criterion = nn.BCEWithLogitsLoss()

        if tensorboard:
            writer = SummaryWriter(
                log_dir=(
                    f"../runs/attention_mil_{timestamp}/"
                    f"{test_source}"
                )
            )

            best_model_state = full_training_per_split(
                num_epochs=LOCKED_CONFIG["num_epochs"],
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=LOCKED_CONFIG,
                writer=writer,
                log=False
            )

            writer.close()
        else:
            best_model_state = full_training_per_split(
                num_epochs=LOCKED_CONFIG["num_epochs"],
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=LOCKED_CONFIG,
                writer=None,
                log=False
            )


        avg_loss, test_y_true, test_y_pred, attention_results = (
            full_test_evaluation_per_split(
                best_model_state=best_model_state,
                results=None,
                model=model,
                test_loader=test_loader,
                test_source=test_source,
                criterion=criterion,
                device=device,
                config=LOCKED_CONFIG,
                log=False
            )
        )

        loss += avg_loss
        test_all_y_true.extend(test_y_true)
        test_all_y_pred.extend(test_y_pred)
        all_attention_results.extend(attention_results)

    loss /= len(sources)

    acc = balanced_accuracy_score(
        test_all_y_true,
        test_all_y_pred
    )

    cr = classification_report(
        test_all_y_true,
        test_all_y_pred,
        labels=[0, 1],
        zero_division=0
    )

    print("Accuracy (Balanced):", acc)
    print("Avg. Loss:", loss)
    print(cr)

    heatmap_cm(
        confusion_matrix(
            test_all_y_true,
            test_all_y_pred
        )
    )

    return all_attention_results

def attention_based_mil_combination_comparison(df, au_cols, log=False):
    results = []

    for au in au_cols:
        # print('=' * 40)
        # print(f"TESTED AU(s): {au}")
        # print('=' * 40)

        if isinstance(au, (list, tuple)):
            dim = len(au)
        else:
            dim = 1

        avg_loss = 0.0
        all_y_true_per_au = []
        all_y_pred_per_au = []

        # CREATING WINDOWS PER ID CLIP
        bags, labels, clip_ids, source_ids, metadata = initialize_all_bags(df=df, au_cols=au, config=LOCKED_CONFIG)

        # LOSO-Split
        for fold_idx, test_source in enumerate(sources):
            # print()
            # print('-' * 20)
            # print(f"Test Source: {test_source}")
            # print('-' * 20)

            fold_seed = SEED + fold_idx
            set_seed(fold_seed)

            # Initializing Train, Validation and Test DataLoader
            train_loader, val_loader, test_loader = initialize_loaders_per_split(test_source=test_source,
                                                                                 sources=sources, metadata=metadata,
                                                                                 bags=bags, labels=labels,
                                                                                 clip_ids=clip_ids,
                                                                                 config=LOCKED_CONFIG, seed=fold_seed)

            # Initializing Attention MIL model and Hyperparameters
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = AttentionMIL(input_dim=dim, dropout=LOCKED_CONFIG["dropout"]).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LOCKED_CONFIG["learning_rate"],
                weight_decay=LOCKED_CONFIG["weight_decay"]
            )
            criterion = nn.BCEWithLogitsLoss()

            num_epochs = LOCKED_CONFIG["num_epochs"]

            # Training & Validation
            best_model_state = full_training_per_split(num_epochs=num_epochs, model=model, optimizer=optimizer,
                                                       criterion=criterion, train_loader=train_loader,
                                                       val_loader=val_loader, device=device, config=LOCKED_CONFIG,
                                                       log=log)

            # Load Checkpoint & Evaluate on Testset
            test_avg_loss, test_y_true, test_y_pred, _ = full_test_evaluation_per_split(
                best_model_state=best_model_state, results=results, model=model, test_loader=test_loader,
                test_source=test_source, criterion=criterion, device=device, config=LOCKED_CONFIG, single_signal=au,
                log=log)

            avg_loss += test_avg_loss
            all_y_true_per_au.extend(test_y_true)
            all_y_pred_per_au.extend(test_y_pred)

        acc = balanced_accuracy_score(y_true=all_y_true_per_au, y_pred=all_y_pred_per_au)
        loss = avg_loss / len(sources)
        cr = classification_report(
            all_y_true_per_au,
            all_y_pred_per_au,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )

        results.append({
            "au": au,
            "loss": loss,
            "bal. accuracy": acc,
            "precision_0": cr["0"]["precision"],
            "precision_1": cr["1"]["precision"],
            "recall_0": cr["0"]["recall"],
            "recall_1": cr["1"]["recall"],
            "f1_0": cr["0"]["f1-score"],
            "f1_1": cr["1"]["f1-score"]

        })

    summary = pd.DataFrame(results)
    summary = summary.sort_values("bal. accuracy", ascending=False)

    return summary