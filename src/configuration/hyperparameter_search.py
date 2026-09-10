import torch
import torch.nn as nn
import pandas as pd

from src.configuration.config import SEED, set_seed
from src.training.evaluation import full_test_evaluation_per_split
from src.training.training import full_training_per_split
from src.MILArchitecture.AttentionMIL import AttentionMIL
from src.training.loaders import initialize_loaders_per_split
from src.MILArchitecture.windows import initialize_all_bags
from sklearn.metrics import balanced_accuracy_score, classification_report

base_config = {
    # Window generation: Strongly correlated (!)
    "window_size": 20,
    "stride": 10,

    # Optimization
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "dropout": 0.2,

    # Training and evaluation (Not tuned)
    "num_epochs": 25,
    "threshold": 0.5
}

search_space = {
    "window_config": [
        {"window_size": 10, "stride": 5},
        {"window_size": 10, "stride": 10},
        {"window_size": 20, "stride": 10},
        {"window_size": 20, "stride": 20},
        {"window_size": 30, "stride": 15},
        {"window_size": 30, "stride": 30}
    ],

    "learning_rate": [
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2
    ],

    "weight_decay": [
        0.0,
        1e-5,
        1e-4,
        1e-3,
        1e-2
    ],

    "batch_size": [
        8,
        16,
        32,
        64
    ],

    "dropout": [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5
    ]
}

def sequential_hyperparameter_search(
        df,
        test_au,
        base_config,
        search_space,
        sources,
        device
):
    best_config = base_config.copy()
    all_results = []

    for param_name, values in search_space.items():
        # print("\n" + "=" * 50)
        # print(f"TESTING HYPERPARAMETER: {param_name}")
        # print("=" * 50)

        step_results = []

        for value in values:
            config = best_config.copy()

            # Tuning window_size and stride together
            if param_name == "window_config":
                config.update(value)
            else:
                config[param_name] = value

            # print(f"\nTesting {param_name} = {value}")
            # print("Current config:")
            # print(config)

            fold_result = LOSO_loop_config(
                df=df,
                test_au=test_au,
                config=config,
                param_name=param_name,
                param_value=value,
                sources=sources,
                device=device,
                seed=SEED
            )

            step_results.append(fold_result)
            all_results.append(fold_result)

        step_results_df = pd.DataFrame(step_results)

        step_results_df["param_value_group"] = (
            step_results_df["param_value"].astype(str)
        )

        mean_results = (
            step_results_df
            .groupby("param_value_group", as_index=False)["bal. accuracy"]
            .mean()
        )

        best_row = mean_results.loc[
            mean_results["bal. accuracy"].idxmax()
        ]

        best_param_value = next(
            result["param_value"]
            for result in step_results
            if str(result["param_value"]) == best_row["param_value_group"]
        )

        if param_name == "window_config":
            best_config.update(best_param_value)
        else:
            best_config[param_name] = best_param_value

        # print("\nBest result in current step:")
        # print(f"Parameter: {param_name}")
        # print(f"Value: {best_param_value}")
        # print(f"Mean Loss: {best_row['bal. accuracy']:.4f}")

    all_results = pd.DataFrame(all_results)
    return best_config, all_results


def LOSO_loop_config(
        df,
        test_au,
        config,
        param_name,
        param_value,
        sources,
        device,
        input_dim=8,
        seed=42
):
    bags, labels, clip_ids, source_ids, metadata = initialize_all_bags(
        df,
        test_au,
        config
    )

    avg_loss = 0.0
    all_y_true = []
    all_y_pred = []

    for fold_idx, test_source in enumerate(sources):
        fold_seed = seed + fold_idx
        set_seed(fold_seed)

        train_loader, val_loader, test_loader = initialize_loaders_per_split(
            test_source=test_source,
            sources=sources,
            metadata=metadata,
            bags=bags,
            labels=labels,
            clip_ids=clip_ids,
            config=config,
            seed=fold_seed
        )

        model = AttentionMIL(input_dim=input_dim, dropout=config["dropout"]).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        criterion = nn.BCEWithLogitsLoss()

        best_model_state = full_training_per_split(
            num_epochs=config["num_epochs"],
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            log=False
        )

        test_avg_loss, test_all_y_true, test_all_y_pred, _ = full_test_evaluation_per_split(
            best_model_state=best_model_state,
            model=model,
            criterion=criterion,
            test_loader=test_loader,
            test_source=test_source,
            device=device,
            config=config,
            log=False
        )

        avg_loss += test_avg_loss
        all_y_true.extend(test_all_y_true)
        all_y_pred.extend(test_all_y_pred)

    acc = balanced_accuracy_score(y_true=all_y_true, y_pred=all_y_pred)
    loss = avg_loss / len(sources)
    cr = classification_report(
        all_y_true,
        all_y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    result = ({
        "param_name": param_name,
        "param_value": param_value,
        "loss": loss,
        "bal. accuracy": acc,
        "precision_0": cr["0"]["precision"],
        "precision_1": cr["1"]["precision"],
        "recall_0": cr["0"]["recall"],
        "recall_1": cr["1"]["recall"],
        "f1_0": cr["0"]["f1-score"],
        "f1_1": cr["1"]["f1-score"]
    })

    return result