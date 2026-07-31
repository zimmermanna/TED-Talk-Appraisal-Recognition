import pandas as pd

from sklearn.metrics import confusion_matrix, log_loss, balanced_accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from src.utils.data import sources, au_cols, au_cols_adv
from src.visualization.confusion_matrix import heatmap_cm


def LOSO_Logistic_Regression(df, au_cols):
    all_y_pred = []
    all_y_true = []
    all_y_prob = []
    source_results = []

    for test_source in sources:
        df_train = df[df["source_id"] != test_source]
        df_test = df[df["source_id"] == test_source]

        X_train = df_train[au_cols]
        y_train = df_train["label"]

        X_test = df_test[au_cols]
        y_test = df_test["label"]

        model = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Saving Results for Single Sources
        source_loss = log_loss(y_test, y_prob, labels=[0, 1])

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        source_results.append({
            "Source": test_source,
            "Accuracy": (tn + tp) / (tn + fp + fn + tp),
            "Loss": source_loss,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        })

        # Saving collected y_pred and y_true
        all_y_pred.extend(y_pred)
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

    # Printing Overview at the End
    print(pd.DataFrame(source_results))
    print(confusion_matrix(all_y_true, all_y_pred))

    return all_y_pred, all_y_true, all_y_prob

def LogReg_single_signal(df):
    single_au_results = []
    for au in au_cols:
        print("=" * 40)
        print("ACTION UNIT: ", au)
        print("=" * 40)

        all_y_pred, all_y_true, all_y_prob = LOSO_Logistic_Regression(df, [au])

        acc_bal = balanced_accuracy_score(all_y_true, all_y_pred)
        overall_loss = log_loss(all_y_true, all_y_prob, labels=[0, 1])
        cr = classification_report(
            all_y_true,
            all_y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )

        single_au_results.append({
            "Action Unit": au,
            "Accuracy (Balanced)": acc_bal,
            "Loss": overall_loss,
            "precision_0": cr["0"]["precision"],
            "precision_1": cr["1"]["precision"],
            "recall_0": cr["0"]["recall"],
            "recall_1": cr["1"]["recall"],
            "f1_0": cr["0"]["f1-score"],
            "f1_1": cr["1"]["f1-score"]
        })

    summary = pd.DataFrame(single_au_results)
    summary["macro_f1"] = (summary["f1_0"] + summary["f1_1"]) / 2
    summary = summary.sort_values("Accuracy (Balanced)", ascending=False).round(3)

    return summary

def LogReg_all_signals(df, results, aggregation="Simple", normalized=False):
    if (aggregation == "Advanced"):
        all_y_pred, all_y_true, all_y_prob = LOSO_Logistic_Regression(df, au_cols_adv)
    else:
        all_y_pred, all_y_true, all_y_prob = LOSO_Logistic_Regression(df, au_cols)

    acc_bal = balanced_accuracy_score(all_y_true, all_y_pred)
    overall_loss = log_loss(all_y_true, all_y_prob, labels=[0, 1])
    cr = classification_report(
        all_y_true,
        all_y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    results.append({
        "Aggregation": aggregation,
        "Normalized": normalized,
        "Accuracy (Balanced)": acc_bal,
        "Loss": overall_loss,
        "precision_0": cr["0"]["precision"],
        "precision_1": cr["1"]["precision"],
        "recall_0": cr["0"]["recall"],
        "recall_1": cr["1"]["recall"],
        "f1_0": cr["0"]["f1-score"],
        "f1_1": cr["1"]["f1-score"]
    })

    heatmap_cm(confusion_matrix(all_y_true, all_y_pred))

    return results