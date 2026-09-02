import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def heatmap_cm(confusion_matrix):
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix,annot=True,fmt='d',cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

def heatmap_corr(correlation_matrix):
    custom_cmap_bo = LinearSegmentedColormap.from_list(
        "blue_orange",
        ["#E67E22", "#F7F3ED", "#3F6F9F"]
    )

    plt.figure(figsize=(8,6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap=custom_cmap_bo,
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Matrix of Facial Action Units")
    plt.tight_layout()
    plt.show()