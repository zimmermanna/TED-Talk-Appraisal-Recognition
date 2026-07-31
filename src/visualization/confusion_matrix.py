import matplotlib.pyplot as plt
import seaborn as sns

def heatmap_cm(confusion_matrix):
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix,annot=True,fmt='d',cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")