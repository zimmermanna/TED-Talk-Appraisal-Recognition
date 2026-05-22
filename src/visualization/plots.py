import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(
    style="whitegrid",
    context="paper",
    palette="tab20c",
    font="serif",
    rc={
        "figure.figsize": (9, 5),

        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",

        "axes.edgecolor": "black",
        "axes.linewidth": 0.4,

        "grid.color": "#d0d0d0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.3,

        "axes.titlesize": 14,
        "axes.labelsize": 10,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "legend.frameon": False
    }
)

def boxplot_scatter(df_agg, au, description):
    plt.figure(figsize = (7,5))

    sns.boxplot(
        x = "label",
        y = au + "_r",
        data = df_agg,
        width = 0.5,
        showcaps = True,
        boxprops = {"alpha": 0.8},
        color = "aliceblue",
        showfliers=False
    )

    sns.stripplot(
        x = "label",
        y = au + "_r",
        hue = "source_id",
        data = df_agg,
        jitter = 0.2,
        dodge = True,
        size = 6,
        alpha = 1,
        linewidth = 0.5
    )

    plt.title(f"{au} - {description}", pad=15)
    plt.xlabel("Label")
    plt.ylabel(f"Mean {au} Intensity")

    plt.legend(
        title = "Source Video",
        bbox_to_anchor = (1.02, 1),
        loc = "upper left",
        borderaxespad = 0
    )

    plt.tight_layout()
    plt.show()

def label_distribution(df_agg):
    plt.figure(figsize = (7,5))

    sns.countplot(
        data=df_agg,
        x="label",
        hue="source_id",
    )

    plt.title("Label Distribution by Source ID")
    plt.xlabel("Label")
    plt.ylabel("Count")

    plt.legend(
        title="Source ID",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )