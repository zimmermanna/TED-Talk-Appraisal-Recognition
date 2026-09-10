import numpy as np
import matplotlib.pyplot as plt


def plot_attention_heatmap_with_two_au(example):
    attention = np.asarray(
        example["attention"].iloc[0],
        dtype=float
    )

    X = np.asarray(
        example["X"].iloc[0],
        dtype=float
    )

    au02 = X[:, 0]
    au05 = X[:, 1]
    # au12 = X[:, 2]

    # Darstellung beginnt bei Window 1
    windows = np.arange(1, len(attention) + 1)

    if len(attention) != len(au02) or len(attention) != len(au05):
        raise ValueError(
            f"Attention has {len(attention)} windows, "
            f"AU02 has {len(au02)} values and "
            f"AU05 has {len(au05)} values."
        )

    clip_id = example["clip_id"].iloc[0]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 4),
        sharex=True,
        gridspec_kw={
            "height_ratios": [0.35, 3],
            "hspace": 0.08
        }
    )

    # Attention-Heatmap
    heatmap = axes[0].imshow(
        attention.reshape(1, -1),
        aspect="auto",
        cmap="Blues",
        interpolation="nearest",
        extent=[
            0.5,
            len(attention) + 0.5,
            0,
            1
        ]
    )

    axes[0].set_yticks([])
    axes[0].set_ylabel("Attention")
    axes[0].set_title(
        f"Attention and Action Unit Activity – Clip {clip_id}"
    )

    axes[1].plot(
        windows,
        au02,
        marker="o",
        markersize=3,
        linewidth=1.5,
        color="dimgray",
        label="AU02_r"
    )

    axes[1].plot(
        windows,
        au05,
        marker="o",
        markersize=3,
        linewidth=1.5,
        color="darkgray",
        label="AU12_r"
    )

    axes[1].set_xlabel("Window")
    axes[1].set_ylabel("Normalized AU Value")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.colorbar(
        heatmap,
        ax=axes,
        label="Attention Weight",
        fraction=0.025,
        pad=0.02
    )

    plt.show()

def plot_attention_heatmap_with_all_aus(example, au_names):
    attention = np.asarray(
        example["attention"].iloc[0],
        dtype=float
    )

    X = np.asarray(
        example["X"].iloc[0],
        dtype=float
    )

    clip_id = example["clip_id"].iloc[0]
    windows = np.arange(len(attention))

    if X.shape[0] != len(attention):
        raise ValueError(
            f"Attention has {len(attention)} windows, "
            f"but X has {X.shape[0]} windows."
        )

    if X.shape[1] != len(au_names):
        raise ValueError(
            f"X contains {X.shape[1]} features, "
            f"but {len(au_names)} AU names were provided."
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 4]
        }
    )

    # Attention-Heatmap
    heatmap = axes[0].imshow(
        attention.reshape(1, -1),
        aspect="auto",
        cmap="Reds",
        interpolation="nearest",
        extent=[
            -0.5,
            len(attention) - 0.5,
            0,
            1
        ]
    )

    axes[0].set_yticks([])
    axes[0].set_ylabel("Attention")
    axes[0].set_title(
        f"Attention and Action Unit activity – Clip {clip_id}"
    )

    # Alle AUs plotten
    for feature_index, au_name in enumerate(au_names):
        axes[1].plot(
            windows,
            X[:, feature_index],
            label=au_name,
            linewidth=1.5
        )

    axes[1].set_xlabel("Window Index")
    axes[1].set_ylabel("Normalized AU value")
    axes[1].grid(alpha=0.3)

    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4
    )

    fig.colorbar(
        heatmap,
        ax=axes,
        label="Attention Weight",
        fraction=0.025,
        pad=0.02
    )

    plt.subplots_adjust(bottom=0.23)
    plt.show()