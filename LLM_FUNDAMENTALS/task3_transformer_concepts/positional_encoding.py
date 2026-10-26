
import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(max_positions=10, d_model=8):
    position = np.arange(max_positions)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]

    angles = position / np.power(
        10000,
        (2 * (dimensions // 2)) / d_model
    )

    encoding = np.zeros((max_positions, d_model))

    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return encoding


if __name__ == "__main__":
    encoding = positional_encoding(10, 8)

    print("Positional encoding shape:", encoding.shape)
    print("\nPositional encoding matrix:")
    print(np.round(encoding, 4))

    plt.figure(figsize=(10, 5))
    plt.imshow(encoding, cmap="viridis", aspect="auto")
    plt.colorbar(label="Encoding value")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Token Position")
    plt.title("Sinusoidal Positional Encoding (10 × 8)")
    plt.xticks(range(8))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.savefig(
        "task3_transformer_concepts/positional_encoding.png",
        dpi=150
    )
    plt.show()

    print("\nGraph saved to positional_encoding.png")
