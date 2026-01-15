import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Settings
SEED = 42
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

def main():
    # Create output folder
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load CSV
    df = pd.read_csv(RAW_PATH)

    # Basic cleaning
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_SIZE),
        random_state=SEED,
        stratify=df["label"]
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)),
        random_state=SEED,
        stratify=temp_df["label"]
    )


    # Print summary
    print("✅ Dataset split completed")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    main()
