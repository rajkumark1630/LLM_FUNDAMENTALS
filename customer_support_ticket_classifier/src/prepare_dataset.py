from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

RANDOM_STATE = 42
DATASET_SIZE_PER_COMBINATION = 33

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Define categories, priorities and ticket text
# --------------------------------------------------

TICKET_TEMPLATES = {
    "Technical": {
        "High": [
            "The application crashes whenever I log in",
            "The website is completely unavailable",
            "The system fails during checkout",
            "I cannot access an important application feature",
            "The application displays a critical error",
        ],
        "Medium": [
            "The application loads very slowly",
            "I cannot upload a document",
            "The search feature is not working correctly",
            "Some buttons are not responding",
            "The dashboard displays an unexpected error",
        ],
        "Low": [
            "How do I change the application theme",
            "Please explain how to use the dashboard",
            "I need information about software features",
            "How can I customize my application settings",
            "Where can I find the user guide",
        ],
    },
    "Billing": {
        "High": [
            "I was charged twice for my subscription",
            "An unexpected payment was deducted",
            "My payment failed but money was deducted",
            "I need urgent help with an incorrect charge",
            "I received an incorrect payment confirmation",
        ],
        "Medium": [
            "My invoice contains an incorrect amount",
            "I cannot download my billing statement",
            "Please explain the charges on my invoice",
            "My subscription renewal amount looks incorrect",
            "My payment receipt shows incorrect details",
        ],
        "Low": [
            "Where can I find my previous invoices",
            "How can I update my billing address",
            "What payment methods are available",
            "When will my next invoice be generated",
            "How do I view my subscription charges",
        ],
    },
    "Account": {
        "High": [
            "My account appears to be compromised",
            "I am locked out of my account",
            "Someone changed my account recovery information",
            "I cannot access my account after a security alert",
            "My account access was blocked unexpectedly",
        ],
        "Medium": [
            "I cannot reset my password",
            "My profile information is not updating",
            "I cannot verify my email address",
            "The account settings page displays an error",
            "I am unable to update my account details",
        ],
        "Low": [
            "How can I change my profile picture",
            "I want to update my display name",
            "Where can I manage notification preferences",
            "How do I add an alternative email address",
            "How can I change my profile language",
        ],
    },
    "Delivery": {
        "High": [
            "My urgent order appears to be lost",
            "My package has not arrived and tracking shows an exception",
            "My delivery is missing despite being marked delivered",
            "I need immediate help with a lost shipment",
            "My important shipment cannot be located",
        ],
        "Medium": [
            "My order is delayed beyond the expected date",
            "The tracking page has not updated",
            "My package was delivered to the wrong address",
            "I need an update about my delayed shipment",
            "My order has not moved from the sorting facility",
        ],
        "Low": [
            "How can I track my order",
            "What are the standard delivery times",
            "Can I change my preferred delivery date",
            "Where can I find my tracking number",
            "How do I update my delivery preferences",
        ],
    },
    "Refund": {
        "High": [
            "My approved refund has not arrived",
            "I was incorrectly denied a refund for a cancelled order",
            "I urgently need help with my refund request",
            "My returned item was received but the refund is missing",
            "An incorrect amount was refunded to my account",
        ],
        "Medium": [
            "I would like to request a refund",
            "How long does the refund process take",
            "My refund status has not changed",
            "I returned an item and need a refund update",
            "My refund request is still being processed",
        ],
        "Low": [
            "What is your refund policy",
            "Can you explain refund eligibility requirements",
            "Where can I view my refund history",
            "Which payment methods support refunds",
            "How can I submit a refund request",
        ],
    },
    "General Inquiry": {
        "High": [
            "I need urgent assistance with a critical service issue",
            "Our team cannot access critical support documentation",
            "Please urgently clarify a service interruption notice",
            "I need immediate guidance about a serious service issue",
            "Our business operations are affected by a service problem",
        ],
        "Medium": [
            "I need help understanding the service terms",
            "Can you explain how your subscription plans work",
            "I have a question about the available services",
            "Please clarify a feature in your documentation",
            "I need assistance understanding a service policy",
        ],
        "Low": [
            "What are your business hours",
            "Where can I find the product documentation",
            "Do you offer tutorials for new customers",
            "I need general information about your services",
            "How can I contact customer support",
        ],
    },
}


# --------------------------------------------------
# 3. Generate a synthetic dataset
# --------------------------------------------------

def generate_dataset():
    records = []

    for category, priority_templates in TICKET_TEMPLATES.items():
        for priority, messages in priority_templates.items():

            for index in range(DATASET_SIZE_PER_COMBINATION):
                message = messages[index % len(messages)]

                # Unique ticket reference for every record
                ticket_id = (
                    f"{category[:2].upper()}-"
                    f"{priority[:2].upper()}-{index:03d}"
                )

                records.append({
                    "text": f"{message}. Reference: {ticket_id}",
                    "category": category,
                    "priority": priority,
                })

    return pd.DataFrame(records)


# --------------------------------------------------
# 4. Clean the dataset
# --------------------------------------------------

def clean_dataset(df):
    df = df.copy()

    # Normalize ticket text
    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Clean category and priority labels
    for column in ["category", "priority"]:
        df[column] = df[column].astype("string").str.strip()

    # Remove missing and empty records
    df = df.dropna(
        subset=["text", "category", "priority"]
    )

    df = df[
        df["text"].ne("")
        & df["category"].ne("")
        & df["priority"].ne("")
    ].copy()

    # Remove duplicate ticket records
    df = df.drop_duplicates(
        subset=["text", "category", "priority"]
    )

    # Keep only the required columns
    df = df[["text", "category", "priority"]]

    return df.reset_index(drop=True)


# --------------------------------------------------
# 5. Display class balance
# --------------------------------------------------

def show_class_balance(df):
    print("\nCATEGORY COUNTS")
    print(df["category"].value_counts().sort_index())

    print("\nPRIORITY COUNTS")
    print(df["priority"].value_counts().sort_index())

    print("\nCATEGORY-PRIORITY COMBINATIONS")
    print(pd.crosstab(df["category"], df["priority"]))


# --------------------------------------------------
# 6. Create stratified train/validation/test splits
# --------------------------------------------------

def split_dataset(df):
    # Combined strata preserve both labels' combinations
    strata = (
        df["category"].astype(str)
        + "||"
        + df["priority"].astype(str)
    )

    # Every combination needs enough rows for both splits
    if strata.value_counts().min() < 3:
        raise ValueError(
            "Not enough records in a category-priority combination "
            "for the requested stratified split."
        )

    # 70% training and 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    # Split temporary data into 15% validation and 15% test
    temp_strata = (
        temp_df["category"].astype(str)
        + "||"
        + temp_df["priority"].astype(str)
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_strata,
    )

    return train_df, val_df, test_df


# --------------------------------------------------
# 7. Save CSV files and print summary
# --------------------------------------------------

def save_dataset(df, train_df, val_df, test_df):
    df.to_csv(DATA_DIR / "tickets.csv", index=False)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "validation.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print("\nDATASET SUMMARY")
    print("------------------------------")
    print(f"Cleaned total: {len(df)}")
    print(f"Train:         {len(train_df)}")
    print(f"Validation:    {len(val_df)}")
    print(f"Test:          {len(test_df)}")

    combined = pd.concat([
        train_df.assign(split="Train"),
        val_df.assign(split="Validation"),
        test_df.assign(split="Test"),
    ], ignore_index=True)

    print("\nCATEGORY DISTRIBUTION BY SPLIT")
    print(pd.crosstab(combined["split"], combined["category"]))

    print("\nPRIORITY DISTRIBUTION BY SPLIT")
    print(pd.crosstab(combined["split"], combined["priority"]))

    print("\nSaved files:")
    for filename in [
        "tickets.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
    ]:
        print(DATA_DIR / filename)


# --------------------------------------------------
# 8. Main program
# --------------------------------------------------

def main():
    print("Generating synthetic support tickets...")

    raw_df = generate_dataset()
    df = clean_dataset(raw_df)

    if len(df) < 500:
        raise ValueError("The cleaned dataset must contain at least 500 tickets.")

    print(f"\nGenerated rows: {len(raw_df)}")
    print(f"Cleaned rows:   {len(df)}")

    show_class_balance(df)

    train_df, val_df, test_df = split_dataset(df)

    save_dataset(df, train_df, val_df, test_df)

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()