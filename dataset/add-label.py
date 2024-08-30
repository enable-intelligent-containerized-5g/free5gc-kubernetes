import pandas as pd

def analyze_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Ensure the necessary columns exist
    if 'cpu-usage' not in df.columns or 'cpu-limit' not in df.columns:
        raise ValueError("The CSV file must contain 'cpu-usage' and 'cpu-limit' columns.")

    # Calculate the label column
    df['label'] = df.apply(lambda row: 1 if row['cpu-limit'] < row['cpu-request'] else 0, axis=1)

    print(df["label"].sum())

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Example usage
input_file = 'dataset.csv'  # Replace with the path to your input CSV file
output_file = 'output.csv'  # Replace with the desired path for the output CSV file

analyze_csv(input_file, output_file)