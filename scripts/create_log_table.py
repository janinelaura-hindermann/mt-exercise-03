import argparse
import pandas as pd
import os
import re

"""
Example call:

python3 create_log_table.py --test_dir ../logs_clean/test --valid_dir ../logs_clean/validation --train_dir ../logs_clean/training --output_dir ../logs_csv
"""


def create_dataframe_from_logs(logs_directory):
    # Initialize a dictionary to hold all the data
    data = {}

    # Iterate through all files in the directory
    for file_name in os.listdir(logs_directory):
        # Extract the dropout rate from the file name using regular expression
        match = re.search(r'perplexities_(\d+\.\d+)_clean_', file_name)
        if match:
            dropout_rate = 'Dropout ' + match.group(1)  # Label for column
            # Read the file into a dataframe using tab (\t) as the separator
            file_path = os.path.join(logs_directory, file_name)
            df = pd.read_csv(file_path, sep="\t", engine='python', header=None, names=['Epoch', dropout_rate])
            # Pivot the dataframe for the current dropout rate
            df.set_index('Epoch', inplace=True)
            data[dropout_rate] = df[dropout_rate]

    # Combine all dataframes on the index (Epoch)
    combined_df = pd.concat(data.values(), axis=1, keys=data.keys())

    # Sort the columns by dropout rate for display purposes
    combined_df.sort_index(axis=1, inplace=True)

    return combined_df


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate dataframes from cleaned log files.")
    parser.add_argument('--test_dir', type=str, required=True, help="Directory containing cleaned test log files.")
    parser.add_argument('--valid_dir', type=str, required=True,
                        help="Directory containing cleaned validation log files.")
    parser.add_argument('--train_dir', type=str, required=True, help="Directory containing cleaned training log files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store the output CSV files.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create dataframes for each category
    df_test = create_dataframe_from_logs(args.test_dir)
    df_validation = create_dataframe_from_logs(args.valid_dir)
    df_training = create_dataframe_from_logs(args.train_dir)

    # Save each dataframe to a CSV file
    df_test.to_csv(os.path.join(args.output_dir, 'test_dataframe.csv'))
    df_validation.to_csv(os.path.join(args.output_dir, 'validation_dataframe.csv'))
    df_training.to_csv(os.path.join(args.output_dir, 'training_dataframe.csv'))

    # Print the paths to the saved CSV files
    print(f"Test Dataframe saved to: {os.path.join(args.output_dir, 'test_dataframe.csv')}")
    print(f"Validation Dataframe saved to: {os.path.join(args.output_dir, 'validation_dataframe.csv')}")
    print(f"Training Dataframe saved to: {os.path.join(args.output_dir, 'training_dataframe.csv')}")


if __name__ == "__main__":
    main()
