import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Example call:

python3 plot_perplexity.py --validation_csv ../logs_csv/validation_dataframe.csv --training_csv ../logs_csv/training_dataframe.csv --output_dir ../plots
"""


def plot_perplexity(csv_file_path, title, plot_type, output_dir):
    # Read the CSV file into a dataframe
    df = pd.read_csv(csv_file_path, index_col='Epoch')

    # Plotting all columns against the index (Epoch)
    plt.figure(figsize=(10, 5))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    # Adding title and labels
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a filename for the plot based on the input CSV filename
    output_file_name = f"{plot_type}_perplexity_plot.png"
    output_file_path = os.path.join(output_dir, output_file_name)

    # Save the plot to a file
    plt.savefig(output_file_path)
    print(f"Plot saved to: {output_file_path}")

    # Show the plot
    plt.show()


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Plot perplexity from CSV log files.")
    parser.add_argument('--validation_csv', type=str, required=True,
                        help="CSV file containing validation perplexity data.")
    parser.add_argument('--training_csv', type=str, required=True, help="CSV file containing training perplexity data.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output plots.")
    args = parser.parse_args()

    # Plot perplexity for validation and training data
    plot_perplexity(args.validation_csv, 'Validation Perplexity Over Epochs', 'validation', args.output_dir)
    plot_perplexity(args.training_csv, 'Training Perplexity Over Epochs', 'training', args.output_dir)


if __name__ == "__main__":
    main()
