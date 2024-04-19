import argparse
import os

"""
Example usage:

python3 clean_log_folder.py ../logs ../logs_clean

"""


def process_log_file(input_file_path, output_base):
    # Ensure the base output directory exists
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    # Specific directories for train, valid, and test
    train_dir = os.path.join(output_base, 'training')
    valid_dir = os.path.join(output_base, 'validation')
    test_dir = os.path.join(output_base, 'test')

    # Create directories if they don't exist
    for dir_path in [train_dir, valid_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Prepare output file names based on the input file name
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    train_output_path = os.path.join(train_dir, f"{base_name}_clean_train.txt")
    valid_output_path = os.path.join(valid_dir, f"{base_name}_clean_valid.txt")
    test_output_path = os.path.join(test_dir, f"{base_name}_clean_test.txt")

    # Dictionaries and lists to hold the necessary data
    last_train_per_epoch = {}
    validation_data = []
    test_data = []

    # Read the input file
    with open(input_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                epoch, data_type, perplexity = int(parts[0]), parts[1], float(parts[2])
                if data_type == 'Train':
                    last_train_per_epoch[epoch] = perplexity
                elif data_type == 'Validation':
                    validation_data.append((epoch, perplexity))
                elif data_type == 'Test':
                    test_data.append(perplexity)

    # Write the last train entry for each epoch to a file
    with open(train_output_path, 'w') as f:
        for epoch in sorted(last_train_per_epoch):
            f.write(f"{epoch}\t{last_train_per_epoch[epoch]}\n")

    # Write all validation data to a file
    with open(valid_output_path, 'w') as f:
        for epoch, perplexity in validation_data:
            f.write(f"{epoch}\t{perplexity}\n")

    max_epoch = max(validation_data, default=(0, 0))[0]  # default to 0 if no validation data

    # Write the test data to a file
    with open(test_output_path, 'w') as f:
        for perplexity in test_data:
            f.write(f"{max_epoch}\t{perplexity}\n")


def process_directory(directory_path, output_base):
    # List all files in the provided directory
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            print(f"Processing file: {file_path}")
            process_log_file(file_path, output_base)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process all log files in a directory and output to specified base folder.')
    parser.add_argument('input_directory', type=str,
                        help='The path to the directory containing log files.')
    parser.add_argument('output_directory', type=str,
                        help='The path to the base output directory where processed files will be stored.')
    args = parser.parse_args()

    # Process the directory using the provided path
    process_directory(args.input_directory, args.output_directory)


if __name__ == "__main__":
    main()
