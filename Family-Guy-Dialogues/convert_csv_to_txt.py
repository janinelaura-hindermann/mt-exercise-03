# MT Exercise 3
# 1 Training a recurrent neural network language model - Finding an interesting data set

# Student Names: Luana Cheda, Janine Laura Hindermann

# This program converts a CSV file containing dialogue data from the "Family Guy" TV
# show into a text file. It filters out segments with less than 17 words, removes
# duplicate lines, and writes the unique dialogues to the output text file.
# The conversion process ensures that the resulting text file contains only unique
# dialogues with a minimum length of 17 words. Finally, it prints a message indicating
# the successful completion of the conversion process.


import pandas as pd

def convert_csv_to_txt(csv_file, txt_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter out segments with less than 17 words
    df = df[df["Dialogue"].astype(str).str.split().apply(len) >= 17]
    
    # Extract the "Dialogue" column
    dialogues = df["Dialogue"].astype(str).tolist()  # Convert to string
    
    # Remove duplicate lines
    unique_dialogues = set(dialogues)
    
    # Write the dialogues to a text file
    with open(txt_file, "w") as f:
        for dialogue in unique_dialogues:
            f.write(str(dialogue).strip() + "\n")  # Convert to string before stripping

# Input CSV file and output text file
csv_file = "Family_Guy_Final_NRC_AFINN_BING.csv"
txt_file = "family_guy_dialogues.txt"

# Convert the CSV file to a text file
convert_csv_to_txt(csv_file, txt_file)

print("Conversion completed successfully.")