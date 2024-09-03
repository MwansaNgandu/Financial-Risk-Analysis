import os

# Get the current working directory
current_directory = os.getcwd()

# List all CSV files in the current working directory
csv_files = [file for file in os.listdir(current_directory) if file.endswith('.csv')]

# Print the CSV file names
for file in csv_files:
    print(file)


