import sys

import numpy as np

if __name__ == '__main__':
    # Reading input from stdin (this can be from CSV or binary)
    if not sys.stdin.isatty():  # Ensure there's input being piped to the script
        input_data = sys.stdin.read().decode('utf-8')

        # Parse the input into a NumPy array
        # First, we try reading it as a CSV (if in text format)
        try:
            # Use np.genfromtxt to read the data as CSV
            input_array = np.genfromtxt(input_data.splitlines(), delimiter=",", dtype=int)
        except Exception as e:
            print(f"Error reading CSV format: {e}")
            sys.exit(1)

        # Print the processed array (for the next script to read)
        np.savetxt(sys.stdout, input_array, delimiter=",", fmt="%d")
    else:
        print("No input data received.")
        sys.exit(1)