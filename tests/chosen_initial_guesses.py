import json
import os

def save_chosen_initial_guesses():
    # Define the initial guesses for each ion velocity weight
    initial_guesses = {
        '0.1': {'v1': 0.006738, 'v2': 0.017240},  # For weight 0.1
        '1.0': {'v1': 0.006738, 'v2': 0.017240},  # For weight 1.0
        '2.0': {'v1': 0.006738, 'v2': 0.017240},  # For weight 2.0
        '3.0': {'v1': 0.006738, 'v2': 0.017240},  # For weight 3.0
        '5.0': {'v1': 0.006738, 'v2': 0.017240},  # For weight 5.0
        '10.0': {'v1': 0.006738, 'v2': 0.017240}  # For weight 10.0
    }

    # Define the path to save the JSON file
    save_path = os.path.join("..", "results-L-BFGS-B", "chosen_initial_guesses.json")

    # Save the initial guesses to a JSON file
    with open(save_path, 'w') as json_file:
        json.dump(initial_guesses, json_file, indent=4)
    
    print(f"Initial guesses saved to {save_path}")

# Run the function to save the initial guesses
save_chosen_initial_guesses()
