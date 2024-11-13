import json

with open("../results-Nelder-Mead/best_initial_guess_w_2_0.json", 'r') as f:
    data = json.load(f)
print(data)
