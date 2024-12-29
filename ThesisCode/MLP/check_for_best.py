import json
import os

if __name__ == "__main__":

    old_path = os.path.join(os.path.dirname(__file__), "best.json")
    new_path = os.path.join(os.path.dirname(__file__), "outputtest.json")

    with open(old_path, "r") as f:
        old_data = json.load(f)
    
    with open(new_path, "r") as f:
        new_data = json.load(f)

    if new_data["fitness"] > old_data["fitness"]:
        with open(old_path, "w") as f:
            json.dump(new_data, f)
        print("New best fitness achieved.")
    else:
        print("New contestant not improving on fitness.")