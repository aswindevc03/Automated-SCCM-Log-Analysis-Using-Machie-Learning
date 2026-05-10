import re
import pickle


def extract_latest_error(log_path):
    with open(log_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in reversed(lines):
        if "error" in line.lower() or "failed" in line.lower() or "0x" in line.lower():
            match = re.search(r"0x[0-9a-fA-F]+", line)
            error_code = match.group() if match else "NoCode"
            return error_code, line.strip()

    return None, "No error found"


# Load ML model
model = pickle.load(open("sccm_model.pkl", "rb"))

# Predict
log_file = "sample_sccm_error.log"

code, logline = extract_latest_error(log_file)
print("\n Log Error Found:")
print("Line:", logline)
print("Error Code:", code)

# ML prediction
text = code + " " + logline
prediction = model.predict([text])[0]

print("\nPrediction:")
print("Category:", prediction)
