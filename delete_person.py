import pickle

ENCODINGS_PATH = "encodings.pickle"
person_to_delete = "adhyashree"

with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

new_encodings = []
new_names = []

for enc, name in zip(data["encodings"], data["names"]):
    if name != person_to_delete:
        new_encodings.append(enc)
        new_names.append(name)

with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump({"encodings": new_encodings, "names": new_names}, f)

print("Removed:", person_to_delete)
