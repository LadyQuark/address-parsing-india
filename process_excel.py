import argparse
import pandas as pd
import spacy
import time

from pathlib import Path
from sys import exit

from training_data_prep import massage_data

nlp = spacy.load("output/models/model-best")
TAGS = [
    "ADDRESS_LINE_1",
    "ADDRESS_LINE_2",
    "CITY",
    "DISTRICT",
    "PINCODE",
]
COLUMNS = [
    "ADDRESS",
    "ADDRESS_LINE_1",
    "ADDRESS_LINE_2",
    "CITY",
    "DISTRICT",
    "PINCODE",
]


def extract_address(address):
    cleansed_address = massage_data(address)
    doc = nlp(cleansed_address)
    ent_list = [(ent.text, ent.label_) for ent in doc.ents]
    # print("Address string -> ", address)
    # print("Parsed address -> ", str(ent_list))
    # print("******")

    results = {tag: "" for tag in TAGS}
    results["ADDRESS"] = cleansed_address
    for ent in doc.ents:
        results[ent.label_] += ent.text if results[ent.label_] == "" else " " + ent.text
    # print(results)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()

    try:
        og_filepath = Path(args.input)
        original = pd.read_excel(args.input, usecols=[0],  dtype=str)
    except FileNotFoundError as e:
        exit(e)

    results = []
    for address in original['Address']:
        row = extract_address(address)
        results.append(row)
    
    df = pd.DataFrame.from_dict(results)
    df = df.reindex(columns=COLUMNS)
    timestamp = str(int(time.time()))
    output_filepath = og_filepath.with_name(og_filepath.stem + " " + timestamp).with_suffix(".xlsx")
    df.to_excel(output_filepath, index=False)
    print("File saved to:", output_filepath)



if __name__ == "__main__":
    main()