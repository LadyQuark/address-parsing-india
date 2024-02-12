import argparse
import os
import pandas as pd
import spacy
import time

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from spacy import displacy
from sys import exit

from training_data_prep import clean_data

# Get ENV variables
load_dotenv(find_dotenv())
MODELS = os.getenv('MODELS', "models/model-best")

# Load model
nlp = spacy.load(MODELS)

# Constants
TAGS = [
    "ADDRESS_LINE_1",
    "ADDRESS_LINE_2",
    "CITY",
    "DISTRICT",
    "STATE",
    "PINCODE",
]
COLUMNS = [
    "ADDRESS",
    "ADDRESS_LINE_1",
    "ADDRESS_LINE_2",
    "CITY",
    "DISTRICT",
    "STATE",
    "PINCODE",
]


def extract_address_components(address):
    """
    Extract components of address using Named Entity Recognition + Rule-based Matching
    """
    cleansed_address = clean_data(address)
    doc = nlp(cleansed_address)
    displacy.render(doc, style="ent")

    results = {tag: "" for tag in TAGS}
    results["ADDRESS"] = cleansed_address
    for ent in doc.ents:
        results[ent.label_] += ent.text if results[ent.label_] == "" else " " + ent.text
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()

    # Read Excel into Pandas Data Frame
    try:
        og_filepath = Path(args.input)
        original = pd.read_excel(args.input, usecols=[0],  dtype=str)
    except FileNotFoundError as e:
        exit(e)

    # Predict
    results = []
    for address in original['Address']:
        row = extract_address_components(address)
        results.append(row)
    
    # Write to Excel
    df = pd.DataFrame.from_dict(results)
    df = df.reindex(columns=COLUMNS)
    
    timestamp = str(int(time.time()))
    output_filepath = og_filepath.with_name(og_filepath.stem + " " + timestamp).with_suffix(".xlsx")
    df.to_excel(output_filepath, index=False)
    print("File saved to:", output_filepath)



if __name__ == "__main__":
    main()