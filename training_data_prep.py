# Modified from https://github.com/swapnil-saxena/address-parser/blob/main/training_data_prep.py

import argparse
import os
import pandas as pd
import re
import spacy

from collections import OrderedDict
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from spacy.tokens import DocBin

pd.set_option('display.max_colwidth', None)

# Define custom entity tag list
tag_info = OrderedDict({
    "AddressLine1Tag":  { "column_name": "Address Line 1",   "label": "ADDRESS_LINE_1"  },
    "AddressLine2Tag":  { "column_name": "Address Line 2",   "label": "ADDRESS_LINE_2"  },
    "StateTag":         { "column_name": "State",            "label": "STATE"           },
    "DistrictTag":      { "column_name": "District",         "label": "DISTRICT"        },
    "CityTag":          { "column_name": "City",             "label": "CITY"            },
    "PincodeTag":       { "column_name": "Pincode",          "label": "PINCODE"         },
})

# Get ENV variables
load_dotenv(find_dotenv())
TRAINING_CSV = os.getenv("TRAINING_CSV", "corpus/dataset/training_data.csv")
TEST_CSV     = os.getenv("TEST_CSV", "corpus/dataset/training_data.csv")
DOCBINS      = os.getenv("DOCBINS", "corpus/spacy_docbins")

# Regexes
re_commas   = re.compile(r'(,)(?!\s)')
re_newlines = re.compile(r'(,*\s*\n\s*)')
re_hyphen   = re.compile(r'(\s*-\s*)')
re_colon    = re.compile(r'(\s*:\s*)')
re_po       = re.compile(r'\b(P\.O\.?\s*)[\W]')
re_ps       = re.compile(r'\b(P\.S\.?\s*)[\W]')
re_end      = re.compile(r'(\s*,*\s*)$')


def clean_data(address, verbose=False):
    """
    Pre-process address string to remove new line characters, fix spaces between
    commas, hyphens and colons. Also standardizes appearance of common abbreviations
    like P.O. (post office) and P.S. (police station) in Indian addresses
    """

    cleansed_address = address
    cleansed_address = re_po.sub('P.O. ', cleansed_address)
    cleansed_address = re_ps.sub('P.S. ', cleansed_address)
    cleansed_address = re_commas.sub(', ', cleansed_address)
    cleansed_address = re_newlines.sub(', ', cleansed_address, re.M)
    cleansed_address = re_hyphen.sub(' - ', cleansed_address)
    cleansed_address = re_colon.sub(' : ', cleansed_address)
    cleansed_address = re_end.sub('', cleansed_address)

    if verbose and cleansed_address != address:
        print("ORIGINAL:", address)
        print("CLEANSED:", cleansed_address)

    return cleansed_address

def reverse_interval_generator(matches):
    """
    Takes a list of Matches and yields the right-most Match in the form of 
    an Interval object denoting the start & end index of the substring
    """
    for match in reversed(matches):
        yield pd.Interval(match.start(), match.end())

def get_cols_to_check(label):
    """
    Get all column names containing entity spans on the left of given tag
    """
    check_cols = []
    for key in tag_info:
        if tag_info[key]["label"] == label:
            break
        check_cols.append(key)
    return check_cols

def get_address_span(row, address_component=None, label=None):
    """
    Search for specified address component and get a non-overlapping span.
    Eg: get_address_span(row=row, address="111 MG Road, Navi Mumbai, Mumbai",address_component="Mumbai", label="CITY") would return (26, 32, "CITY")
    Checks that the span does not overlap with entity span values in other columns
    """

    if pd.isna(address_component) or str(address_component) == 'nan':
        # print("Found None")
        return None
    
    # Find all Matches of `address component` in `address`
    address = row['Address']
    pattern = r'\b(' + re.escape(address_component) + r')(?![a-zA-Z])'
    match_iter = re.finditer(pattern, address)
    matches = [match for match in match_iter if match != None]    
    if not matches:
        print(f"Could not find {address_component} in {address}")
        return None
    
    # Get all entity span columns to the right
    check_cols = get_cols_to_check(label)
    intervals = reverse_interval_generator(matches)

    # Check that selected span does not overlap with entity span values in other columns
    try:
        interval = next(intervals) 
        for col in check_cols:
            if col in row and row[col] != None:
                compare_interval = pd.Interval(row[col][0], row[col][1])
                if interval.overlaps(compare_interval):
                    interval = next(intervals)
                    print(f"Changed INTERVAL to: {interval}")       
    
    except StopIteration:
        print(f"Reached end of matches for {label}: \n\t{address_component}\n\tin {address}")
        print(matches)
        return None
    
    return (interval.left, interval.right, label)


def extend_list(entity_list, entity):
    if pd.isna(entity):
        return entity_list
    else:
        entity_list.append(entity)
        return entity_list

def create_entity_spans(df, verbose=False):
    """Create entity spans for training/test datasets"""

    df['Address']        = df['Address'].apply(lambda x: clean_data(x, verbose))
    df['Address Line 1'] = df['Address Line 1'].apply(lambda x: clean_data(x, verbose))
    df['Address Line 2'] = df['Address Line 2'].apply(lambda x: clean_data(x, verbose))

    for tag in tag_info:
        column_name = tag_info[tag]["column_name"]
        label = tag_info[tag]["label"]

        df[tag] = df.apply(
            lambda row:get_address_span(
                row, address_component=row[column_name], label=label),
            axis=1)
        
    df['EmptySpan'] = df.apply(lambda x: [], axis=1)
    df = df[df.notnull()]

    for tag in tag_info:
        df['EntitySpans'] = df.apply(
            lambda row: extend_list(row['EmptySpan'], row[tag]), 
            axis=1)
    df['EntitySpans'] = df[['EntitySpans','Address']].apply(
        lambda x: (x.iloc[1], x.iloc[0]),
        axis=1)
    
    return df['EntitySpans']

def get_doc_bin(training_data, nlp, verbose=False):
    """Create DocBin object for building training/test corpus"""

    db = DocBin()
    for text, annotations in training_data:
        if verbose:
            print("TEXT:", text)
            print("ANNOTATIONS:", annotations)
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        if verbose: print(ents)
        doc.ents = ents
        db.add(doc)
    return db

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-r", "--train", default=TRAINING_CSV, 
                        help="File path to CSV file containing training data")
    parser.add_argument("-t", "--test", default="", 
                        help="File path to CSV file containing testing data")
    parser.add_argument("-o", "--output", default=DOCBINS,
                        help="Path to folder to store generated docbins")
    args = parser.parse_args()

    # Load blank English model
    nlp = spacy.blank("en")
    
    ###### Training/Validation dataset prep ###########

    # Read training dataset, shuffle rows, strip all strings
    df_train = pd.read_csv(filepath_or_buffer=args.train, sep=",", dtype=str)
    df_train = df_train.sample(frac=1).apply(lambda x: x.str.strip())
    print("Size of dataset:", df_train.shape[0])

    if args.test:
        # Read validation dataset, shuffle rows, strip all strings
        df_test = pd.read_csv(filepath_or_buffer=args.test, sep=",", dtype=str)
        df_test = df_test.sample(frac=1).apply(lambda x: x.str.strip())
    else:
        # If no validation dataset provided, split training data by 70:10 ratio
        ratio = 0.7
        train_size = round(df_train.shape[0] * ratio)
        df_test = df_train[train_size:]
        df_train = df_train[0:train_size]

    print("Training size:", df_train.shape[0])
    print("Testing size:", df_test.shape[0])

    # Get entity spans
    df_entity_spans = create_entity_spans(df_train.astype(str), verbose=args.verbose)
    training_data = df_entity_spans.values.tolist()

    # Create & Write DocBin to disk
    doc_bin_train = get_doc_bin(training_data, nlp, verbose=args.verbose)
    train_path = Path(args.output) / Path("train.spacy")
    doc_bin_train.to_disk(train_path)
    print("Docbin files saved to:", train_path)
    
    # Get entity spans
    df_entity_spans = create_entity_spans(df_test.astype(str))
    validation_data = df_entity_spans.values.tolist()

    # Create & Write DocBin to disk
    doc_bin_test = get_doc_bin(validation_data, nlp)
    test_path = Path(args.output) / Path("test.spacy")
    doc_bin_test.to_disk(test_path)
    print("Docbin files saved to:", test_path)


if __name__ == "__main__":
    main()