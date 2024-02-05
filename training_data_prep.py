# Modified from https://github.com/swapnil-saxena/address-parser/blob/main/training_data_prep.py

import argparse
import pandas as pd
import re
import spacy

from spacy.tokens import DocBin

# Define custom entity tag list
tag_info = {
    "AddressLine1Tag":  { "column_name": "Address Line 1",   "label": "ADDRESS_LINE_1"  },
    "AddressLine2Tag":  { "column_name": "Address Line 2",   "label": "ADDRESS_LINE_2"  },
    "CityTag":          { "column_name": "City",             "label": "CITY"            },
    "DistrictTag":      { "column_name": "District",         "label": "DISTRICT"        },
    "PincodeTag":       { "column_name": "Pincode",          "label": "PINCODE"         },
}

# Regexes
re_commas   = re.compile(r'(,)(?!\s)')
re_newlines = re.compile(r'(,*\s*\n\s*)')
re_hyphen   = re.compile(r'(\s*-\s*)')
re_po       = re.compile(r'\b(P\.O\.?\s*)[\W]')
re_ps       = re.compile(r'\b(P\.S\.?\s*)[\W]')
re_end      = re.compile(r'(\s*,*\s*)$')
# re_period   = re.compile(r'\s*\.\s*')


def massage_data(address):
    '''Pre process address string to remove new line characters, add comma punctuations etc.'''

    cleansed_address = address
    cleansed_address = re_po.sub('P.O. ', cleansed_address)
    cleansed_address = re_ps.sub('P.S. ', cleansed_address)
    cleansed_address = re_commas.sub(', ', cleansed_address)
    cleansed_address = re_newlines.sub(', ', cleansed_address, re.M)
    cleansed_address = re_hyphen.sub(' - ', cleansed_address)
    cleansed_address = re_end.sub('', cleansed_address)
    # if cleansed_address != address:
    #     print("ORIGINAL:", address)
    #     print("CLEANSED:", cleansed_address)
    return cleansed_address

def get_address_span(address=None, address_component=None,label=None):
    '''Search for specified address component and get the span.
    Eg: get_address_span(address="221 B, Baker Street, London",address_component="221",label="BUILDING_NO") would return (0,2,"BUILDING_NO")'''
    
    if pd.isna(address_component) or str(address_component) == 'nan':
        pass
    else:
        # address_component1 = re.sub(r'\.','',address_component)
        # address_component2 = re.sub(r'(?!\s)(-)(?!\s)',' - ',address_component1)
        # print(label.upper(), address_component2, address)
        span = re.search(r'\b(?:' + re.escape(address_component) +r')\b', address)
        if not span:
            return None
        # print(span)
        return (span.start(),span.end(),label)

def extend_list(entity_list, entity):
    if pd.isna(entity):
        return entity_list
    else:
        entity_list.append(entity)
        return entity_list

def create_entity_spans(df):
    '''Create entity spans for training/test datasets'''

    df['Address']        = df['Address'].apply(lambda x: massage_data(x))
    df['Address Line 1'] = df['Address Line 1'].apply(lambda x: massage_data(x))
    df['Address Line 2'] = df['Address Line 2'].apply(lambda x: massage_data(x))

    for tag in tag_info:
        column_name = tag_info[tag]["column_name"]
        label = tag_info[tag]["label"]

        df[tag] = df.apply(
            lambda row:get_address_span(
                address=row['Address'], address_component=row[column_name], label=label),
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
    '''Create DocBin object for building training/test corpus'''
    # the DocBin will store the example documents
    db = DocBin()
    for text, annotations in training_data:
        if verbose:
            print("TEXT:", text)
            print("ANNOTATIONS:", annotations)
        doc = nlp(text) #Construct a Doc object
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
    args = parser.parse_args()


    # Load blank English model. This is needed for initializing a Document object for our training/test set.
    nlp = spacy.blank("en")
    
    ###### Training dataset prep ###########
    # Read the training dataset into pandas
    df_train = pd.read_csv(filepath_or_buffer="corpus/dataset/training_data.csv", sep=",", dtype=str)

    # Get entity spans
    df_entity_spans = create_entity_spans(df_train.astype(str))
    training_data = df_entity_spans.values.tolist()

    # Get & Persist DocBin to disk
    doc_bin_train = get_doc_bin(training_data, nlp, verbose=args.verbose)
    doc_bin_train.to_disk("corpus/spacy_docbins/train.spacy")
    ######################################


    ###### Validation dataset prep ###########
    # Read the validation dataset into pandas
    df_test = pd.read_csv(filepath_or_buffer="corpus/dataset/training_data.csv", sep=",", dtype=str)

    # Get entity spans
    df_entity_spans = create_entity_spans(df_test.astype(str))
    validation_data = df_entity_spans.values.tolist()
    # validation_data = training_data

    # Get & Persist DocBin to disk
    doc_bin_test = get_doc_bin(validation_data, nlp)
    doc_bin_test.to_disk("corpus/spacy_docbins/test.spacy")
    ##########################################

if __name__ == "__main__":
    main()