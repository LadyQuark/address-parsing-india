## Objective
Train a spaCY NLP model to break down an address into following components:
- Address Line 1
- Address Line 2
- District
- City
- Pincode

After training, a user should be able to provide an Excel file with the first column containing addresses, and recieve a new Excel file with new columns containing these components.

## Files
`config` folder contains training config files

`corpus`:
- `corpus/dataset/training_data.csv`: Limited sample data of Indian addresses
- `corpus/rules/entity_ruler_patterns.jsonl`: Entity patterns containing city names to improve the model


## Training

Steps taken to train the model:

1. Prepared `.docbin` files
```shell
python3 training_data_prep.py
```

2. Created entity-ruler pattern file

3. Modified `base_config.cfg` to include `entity-ruler`

4. Created config files
```shell
python3 -m spacy init fill-config config/base_config.cfg config/config.cfg
```

5. Started training
```shell
python3 -m spacy train config/config.cfg \
--paths.train corpus/spacy_docbins/train.spacy \
--paths.dev corpus/spacy_docbins/test.spacy \
--output output/models \
--training.eval_frequency 10 \
--training.max_steps 300
```


## Usage

When given a `.xlsx` file where the first column contains addresses, `process_excel.py` will generate a new Excel file with new columns containing the extracted columns.

```shell
python3 process_excel.py -i test_data.xlsx
```