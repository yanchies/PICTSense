import streamlit as st
import pandas as pd
import csv
import json

def file_uploader(file):
    data = {}

    with open(file.name, encoding='utf-8', newline='') as csvf:
        csvReader = csv.reader(csvf)
        header = next(csvReader)

        oer_index = header.index('OER') if 'OER' in header else 0

        for idx, row in enumerate(csvReader):
            data[f'response_{idx + 1}'] = row[oer_index]

    json_file_path = file.name.replace('.csv', '.json')

    with open(json_file_path, 'w', encoding='utf-8') as jsonf:
        json.dump(data, jsonf, ensure_ascii=False, indent=4)

    return [file.name, json_file_path]
    