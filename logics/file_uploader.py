import streamlit as st
import pandas as pd
import csv
import json

def file_uploader(file):
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())

    data = {} 

    with open(file.name, encoding='utf-8') as csvf:
        csvReader = csv.reader(csvf)
        header = next(csvReader)

        if 'OER' in header:
            oer_index = header.index('OER')
        else:
            # If the column name isn't exactly 'OER', assume the first column contains the open-ended responses
            oer_index = 0

        # Convert each row into a dictionary and add it to data
        for idx, row in enumerate(csvReader):
            response = row[oer_index]
            data[f'response_{idx + 1}'] = response  # Use unique keys for each response

    csv_file_path = file.name
    json_file_path = file.name.replace('.csv', '.json')

    with open(json_file_path, 'w', encoding='utf-8') as jsonf:
        json.dump(data, jsonf, ensure_ascii=False, indent=4)

    return [csv_file_path, json_file_path]    
    