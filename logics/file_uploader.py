def file_uploader(file):
    csv_file_path = file.name
    # Save uploaded file directly
    with open(csv_file_path, "wb") as f:
        f.write(file.getbuffer())

    # Return the path for processing
    json_file_path = csv_file_path.replace('.csv', '.json')
    return csv_file_path, json_file_path
    
    