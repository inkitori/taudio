import pandas as pd
import json

def process_data(csv_path, ontology_path, output_path, labels_to_filter):
    """
    Processes a CSV and ontology JSON file to produce a new CSV with transformed data.

    Args:
        csv_path (str): The file path for the input CSV.
        ontology_path (str): The file path for the ontology JSON.
        output_path (str): The file path for the output CSV.
        labels_to_filter (list): A list of labels to filter the CSV by.
    """
    # Read the original CSV and the ontology JSON file
    df = pd.read_csv(csv_path, skipinitialspace=True)
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)

    # Create a dictionary to map label IDs to names
    label_map = {item['id']: item['name'] for item in ontology}

    # Filter the DataFrame to only include the specified labels
    df_filtered = df[df['label'].isin(labels_to_filter)].copy()

    # Create the new DataFrame with the specified columns
    new_df = pd.DataFrame()
    new_df['id'] = df_filtered['segment_id'].str.split('_').str[:-1].str.join('_')
    new_df['start_time'] = df_filtered['segment_id'].str.split('_').str[-1].astype(int) / 1000
    new_df['end_time'] = new_df['start_time'] + 10
    new_df['ontology_label'] = df_filtered['label']
    new_df['label_name'] = df_filtered['label'].map(label_map)
    new_df['label_start'] = df_filtered['start_time_seconds'] + new_df['start_time']
    new_df['label_end'] = df_filtered['end_time_seconds'] + new_df['start_time']

    # Save the new DataFrame to a CSV file
    new_df.to_csv(output_path, index=False)
    print(f"Processing complete. The output has been saved to {output_path}")

if __name__ == '__main__':
    # Define file paths and the labels to filter
    input_csv_path = 'input.csv'
    ontology_json_path = 'ontology.json'
    output_csv_path = 'output.csv'
    labels_to_include = ['/m/05zppz', '/m/02zsn', '/m/0ytgt']

    # Create dummy files for testing
    csv_data = """segment_id,start_time_seconds,end_time_seconds,label
b0RFKhbpFJA_30000,0.000,10.000,/m/03m9d0z
b0RFKhbpFJA_30000,4.753,5.720,/m/05zppz
b0RFKhbpFJA_30000,0.000,10.000,/m/07pjwq1
NQNtnl0zaqU_70000,1.200,2.183,/m/02zsn
4PPmY_-YrA_30000,0.000,10.000,/m/0ytgt
"""

    ontology_data = """
[
  {
    "id": "/m/05zppz",
    "name": "Speech"
  },
  {
    "id": "/m/02zsn",
    "name": "Inside, small room"
  },
  {
    "id": "/m/0ytgt",
    "name": "Child speech, kid speaking"
  }
]
"""
    with open(input_csv_path, 'w') as f:
        f.write(csv_data)
    with open(ontology_json_path, 'w') as f:
        f.write(ontology_data)

    # Process the data
    process_data(input_csv_path, ontology_json_path, output_csv_path, labels_to_include)

    # Optional: print the content of the output file to verify
    print("\nContent of the output CSV file:")
    with open(output_csv_path, 'r') as f:
        print(f.read())