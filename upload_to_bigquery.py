#!/usr/bin/env python3
"""
Script to upload cleaned CSV to BigQuery
"""
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import argparse

def upload_csv_to_bigquery(
    csv_file_path,
    project_id,
    dataset_id,
    table_id,
    gcs_bucket_name=None,
    skip_leading_rows=1,
    create_table=True
):
    """Upload CSV file to BigQuery"""
    
    client = bigquery.Client(project=project_id)
    
    # Create dataset if it doesn't exist
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_id} already exists")
    except NotFound:
        print(f"Creating dataset {dataset_id}")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"  # or your preferred location
        client.create_dataset(dataset)
    
    # Configure the load job
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=skip_leading_rows,
        autodetect=True,  # Auto-detect schema
        allow_quoted_newlines=True,  # Important for CSV with text fields
        allow_jagged_rows=False,  # Don't allow rows with different column counts
        max_bad_records=10,  # Allow some bad records but not too many
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Replace table
    )
    
    print(f"Starting upload of {csv_file_path} to {project_id}.{dataset_id}.{table_id}")
    
    # Upload from local file
    with open(csv_file_path, "rb") as source_file:
        job = client.load_table_from_file(
            source_file, 
            table_ref, 
            job_config=job_config
        )
    
    # Wait for the job to complete
    print("Upload job started. Waiting for completion...")
    job.result()  # Waits for the job to complete
    
    if job.state == 'DONE':
        if job.errors:
            print("Upload completed with errors:")
            for error in job.errors:
                print(f"  - {error}")
        else:
            print("Upload completed successfully!")
            
        # Get final table info
        table = client.get_table(table_ref)
        print(f"Table {table_id} now has {table.num_rows} rows and {len(table.schema)} columns")
        
        # Print schema info
        print("\nTable schema:")
        for field in table.schema:
            print(f"  {field.name}: {field.field_type}")
    else:
        print(f"Upload failed with state: {job.state}")
        if job.errors:
            for error in job.errors:
                print(f"Error: {error}")

def main():
    parser = argparse.ArgumentParser(description='Upload CSV to BigQuery')
    parser.add_argument('--csv-file', required=True, help='Path to CSV file')
    parser.add_argument('--project-id', required=True, help='BigQuery project ID')
    parser.add_argument('--dataset-id', required=True, help='BigQuery dataset ID')
    parser.add_argument('--table-id', required=True, help='BigQuery table ID')
    parser.add_argument('--skip-rows', type=int, default=1, help='Number of header rows to skip')
    
    args = parser.parse_args()
    
    upload_csv_to_bigquery(
        csv_file_path=args.csv_file,
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        skip_leading_rows=args.skip_rows
    )

if __name__ == "__main__":
    main() 