"""
Example usage of the Provenance Tracker system.
Demonstrates various use cases and patterns.
"""
import os
import pandas as pd
import numpy as np
#go back to previous directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from provenance_pandas import ProvenanceTracker, ProvenanceDataFrame


def example_1_basic_usage():
    """
    Example 1: Basic usage with simple operations
    Shows the minimal code needed to get started
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create sample data with missing values
    data = {
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', '?', 'David', 'Eve'],
        'age': [25, np.nan, 30, 35, 40],
        'salary': [50000, 60000, 65000, np.nan, 75000]
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal data:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Initialize tracker
    tracker = ProvenanceTracker(
        dataset_name="customer_data",
        agent="analyst@company.com"
    )
    pdf = ProvenanceDataFrame(df, tracker)
    
    # Apply transformations (automatically logged)
    pdf = pdf.rename(columns={'customer_id': 'cust_id'})
    pdf = pdf.dropna(subset=['name'])
    pdf = pdf.fillna({'age': pdf.df['age'].median()})
    
    # Get results
    cleaned_df = pdf.to_pandas()
    print("\nCleaned data:")
    print(cleaned_df)
    print(f"Shape: {cleaned_df.shape}")
    
    # Show what was tracked
    print(f"\nOperations logged: {len(tracker.records)}")
    for rec in tracker.records:
        print(f"  - {rec.op_name}: {rec.pre_shape} => {rec.post_shape}")
    
    # Export
    tracker.save_json("example1_provenance.json")
    tracker.save_graph("example1_pipeline.dot")
    print("\n Exported: example1_provenance.json, example1_pipeline.dot")


def example_2_method_chaining():
    """
    Example 2: Method chaining for elegant pipelines
    Shows how to chain multiple operations
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Method Chaining")
    print("="*60)
    
    # Create sample data
    data = {
        'product_id': [1, 2, 3, 4, 5, 6],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'USB Cable', 'Headphones'],
        'price': [999, 25, 79, 349, 15, 129],
        'stock': [5, np.nan, 12, 3, 50, '?'],
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Audio']
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal data:")
    print(df)
    
    # Initialize and chain operations
    tracker = ProvenanceTracker(dataset_name="products", agent="data_engineer")
    
    pdf = (ProvenanceDataFrame(df, tracker)
        .rename(columns={'product_id': 'id', 'product_name': 'name', 'stock': 'quantity'})
        .apply(lambda d: d.replace({'?': None}), name="normalize_missing_markers")
        .fillna({'quantity': 0})
        .apply(lambda d: d[d['price'] > 10], name="filter_high_value_items")
    )
    
    cleaned_df = pdf.to_pandas()
    print("\nCleaned data (chained operations):")
    print(cleaned_df)
    
    print(f"\nOperations in pipeline: {[r.op_name for r in tracker.records]}")
    print(f"Total rows: {df.shape[0]} => {cleaned_df.shape[0]}")
    print(f"Total cols: {df.shape[1]} => {cleaned_df.shape[1]}")
    
    tracker.save_json("example2_provenance.json")
    tracker.save_graph("example2_pipeline.dot")
    print("\n Exported: example2_provenance.json, example2_pipeline.dot")


def example_3_data_quality_tracking():
    """
    Example 3: Data quality tracking
    Shows how missingness is tracked across operations
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Quality Tracking")
    print("="*60)
    
    # Create data with various missing markers
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'email': ['a@test.com', 'b@test.com', '?', 'NA', 'e@test.com', 'f@test.com', ' ?', 'h@test.com'],
        'phone': ['111-1111', np.nan, '333-3333', '444-4444', np.nan, '666-6666', 'NULL', '888-8888'],
        'address': ['123 St', '456 Ave', '789 Rd', np.nan, '111 Way', '222 Ln', '333 Pl', '444 Ct'],
        'value': [100, 200, np.nan, 400, 500, '?', 700, 800]
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal data with various missing markers:")
    print(df)
    print(f"\nMissing values per column:\n{df.isna().sum()}")
    
    tracker = ProvenanceTracker(dataset_name="contact_data", agent="qa_team")
    pdf = ProvenanceDataFrame(df, tracker)
    
    # Normalize missing markers
    pdf = pdf.apply(
        lambda d: d.replace({'?': None, ' ?': None, 'NA': None, 'NULL': None}),
        name="normalize_missing_markers"
    )
    
    # Remove rows with missing emails
    pdf = pdf.dropna(subset=['email'])
    
    # Fill phone with placeholder
    pdf = pdf.fillna({'phone': 'UNKNOWN'})
    
    print("\n" + "-"*60)
    print("Quality Metrics After Transformations:")
    print("-"*60)
    
    for i, rec in enumerate(tracker.records, 1):
        print(f"\nOperation {i}: {rec.op_name}")
        print(f"  Shape: {rec.pre_shape} => {rec.post_shape}")
        print(f"  Pre-missing NaNs:  {rec.data_quality['pre_missing']['na_by_col']}")
        print(f"  Post-missing NaNs: {rec.data_quality['post_missing']['na_by_col']}")
        print(f"  Pre-missing '?':  {rec.data_quality['pre_missing']['question_mark_like_by_col']}")
        print(f"  Post-missing '?': {rec.data_quality['post_missing']['question_mark_like_by_col']}")
    
    tracker.save_json("example3_provenance.json")
    print("\n Exported: example3_provenance.json")


def example_4_merge_and_combine():
    """
    Example 4: Merging and combining DataFrames
    Shows how to track merge and append operations
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Merge and Combine Operations")
    print("="*60)
    
    # Create two related datasets
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4],
        'customer_name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'city': ['NYC', 'LA', 'Chicago', 'Boston']
    })
    
    orders = pd.DataFrame({
        'customer_id': [1, 2, 2, 3, 4, 4],
        'order_id': [101, 102, 103, 104, 105, 106],
        'amount': [150, 200, 75, 300, 125, 180]
    })
    
    print("\nCustomers data:")
    print(customers)
    print("\nOrders data:")
    print(orders)
    
    # Track the merge
    tracker = ProvenanceTracker(dataset_name="sales_data", agent="analyst")
    pdf = ProvenanceDataFrame(customers, tracker)
    
    # Merge with orders
    pdf = pdf.merge(orders, on='customer_id', how='left')
    pdf = pdf.fillna({'order_id': 'NO_ORDER'})
    
    merged_df = pdf.to_pandas()
    print("\nMerged result:")
    print(merged_df)
    print(f"\nShape: {merged_df.shape}")
    
    for rec in tracker.records:
        print(f"\n{rec.op_name}:")
        print(f"  Rows: {rec.pre_shape[0]} => {rec.post_shape[0]}")
        print(f"  Cols: {rec.pre_shape[1]} => {rec.post_shape[1]}")
        print(f"  Added columns: {rec.deltas['column_added']}")
        print(f"  Changed cells: {rec.deltas['changed_cells_on_common']}")
    
    tracker.save_json("example4_provenance.json")
    tracker.save_graph("example4_pipeline.dot")
    print("\n Exported: example4_provenance.json, example4_pipeline.dot")


def example_5_complex_pipeline():
    """
    Example 5: Complex realistic pipeline
    Shows a more complex ETL-like workflow
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Complex ETL Pipeline")
    print("="*60)
    
    # Create realistic messy data
    raw_data = {
        'transaction_id': range(1, 11),
        'date': ['2025-01-01', '2025-01-02', '2025-01-03', np.nan, '2025-01-05',
                 '2025-01-06', '?', '2025-01-08', '2025-01-09', '2025-01-10'],
        'customer_name': ['John', 'Jane', '?', 'Jack', ' ?', 'Jill', 'James', 'Janet', 'Jerry', 'Jessie'],
        'amount': [100, 200, np.nan, 150, 300, np.nan, 250, 175, '?', 225],
        'category': ['Electronics', 'Books', 'Electronics', 'Clothing', np.nan,
                     'Electronics', 'Books', 'Clothing', 'Electronics', 'Books'],
        'status': ['completed', 'completed', 'failed', 'pending', 'completed',
                   'completed', 'failed', 'completed', 'pending', 'completed']
    }
    
    df = pd.DataFrame(raw_data)
    
    print("\nRaw messy data:")
    print(df)
    print(f"\nInitial shape: {df.shape}")
    
    # Initialize tracker
    tracker = ProvenanceTracker(
        dataset_name="transactions_etl",
        agent="etl_pipeline",
        primary_key=['transaction_id']
    )
    pdf = ProvenanceDataFrame(df, tracker)
    
    # Step 1: Normalize missing markers
    pdf = pdf.apply(
        lambda d: d.replace({'?': None, ' ?': None}),
        name="step1_normalize_missing"
    )
    
    # Step 2: Drop rows with missing customer names
    pdf = pdf.dropna(subset=['customer_name'])
    
    # Step 3: Drop rows with missing amounts
    pdf = pdf.dropna(subset=['amount'])
    
    # Step 4: Fill category with default
    pdf = pdf.fillna({'category': 'Uncategorized'})
    
    # Step 5: Filter for completed and pending status only
    pdf = pdf.apply(
        lambda d: d[d['status'].isin(['completed', 'pending'])],
        name="step5_filter_valid_status"
    )
    
    # Step 6: Convert amount to numeric
    pdf = pdf.apply(
        lambda d: d.assign(amount=pd.to_numeric(d['amount'], errors='coerce')),
        name="step6_convert_amount_numeric"
    )
    
    # Step 7: Remove rows with NaN amount (after conversion)
    pdf = pdf.dropna(subset=['amount'])
    
    # Step 8: Rename columns for clarity
    pdf = pdf.rename(columns={
        'transaction_id': 'txn_id',
        'customer_name': 'customer',
        'amount': 'transaction_amount'
    })
    
    final_df = pdf.to_pandas()
    
    print("\n" + "-"*60)
    print("Pipeline Summary:")
    print("-"*60)
    print(f"\nInitial: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Final:   {final_df.shape[0]} rows × {final_df.shape[1]} cols")
    
    print("\nFinal cleaned data:")
    print(final_df)
    
    print("\nDetailed operation log:")
    for i, rec in enumerate(tracker.records, 1):
        print(f"{i}. {rec.op_name:30s} | Shape: {str(rec.pre_shape):10s} => {str(rec.post_shape):10s} | Δrows: {rec.deltas['row_count_change']:+3d}")
    
    # Statistics
    stats = getattr(tracker, "stats_summary", lambda: {})()
    print(f"\nTotal operations: {len(tracker.records)}")
    print(f"Total rows removed: {df.shape[0] - final_df.shape[0]}")
    print(f"Total columns changed: {len(tracker.records[-1].deltas['column_added']) if tracker.records else 0}")
    
    tracker.save_json("example5_provenance.json")
    tracker.save_graph("example5_pipeline.dot")
    print("\n Exported: example5_provenance.json, example5_pipeline.dot")


def example_6_custom_operations():
    """
    Example 6: Custom operations with apply()
    Shows how to use apply() for custom transformations
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Operations with apply()")
    print("="*60)
    
    # Sample data
    data = {
        'user_id': [1, 2, 3, 4, 5],
        'email': ['john@example.com', 'jane@example.com', 'jack@example.com', 'jill@example.com', 'james@example.com'],
        'signup_date': ['2024-01-15', '2024-02-20', '2024-01-10', '2024-03-05', '2024-02-28'],
        'login_count': [45, 3, 89, 12, 156],
        'revenue': [1500.50, 250.75, 3200.00, 450.25, 5600.00]
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal data:")
    print(df)
    
    tracker = ProvenanceTracker(dataset_name="user_analytics", agent="data_scientist")
    pdf = ProvenanceDataFrame(df, tracker)
    
    # Custom operation 1: Extract domain from email
    def extract_email_domain(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d['email_domain'] = d['email'].str.split('@').str[1]
        return d
    
    pdf = pdf.apply(extract_email_domain, name="extract_email_domain")
    
    # Custom operation 2: Calculate user engagement score
    def calc_engagement_score(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        max_logins = d['login_count'].max()
        max_revenue = d['revenue'].max()
        d['engagement_score'] = (d['login_count'] / max_logins * 0.6) + (d['revenue'] / max_revenue * 0.4)
        return d
    
    pdf = pdf.apply(
        calc_engagement_score,
        name="calculate_engagement_score",
        params={"weights": "logins:0.6, revenue:0.4"}
    )
    
    # Custom operation 3: Segment users
    def segment_users(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d['segment'] = pd.cut(d['engagement_score'], bins=3, labels=['Low', 'Medium', 'High'])
        return d
    
    pdf = pdf.apply(segment_users, name="segment_users_by_engagement")
    
    final_df = pdf.to_pandas()
    print("\nFinal data with custom operations:")
    print(final_df)
    
    print("\nOperations applied:")
    for rec in tracker.records:
        print(f"  - {rec.op_name}")
        print(f"    Parameters: {rec.parameters}")
        if rec.deltas['column_added']:
            print(f"    Added cols: {rec.deltas['column_added']}")
    
    tracker.save_json("example6_provenance.json")
    print("\n Exported: example6_provenance.json")


def main():
    """
    Run all examples
    """
    print("\n" + "="*60)
    print("PROVENANCE TRACKER - COMPREHENSIVE EXAMPLES")
    print("="*60)
    
    try:
        example_1_basic_usage()
        example_2_method_chaining()
        example_3_data_quality_tracking()
        example_4_merge_and_combine()
        example_5_complex_pipeline()
        example_6_custom_operations()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nGenerated files:")
        print("  - example1_provenance.json, example1_pipeline.dot")
        print("  - example2_provenance.json, example2_pipeline.dot")
        print("  - example3_provenance.json")
        print("  - example4_provenance.json, example4_pipeline.dot")
        print("  - example5_provenance.json, example5_pipeline.dot")
        print("  - example6_provenance.json")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()