import sys
import os
from test_embedding import test_embed_parquet
from test_parquet_info import test_parquet_info
from test_duckdb_conversion import test_duckdb_conversion
from test_postgres_conversion import test_postgres_conversion
from test_markdown_processing import test_markdown_processing

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        ("Embedding Test", test_embed_parquet),
        ("Parquet Info Test", test_parquet_info),
        ("DuckDB Conversion Test", test_duckdb_conversion),
        ("PostgreSQL Conversion Test", test_postgres_conversion),
        ("Markdown Processing Test", test_markdown_processing)
    ]
    
    success_count = 0
    total_tests = len(tests)
    
    print("\nStarting tests...\n")
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name} passed")
                success_count += 1
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with error: {str(e)}")
    
    print(f"\nTest Summary:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {success_count}")
    print(f"Failed: {total_tests - success_count}")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 