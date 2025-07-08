#!/usr/bin/env python3
"""
Test the AI-powered DDL generation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from modeling import generate_ddl_with_ai

# Environment will be loaded from .env file

def create_test_data_for_ai():
    """Create test data that clearly needs composite keys."""
    
    # True junction table - order items
    order_items = pd.DataFrame({
        'order_id': [1, 1, 1, 2, 2, 2, 3, 3, 1, 2],  # Duplicates
        'product_id': [101, 102, 103, 101, 102, 104, 103, 105, 102, 101],  # Duplicates  
        'quantity': [2, 1, 3, 1, 2, 1, 4, 1, 3, 2],  # Duplicates
        'unit_price': [10.99, 25.50, 8.75, 10.99, 25.50, 15.25, 8.75, 12.00, 25.50, 10.99],  # Duplicates
        'line_total': [21.98, 25.50, 26.25, 10.99, 51.00, 15.25, 35.00, 12.00, 76.50, 21.98]  # Some duplicates
    })
    
    # User permissions - clearly needs composite key
    user_permissions = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 1, 2],  # Many duplicates
        'permission_id': [10, 20, 30, 10, 40, 20, 30, 40, 10, 20, 40, 30],  # Many duplicates
        'granted_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                        '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08', 
                        '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12'],
        'granted_by': ['admin1', 'admin2', 'admin1', 'admin1', 'admin2', 'admin1', 
                      'admin2', 'admin1', 'admin2', 'admin1', 'admin2', 'admin2']  # Duplicates
    })
    
    return {
        'order_items.csv': order_items,
        'user_permissions.csv': user_permissions
    }


def main():
    """Test AI-powered DDL generation."""
    
    print("🤖 Testing AI-Powered DDL Generation")
    print("=" * 50)
    
    test_datasets = create_test_data_for_ai()
    
    # Show the data first
    for filename, df in test_datasets.items():
        print(f"\n📊 Sample data for {filename}:")
        print(df.head())
        print(f"Shape: {df.shape}")
        
        # Check uniqueness of potential keys
        print("\n🔍 Column uniqueness analysis:")
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            print(f"  • {col}: {unique_ratio:.2f} ({df[col].nunique()}/{len(df)} unique)")
        
        print("-" * 40)
    
    # Generate DDL with AI
    print("\n🛠️  Generating DDL with AI Analysis...")
    try:
        ddl_output = generate_ddl_with_ai(test_datasets)
        print("\n📝 Generated DDL with AI:")
        print(ddl_output)
    except Exception as e:
        print(f"❌ Error generating DDL: {str(e)}")
        
        # Fallback to regular DDL generation for comparison
        print("\n🔄 Falling back to statistical DDL generation...")
        try:
            from modeling import generate_ddl
            ddl_output = generate_ddl(test_datasets)
            print("\n📝 Statistical DDL (for comparison):")
            print(ddl_output)
        except Exception as e2:
            print(f"❌ Error with fallback: {str(e2)}")


if __name__ == "__main__":
    main()
