#!/usr/bin/env python3
"""
Test ERD generation with AI-powered composite key detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import os
import sys
from typing import Dict

def test_erd_generation():
    """Test ERD generation with composite keys."""
    print("🎨 Testing ERD Generation with AI Composite Keys")
    print("="*60)
    
    # Create test data that should have composite keys
    order_items_data = {
        'order_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'product_id': [101, 102, 101, 103, 104, 105, 101, 106, 107, 108],
        'quantity': [2, 1, 3, 1, 2, 1, 1, 2, 3, 1],
        'price': [29.99, 15.99, 29.99, 45.99, 12.99, 8.99, 29.99, 22.99, 18.99, 35.99]
    }
    
    customers_data = {
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'created_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    }
    
    # Create DataFrames
    dataframes = {
        'order_items.csv': pd.DataFrame(order_items_data),
        'customers.csv': pd.DataFrame(customers_data)
    }
    
    print(f"📊 Test data created:")
    for name, df in dataframes.items():
        print(f"  • {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Test ERD generation
    try:
        from visualization import create_erd_diagram
        
        print("\n🔍 Generating ERD with AI analysis...")
        erd_html = create_erd_diagram(dataframes)
        
        if erd_html:
            print("✅ ERD generated successfully!")
            
            # Check for composite key indicators
            if '🔗' in erd_html:
                print("✅ Composite keys detected and shown with 🔗 symbol!")
            
            if '🔑' in erd_html:
                print("✅ Single primary keys detected and shown with 🔑 symbol!")
            
            # Save ERD to file for inspection
            with open('test_erd_output.html', 'w') as f:
                f.write(erd_html)
            print("💾 ERD saved to 'test_erd_output.html'")
            
            # Show a snippet of the HTML
            print("\n📄 ERD HTML snippet:")
            lines = erd_html.split('\n')
            for line in lines:
                if '🔗' in line or '🔑' in line:
                    print(f"  {line.strip()}")
                    
        else:
            print("❌ ERD generation failed!")
            
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        print("Testing fallback ERD...")
        
        try:
            from visualization import create_fallback_erd
            fallback_html = create_fallback_erd(dataframes)
            
            if fallback_html:
                print("✅ Fallback ERD generated!")
                
                if '🔗' in fallback_html:
                    print("✅ Composite keys shown in fallback ERD!")
                if '🔑' in fallback_html:
                    print("✅ Single keys shown in fallback ERD!")
                    
                print("\n📄 Fallback ERD snippet:")
                lines = fallback_html.split('\n')
                for line in lines:
                    if '🔗' in line or '🔑' in line:
                        print(f"  {line.strip()}")
            else:
                print("❌ Fallback ERD generation failed!")
                
        except Exception as e:
            print(f"❌ Error in fallback ERD: {e}")
    
    except Exception as e:
        print(f"❌ Error generating ERD: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_erd_generation()
