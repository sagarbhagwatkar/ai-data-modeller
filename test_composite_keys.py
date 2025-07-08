#!/usr/bin/env python3
"""
Test script for composite key detection functionality.
"""

import pandas as pd
from modeling import detect_primary_keys, detect_composite_keys, generate_ddl


def create_test_data():
    """Create test datasets that need composite keys."""
    
    # Test case 1: Order items table - true many-to-many
    order_items = pd.DataFrame({
        'order_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 1],  # Duplicates
        'product_id': [101, 102, 103, 101, 104, 102, 103, 104, 105, 101],  # Duplicates
        'quantity': [2, 1, 3, 1, 2, 4, 1, 2, 1, 1],  # Duplicates
        'price': [10.99, 25.50, 8.75, 10.99, 15.25, 25.50, 8.75, 15.25, 12.00, 10.99]  # Duplicates
    })
    
    # Test case 2: Student enrollments - each student can enroll in each course only once per semester
    enrollments = pd.DataFrame({
        'student_id': [1001, 1001, 1002, 1002, 1003, 1003, 1001],  # Duplicates
        'course_id': ['CS101', 'MATH201', 'CS101', 'PHYS101', 'CS101', 'MATH201', 'PHYS101'],  # Duplicates
        'semester': ['Fall2023', 'Fall2023', 'Spring2024', 'Spring2024', 'Fall2023', 'Spring2024', 'Fall2023'],
        'grade': ['A', 'B+', 'B', 'A-', 'A-', 'B', 'C+']  # Different grades, but not unique
    })
    
    # Test case 3: Many-to-many relationship with no natural single key
    user_permissions = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4],  # Duplicates
        'permission_id': [10, 20, 30, 10, 40, 20, 30, 40, 10],  # Duplicates
        'granted_date': ['2023-01-01', '2023-01-02', '2023-01-03', 
                        '2023-01-01', '2023-01-05', '2023-01-06',
                        '2023-01-07', '2023-01-08', '2023-01-09'],
        'granted_by': ['admin1', 'admin2', 'admin1', 'admin1', 
                      'admin2', 'admin1', 'admin2', 'admin1', 'admin2']  # Not unique
    })
    
    return {
        'order_items.csv': order_items,
        'enrollments.csv': enrollments,
        'user_permissions.csv': user_permissions
    }
    
    return {
        'order_items.csv': order_items,
        'enrollments.csv': enrollments,
        'user_permissions.csv': user_permissions
    }


def test_composite_key_detection():
    """Test composite key detection on various datasets."""
    
    print("🧪 Testing Composite Key Detection")
    print("=" * 50)
    
    test_datasets = create_test_data()
    
    for filename, df in test_datasets.items():
        print(f"\n📊 Analyzing: {filename}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for single-column primary keys first
        pk_candidates = detect_primary_keys(df)
        print(f"\n🔑 Single-column primary key candidates:")
        if pk_candidates:
            for col, confidence in pk_candidates.items():
                print(f"  • {col}: {confidence:.3f}")
        else:
            print("  • None found")
        
        # Check for composite keys
        composite_candidates = detect_composite_keys(df)
        print(f"\n🔗 Composite key candidates:")
        if composite_candidates:
            for i, composite in enumerate(composite_candidates[:3], 1):
                cols = ', '.join(composite['columns'])
                confidence = composite['confidence']
                uniqueness = composite['uniqueness']
                null_ratio = composite['null_ratio']
                print(f"  {i}. ({cols})")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Uniqueness: {uniqueness:.3f}")
                print(f"     Null ratio: {null_ratio:.3f}")
        else:
            print("  • None found")
        
        print("-" * 40)


def test_ddl_generation():
    """Test DDL generation with composite keys."""
    
    print("\n🛠️  Testing DDL Generation with Composite Keys")
    print("=" * 50)
    
    test_datasets = create_test_data()
    
    # Generate DDL
    ddl_output = generate_ddl(test_datasets)
    print("\n📝 Generated DDL:")
    print(ddl_output)


def main():
    """Main test function."""
    test_composite_key_detection()
    test_ddl_generation()
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main()
