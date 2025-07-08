#!/usr/bin/env python3
"""
Create sample data that truly requires composite keys.
"""

import pandas as pd
from modeling import detect_primary_keys, detect_composite_keys, generate_ddl


def create_low_confidence_scenarios():
    """Create data where no single column is a good primary key."""
    
    # All columns have duplicates and none is clearly a primary key
    user_roles = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 4, 4],  # Duplicates
        'role_id': [10, 20, 10, 30, 20, 30, 30, 20, 10, 10, 20, 30],  # Duplicates
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'Finance', 'Finance', 'HR', 'IT', 'IT', 'HR', 'Finance'],  # Duplicates
        'assigned_date': ['2023-01', '2023-01', '2023-02', '2023-02', '2023-01', '2023-03', '2023-04', '2023-05', '2023-06', '2023-01', '2023-02', '2023-03']  # Some duplicates
    })
    
    # Course prerequisites - junction table
    prerequisites = pd.DataFrame({
        'course_id': ['CS201', 'CS201', 'CS301', 'CS301', 'CS301', 'MATH301', 'MATH301', 'PHYS201'],  # Duplicates
        'prerequisite_id': ['CS101', 'MATH101', 'CS201', 'CS101', 'MATH201', 'MATH201', 'MATH101', 'PHYS101'],  # Duplicates
        'semester_required': ['Any', 'Any', 'Any', 'Any', 'Any', 'Spring', 'Any', 'Any'],  # Many duplicates
        'credits_required': [3, 3, 4, 3, 4, 4, 3, 4]  # Duplicates
    })
    
    return {
        'user_roles.csv': user_roles,
        'prerequisites.csv': prerequisites
    }


def main():
    """Test scenarios with low single-column confidence."""
    
    print("🧪 Testing Low Single-Column Confidence Scenarios")
    print("=" * 60)
    
    datasets = create_low_confidence_scenarios()
    
    for filename, df in datasets.items():
        print(f"\n📊 Analyzing: {filename}")
        print(f"Shape: {df.shape}")
        print(df.head())
        
        # Check single-column primary keys
        pk_candidates = detect_primary_keys(df)
        print(f"\n🔑 Single-column primary key candidates:")
        if pk_candidates:
            best_single = max(pk_candidates.values())
            for col, confidence in sorted(pk_candidates.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {col}: {confidence:.3f}")
            print(f"Best single-column confidence: {best_single:.3f}")
        else:
            print("  • None found")
        
        # Check composite keys
        composite_candidates = detect_composite_keys(df)
        print(f"\n🔗 Composite key candidates:")
        if composite_candidates:
            for i, composite in enumerate(composite_candidates[:3], 1):
                cols = ', '.join(composite['columns'])
                confidence = composite['confidence']
                uniqueness = composite['uniqueness']
                print(f"  {i}. ({cols})")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Uniqueness: {uniqueness:.3f}")
        else:
            print("  • None found")
        
        print("-" * 50)
    
    # Test DDL generation
    print("\n🛠️  Testing DDL Generation")
    print("=" * 50)
    ddl_output = generate_ddl(datasets)
    print(ddl_output)


if __name__ == "__main__":
    main()
