#!/usr/bin/env python3
"""
Create sample data specifically designed to need composite keys.
"""

import pandas as pd
from modeling import detect_primary_keys, detect_composite_keys, generate_ddl


def create_true_composite_key_scenarios():
    """Create scenarios where composite keys are truly needed."""
    
    # Junction table - classic many-to-many scenario
    order_items = pd.DataFrame({
        'order_id': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 1, 2],  # Many duplicates
        'product_id': [101, 102, 103, 101, 102, 104, 103, 105, 101, 102, 102, 101],  # Many duplicates
        'quantity': [2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 2],  # Many duplicates
        'unit_price': [10.99, 25.50, 8.75, 10.99, 25.50, 15.25, 8.75, 12.00, 10.99, 25.50, 25.50, 10.99],  # Many duplicates
        'discount': [0.0, 0.1, 0.0, 0.05, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.05]  # Many duplicates
    })
    
    # Movie ratings - user + movie should be unique, but neither alone is
    movie_ratings = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 1],  # Many duplicates
        'movie_id': [101, 102, 103, 101, 104, 102, 105, 106, 101, 107, 103, 104],  # Many duplicates
        'rating': [5, 4, 3, 4, 5, 2, 5, 4, 3, 5, 4, 5],  # Many duplicates (limited range 1-5)
        'review_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', 
                       '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
                       '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12'],  # All different but low meaning
        'helpful_votes': [0, 2, 1, 3, 0, 1, 5, 2, 0, 1, 3, 0]  # Many duplicates
    })
    
    return {
        'order_items.csv': order_items,
        'movie_ratings.csv': movie_ratings
    }


def main():
    """Test composite key detection on true junction tables."""
    
    print("🧪 Testing True Composite Key Scenarios")
    print("=" * 50)
    
    datasets = create_true_composite_key_scenarios()
    
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
