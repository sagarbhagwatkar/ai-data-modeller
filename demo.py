#!/usr/bin/env python3
"""
AI Data Modeller - Demo Script

This script demonstrates how to use the AI Data Modeller components programmatically.
It loads sample data, analyzes it, and generates outputs without the Streamlit UI.
"""

import os
import pandas as pd
from pathlib import Path

# Import our modules
from utils import load_and_validate_file, profile_dataframe
from llm_analysis import analyze_schema_relationships, suggest_column_mappings
from modeling import detect_primary_keys, infer_foreign_keys, generate_ddl
from visualization import create_erd_diagram

def demo_analysis():
    """Run a demonstration of the AI Data Modeller capabilities."""
    
    print("🤖 AI Data Modeller - Demo")
    print("=" * 50)
    
    # Check if sample data exists
    sample_dir = Path("sample_data")
    if not sample_dir.exists():
        print("❌ Sample data directory not found!")
        print("Please run this script from the AI Modeller root directory.")
        return
    
    # Load sample data files
    sample_files = list(sample_dir.glob("*.csv"))
    if not sample_files:
        print("❌ No CSV files found in sample_data directory!")
        return
    
    print(f"📂 Found {len(sample_files)} sample files:")
    for file in sample_files:
        print(f"  - {file.name}")
    
    # Load and process files
    dataframes = {}
    profiles = {}
    
    print("\n📊 Loading and profiling data...")
    for file_path in sample_files:
        # Simulate uploaded file object
        class MockUploadedFile:
            def __init__(self, file_path):
                self.name = file_path.name
                self.size = file_path.stat().st_size
                self.type = "text/csv"
                with open(file_path, 'rb') as f:
                    self.content = f.read()
                self.pos = 0
            
            def read(self, size=-1):
                if size == -1:
                    result = self.content[self.pos:]
                    self.pos = len(self.content)
                else:
                    result = self.content[self.pos:self.pos + size]
                    self.pos += len(result)
                return result
            
            def seek(self, pos):
                self.pos = pos
        
        mock_file = MockUploadedFile(file_path)
        df, info = load_and_validate_file(mock_file, sample_rows=500)
        
        if df is not None:
            dataframes[file_path.name] = df
            profiles[file_path.name] = profile_dataframe(df)
            print(f"  ✅ {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"  ❌ Failed to load {file_path.name}")
    
    if not dataframes:
        print("❌ No data files could be loaded!")
        return
    
    # Analyze primary keys
    print("\n🔑 Detecting Primary Keys...")
    for filename, df in dataframes.items():
        pk_candidates = detect_primary_keys(df)
        print(f"\n{filename}:")
        if pk_candidates:
            for col, confidence in pk_candidates.items():
                print(f"  - {col}: {confidence:.2f} confidence")
        else:
            print("  - No clear primary keys detected")
    
    # Detect relationships
    print("\n🔗 Detecting Relationships...")
    relationships = infer_foreign_keys(dataframes, similarity_threshold=0.8)
    
    if relationships:
        print(f"Found {len(relationships)} potential relationships:")
        for i, rel in enumerate(relationships, 1):
            rel_type = rel.get('type', 'unknown').upper()
            print(f"\n{i}. {rel_type}")
            source_table = rel.get('source_table', 'unknown')
            source_column = rel.get('source_column', 'unknown')
            target_table = rel.get('target_table', 'unknown')
            target_column = rel.get('target_column', 'unknown')
            confidence = rel.get('confidence', 0.0)
            
            print(f"   {source_table}.{source_column} → {target_table}.{target_column}")
            print(f"   Confidence: {confidence:.2f}")
            if 'reasoning' in rel:
                print(f"   Reasoning: {rel['reasoning']}")
    else:
        print("No relationships detected.")
    
    # Analyze column mappings
    print("\n🔄 Analyzing Column Mappings...")
    try:
        column_mappings = suggest_column_mappings(profiles, similarity_threshold=0.8)
        if column_mappings:
            print(f"Found {len(column_mappings)} column mapping suggestions:")
            for mapping in column_mappings:
                print(f"\n- Similarity: {mapping['similarity']:.2f}")
                print(f"  Description: {mapping['description']}")
                for col_info in mapping['columns']:
                    print(f"  • {col_info['file']}.{col_info['column']}")
        else:
            print("No column mappings detected.")
    except Exception as e:
        print(f"Column mapping analysis failed: {str(e)}")
    
    # Generate SQL DDL
    print("\n💾 Generating SQL DDL...")
    try:
        ddl_statements = generate_ddl(dataframes, enable_normalization=True)
        print("Generated DDL:")
        print("-" * 40)
        print(ddl_statements)
        
        # Save DDL to file
        with open("generated_model.sql", "w") as f:
            f.write(ddl_statements)
        print("\n✅ DDL saved to generated_model.sql")
        
    except Exception as e:
        print(f"DDL generation failed: {str(e)}")
    
    # Generate ERD (basic)
    print("\n📈 Generating ERD...")
    try:
        erd_html = create_erd_diagram(dataframes)
        if erd_html:
            # Save ERD to HTML file
            with open("generated_erd.html", "w") as f:
                f.write(erd_html)
            print("✅ ERD saved to generated_erd.html")
        else:
            print("❌ ERD generation failed")
    except Exception as e:
        print(f"ERD generation failed: {str(e)}")
    
    print("\n🎉 Demo completed!")
    print("\nGenerated files:")
    print("- generated_model.sql (SQL DDL statements)")
    print("- generated_erd.html (ERD diagram)")
    print("\nTo see the full interactive experience, run:")
    print("streamlit run app.py")

if __name__ == "__main__":
    # Set up environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("⚠️  Warning: OpenAI API key not configured!")
        print("Some features requiring AI analysis will be skipped.")
        print("Configure your API key in the .env file for full functionality.")
        print()
    
    # Run the demo
    demo_analysis()
