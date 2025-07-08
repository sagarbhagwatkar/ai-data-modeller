"""
AI Data Modeller - Main Streamlit Application

A web application that uses Large Language Models to analyze uploaded data files
and generate intelligent data models with relationship detection and ERD visualization.
"""

import streamlit as st
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Import custom modules
from utils import load_and_validate_file, profile_dataframe, setup_logging
from llm_analysis import analyze_schema_relationships, suggest_column_mappings
from modeling import (detect_primary_keys, detect_composite_keys, 
                      infer_foreign_keys, generate_ddl, generate_ddl_with_ai)
from visualization import create_erd_diagram, display_relationship_summary

# Load environment variables
load_dotenv()

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Data Modeller",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function."""
    st.title("🤖 AI Data Modeller")
    st.markdown("""
    Upload your data files and let AI analyze relationships, detect patterns, 
    and generate clean data models with ERD diagrams and SQL DDL statements.
    """)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API key not found. Please add it to your .env file.")
        st.code("OPENAI_API_KEY=your_api_key_here")
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload settings
        max_files = st.slider("Max files to upload", 1, 10, 5)
        sample_rows = st.slider("Sample rows for analysis", 100, 1000, 500)
        
        # Analysis settings
        st.subheader("Analysis Settings")
        similarity_threshold = st.slider("Column similarity threshold", 0.5, 1.0, 0.8)
        enable_normalization = st.checkbox("Enable normalization suggestions", True)
        
        st.markdown("---")
        st.markdown("💡 **Tip**: Upload related files (e.g., customers, orders, products) for best results.")
    
    # Main content area
    uploaded_files = st.file_uploader(
        "Choose data files",
        type=['csv', 'xlsx', 'json'],
        accept_multiple_files=True,
        help="Upload CSV, Excel, or JSON files for analysis"
    )
    
    if uploaded_files:
        if len(uploaded_files) > max_files:
            st.warning(f"Please upload no more than {max_files} files.")
            return
        
        # Process uploaded files
        with st.spinner("Processing uploaded files..."):
            dataframes = {}
            file_info = {}
            
            for file in uploaded_files:
                try:
                    df, info = load_and_validate_file(file, sample_rows)
                    if df is not None:
                        dataframes[file.name] = df
                        file_info[file.name] = info
                        logger.info(f"Successfully loaded {file.name}: {df.shape}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    logger.error(f"Error processing {file.name}: {str(e)}")
        
        if not dataframes:
            st.error("No valid files could be processed.")
            return
        
        # Display file summary
        st.success(f"✅ Successfully processed {len(dataframes)} files")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 File Preview", 
            "🔍 Schema Analysis", 
            "🔗 Relationships", 
            "📈 ERD Diagram", 
            "💾 Export"
        ])
        
        with tab1:
            display_file_preview(dataframes, file_info)
        
        with tab2:
            display_schema_analysis(dataframes, similarity_threshold)
        
        with tab3:
            display_relationship_analysis(dataframes, similarity_threshold)
        
        with tab4:
            display_erd_diagram(dataframes)
        
        with tab5:
            display_export_options(dataframes, enable_normalization)

def display_file_preview(dataframes: Dict[str, pd.DataFrame], file_info: Dict[str, Dict]):
    """Display preview of uploaded files."""
    st.header("📊 File Preview")
    
    for filename, df in dataframes.items():
        with st.expander(f"📄 {filename} ({df.shape[0]} rows, {df.shape[1]} columns)"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                info = file_info.get(filename, {})
                st.markdown("**File Info:**")
                st.write(f"• Rows: {df.shape[0]:,}")
                st.write(f"• Columns: {df.shape[1]}")
                st.write(f"• Size: {info.get('size', 'Unknown')}")
                st.write(f"• Type: {info.get('type', 'Unknown')}")
                
                # Column data types
                st.markdown("**Data Types:**")
                for col, dtype in df.dtypes.head(5).items():
                    st.write(f"• {col}: {dtype}")
                if len(df.dtypes) > 5:
                    st.write(f"... and {len(df.dtypes) - 5} more")

def display_schema_analysis(dataframes: Dict[str, pd.DataFrame], similarity_threshold: float):
    """Display schema analysis results."""
    st.header("🔍 Schema Analysis")
    
    with st.spinner("Analyzing schemas with AI..."):
        try:
            # Profile each dataframe
            profiles = {}
            for filename, df in dataframes.items():
                profiles[filename] = profile_dataframe(df)
            
            # Analyze with LLM
            schema_analysis = analyze_schema_relationships(profiles)
            column_mappings = suggest_column_mappings(profiles, similarity_threshold)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔑 Primary Key Analysis")
                for filename, df in dataframes.items():
                    pk_candidates = detect_primary_keys(df)
                    st.write(f"**{filename}:**")
                    
                    if pk_candidates:
                        st.write("**Single Column Keys:**")
                        for col, confidence in pk_candidates.items():
                            st.write(f"• {col} (confidence: {confidence:.2f})")
                    else:
                        st.write("• No clear single-column primary key "
                               "detected")
                        
                        # Check for composite keys
                        composite_candidates = detect_composite_keys(df)
                        if composite_candidates:
                            st.write("**Suggested Composite Keys:**")
                            for composite in composite_candidates[:2]:  # Show top 2
                                cols = ', '.join(composite['columns'])
                                confidence = composite['confidence']
                                st.write(f"• ({cols}) "
                                       f"(confidence: {confidence:.2f})")
                        else:
                            st.write("• No suitable composite key found")
                    st.write("")
            
            with col2:
                st.subheader("🔗 Column Mappings")
                if column_mappings:
                    for mapping in column_mappings:
                        st.write(f"**{mapping['similarity']:.2f}** - {mapping['description']}")
                        for col_info in mapping['columns']:
                            st.write(f"  • {col_info['file']}.{col_info['column']}")
                        st.write("")
                else:
                    st.write("No similar columns detected across files.")
            
            # Display AI analysis
            if schema_analysis:
                st.subheader("🤖 AI Schema Insights")
                st.markdown(schema_analysis)
            
            # Add AI Primary Key Analysis section
            with st.expander("🤖 AI Primary Key Recommendations", expanded=False):
                if st.button("Get AI Primary Key Analysis"):
                    with st.spinner("Getting AI analysis for primary keys..."):
                        try:
                            from llm_analysis import analyze_primary_keys_with_ai
                            ai_pk_analysis = analyze_primary_keys_with_ai(dataframes)
                            
                            if ai_pk_analysis:
                                st.success("✅ AI Analysis Complete!")
                                for table_name, recommendation in ai_pk_analysis.items():
                                    st.write(f"**{table_name}:**")
                                    pk_type = recommendation.get('primary_key_type', 'unknown')
                                    recommended_key = recommendation.get('recommended_key', [])
                                    confidence = recommendation.get('confidence', 0)
                                    reasoning = recommendation.get('reasoning', '')
                                    
                                    if pk_type == 'single':
                                        st.write(f"🔑 **Single Key**: {recommended_key[0]}")
                                    elif pk_type == 'composite':
                                        key_str = ', '.join(recommended_key)
                                        st.write(f"🔗 **Composite Key**: ({key_str})")
                                    else:
                                        st.write("❌ **No suitable primary key found**")
                                    
                                    st.write(f"**Confidence**: {confidence:.2f}")
                                    st.write(f"**Reasoning**: {reasoning}")
                                    st.write("---")
                            else:
                                st.warning("No AI analysis results available")
                        except Exception as e:
                            st.error(f"Error in AI analysis: {str(e)}")
                            logger.error(f"AI primary key analysis error: {str(e)}")
                
        except Exception as e:
            st.error(f"Error during schema analysis: {str(e)}")
            logger.error(f"Schema analysis error: {str(e)}")

def display_relationship_analysis(dataframes: Dict[str, pd.DataFrame], similarity_threshold: float):
    """Display relationship analysis results."""
    st.header("🔗 Relationship Analysis")
    
    with st.spinner("Detecting relationships..."):
        try:
            # Detect foreign key relationships
            relationships = infer_foreign_keys(dataframes, similarity_threshold)
            
            if relationships:
                st.success(f"Found {len(relationships)} potential relationships")
                
                for i, rel in enumerate(relationships, 1):
                    with st.expander(f"Relationship {i}: {rel['type']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Source Table:**")
                            st.write(f"File: {rel['source_table']}")
                            st.write(f"Column: {rel['source_column']}")
                        
                        with col2:
                            st.write("**Target Table:**")
                            st.write(f"File: {rel['target_table']}")
                            st.write(f"Column: {rel['target_column']}")
                        
                        with col3:
                            st.write("**Confidence:**")
                            st.write(f"{rel['confidence']:.2f}")
                            st.write(f"**Type:** {rel['type']}")
                        
                        if 'reasoning' in rel:
                            st.markdown(f"**AI Reasoning:** {rel['reasoning']}")
            else:
                st.info("No clear relationships detected between the uploaded files.")
                
        except Exception as e:
            st.error(f"Error during relationship analysis: {str(e)}")
            logger.error(f"Relationship analysis error: {str(e)}")

def display_erd_diagram(dataframes: Dict[str, pd.DataFrame]):
    """Display ERD diagram."""
    st.header("📈 Entity Relationship Diagram")
    
    with st.spinner("Generating ERD diagram..."):
        try:
            # Create ERD
            erd_html = create_erd_diagram(dataframes)
            
            if erd_html:
                st.components.v1.html(erd_html, height=600, scrolling=True)
            else:
                st.warning("Could not generate ERD diagram. Please ensure graphviz is installed.")
                
            # Display relationship summary
            display_relationship_summary(dataframes)
            
        except Exception as e:
            st.error(f"Error generating ERD: {str(e)}")
            logger.error(f"ERD generation error: {str(e)}")

def display_export_options(dataframes: Dict[str, pd.DataFrame], enable_normalization: bool):
    """Display export options."""
    st.header("💾 Export Options")
    
    st.subheader("📄 SQL DDL Export")
    
    if st.button("Generate SQL DDL", type="primary"):
        with st.spinner("Generating SQL statements with AI analysis..."):
            try:
                ddl_statements = generate_ddl_with_ai(
                    dataframes, enable_normalization
                )
                
                st.code(ddl_statements, language="sql")
                
                # Download button
                st.download_button(
                    label="💾 Download SQL Script",
                    data=ddl_statements,
                    file_name="data_model.sql",
                    mime="text/sql"
                )
                
            except Exception as e:
                st.error(f"Error generating DDL: {str(e)}")


if __name__ == "__main__":
    main()
