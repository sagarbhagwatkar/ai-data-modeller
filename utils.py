"""
Utility functions for file processing and data handling.

This module provides functions for loading, validating, and profiling data files
in various formats (CSV, XLSX, JSON).
"""

import pandas as pd
import json
import logging
from typing import Dict, Any, Tuple, Optional
import streamlit as st
from io import StringIO
import numpy as np


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ai_data_modeller.log')
        ]
    )


def load_and_validate_file(
    uploaded_file, 
    sample_rows: int = 500
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load and validate an uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        sample_rows: Maximum number of rows to load for analysis
        
    Returns:
        Tuple of (DataFrame, file_info_dict) or (None, {}) if error
    """
    logger = logging.getLogger(__name__)
    
    try:
        file_info = {
            'name': uploaded_file.name,
            'size': f"{uploaded_file.size / 1024:.1f} KB",
            'type': uploaded_file.type
        }
        
        # Determine file type and load accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, nrows=sample_rows)
            file_info['format'] = 'CSV'
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, nrows=sample_rows)
            file_info['format'] = 'Excel'
            
        elif uploaded_file.name.endswith('.json'):
            # Read JSON and try to normalize it
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            json_data = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                df = pd.json_normalize(json_data)
            elif isinstance(json_data, dict):
                # If it's a dict with one key containing a list, use that
                if len(json_data) == 1:
                    key = list(json_data.keys())[0]
                    if isinstance(json_data[key], list):
                        df = pd.json_normalize(json_data[key])
                    else:
                        df = pd.json_normalize([json_data])
                else:
                    df = pd.json_normalize([json_data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            file_info['format'] = 'JSON'
            
            # Limit rows if needed
            if len(df) > sample_rows:
                df = df.head(sample_rows)
                
        else:
            raise ValueError(f"Unsupported file format: {uploaded_file.name}")
        
        # Basic validation
        if df.empty:
            raise ValueError("File is empty")
        
        if df.shape[1] == 0:
            raise ValueError("No columns found in file")
        
        # Clean column names (remove special characters, spaces)
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Add additional file info
        file_info.update({
            'rows_loaded': len(df),
            'columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        })
        
        logger.info(f"Successfully loaded {uploaded_file.name}: {df.shape}")
        return df, file_info
        
    except Exception as e:
        logger.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return None, {'error': str(e)}


def clean_column_name(column_name: str) -> str:
    """
    Clean and standardize column names.
    
    Args:
        column_name: Original column name
        
    Returns:
        Cleaned column name
    """
    import re
    
    # Convert to string if not already
    column_name = str(column_name)
    
    # Remove special characters, keep only alphanumeric and underscore
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', column_name)
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    # Ensure it doesn't start with a number
    if cleaned and cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    
    # If empty after cleaning, provide a default name
    if not cleaned:
        cleaned = "unnamed_column"
    
    return cleaned.lower()


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive profile of a DataFrame.
    
    Args:
        df: DataFrame to profile
        
    Returns:
        Dictionary containing profile information
    """
    profile = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_names': df.columns.tolist()
        },
        'column_stats': {},
        'data_quality': {
            'total_missing': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'empty_columns': []
        },
        'potential_keys': [],
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Analyze each column
    for col in df.columns:
        col_stats = analyze_column(df[col])
        profile['column_stats'][col] = col_stats
        
        # Check if column could be a primary key
        if col_stats['unique_count'] == len(df) and col_stats['null_count'] == 0:
            profile['potential_keys'].append({
                'column': col,
                'type': 'primary_key',
                'confidence': 1.0
            })
        elif col_stats['unique_ratio'] > 0.95 and col_stats['null_count'] == 0:
            profile['potential_keys'].append({
                'column': col,
                'type': 'candidate_key',
                'confidence': col_stats['unique_ratio']
            })
    
    # Identify empty columns
    for col in df.columns:
        if df[col].isnull().all():
            profile['data_quality']['empty_columns'].append(col)
    
    return profile


def analyze_column(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze a single column/series.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Dictionary with column analysis
    """
    analysis = {
        'dtype': str(series.dtype),
        'null_count': series.isnull().sum(),
        'null_percentage': (series.isnull().sum() / len(series)) * 100,
        'unique_count': series.nunique(),
        'unique_ratio': series.nunique() / len(series) if len(series) > 0 else 0,
        'sample_values': []
    }
    
    # Add sample non-null values
    non_null_values = series.dropna()
    if len(non_null_values) > 0:
        sample_size = min(5, len(non_null_values))
        analysis['sample_values'] = non_null_values.head(sample_size).tolist()
    
    # Type-specific analysis
    if pd.api.types.is_numeric_dtype(series):
        analysis.update({
            'min_value': series.min(),
            'max_value': series.max(),
            'mean_value': series.mean(),
            'median_value': series.median(),
            'std_value': series.std()
        })
    
    elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
        if not series.empty:
            analysis.update({
                'avg_length': series.astype(str).str.len().mean(),
                'min_length': series.astype(str).str.len().min(),
                'max_length': series.astype(str).str.len().max(),
                'common_patterns': detect_patterns(series)
            })
    
    elif pd.api.types.is_datetime64_any_dtype(series):
        analysis.update({
            'min_date': series.min(),
            'max_date': series.max(),
            'date_range_days': (series.max() - series.min()).days if series.notna().any() else 0
        })
    
    return analysis


def detect_patterns(series: pd.Series) -> Dict[str, int]:
    """
    Detect common patterns in string columns.
    
    Args:
        series: String series to analyze
        
    Returns:
        Dictionary of pattern counts
    """
    import re
    
    patterns = {
        'email_pattern': 0,
        'phone_pattern': 0,
        'id_pattern': 0,
        'numeric_pattern': 0,
        'date_pattern': 0
    }
    
    # Sample up to 100 values for pattern detection
    sample = series.dropna().astype(str).head(100)
    
    for value in sample:
        value = str(value).strip()
        
        # Email pattern
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            patterns['email_pattern'] += 1
        
        # Phone pattern (simple)
        elif re.match(r'^[\+]?[\d\s\-\(\)]{7,15}$', value):
            patterns['phone_pattern'] += 1
        
        # ID pattern (starts with letters, followed by numbers)
        elif re.match(r'^[A-Za-z]+\d+$', value):
            patterns['id_pattern'] += 1
        
        # Numeric pattern
        elif re.match(r'^\d+$', value):
            patterns['numeric_pattern'] += 1
        
        # Date pattern (simple)
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', value) or re.match(r'^\d{2}/\d{2}/\d{4}$', value):
            patterns['date_pattern'] += 1
    
    return patterns


def get_column_similarity_score(col1_name: str, col2_name: str) -> float:
    """
    Calculate similarity score between two column names.
    
    Args:
        col1_name: First column name
        col2_name: Second column name
        
    Returns:
        Similarity score between 0 and 1
    """
    from fuzzywuzzy import fuzz
    
    # Clean names
    name1 = clean_column_name(col1_name)
    name2 = clean_column_name(col2_name)
    
    # Calculate different similarity metrics
    ratio = fuzz.ratio(name1, name2) / 100
    partial_ratio = fuzz.partial_ratio(name1, name2) / 100
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100
    
    # Return the maximum similarity score
    return max(ratio, partial_ratio, token_sort_ratio)


def validate_data_consistency(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Validate data consistency across multiple DataFrames.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'consistent_column_types': {},
        'value_overlaps': {},
        'potential_relationships': [],
        'warnings': []
    }
    
    # Get all unique column names
    all_columns = set()
    for df in dataframes.values():
        all_columns.update(df.columns)
    
    # Check for similar column names across files
    similar_columns = {}
    for col in all_columns:
        similar_columns[col] = []
        for filename, df in dataframes.items():
            for df_col in df.columns:
                if col != df_col:
                    similarity = get_column_similarity_score(col, df_col)
                    if similarity > 0.8:  # High similarity threshold
                        similar_columns[col].append({
                            'file': filename,
                            'column': df_col,
                            'similarity': similarity
                        })
    
    # Remove empty entries
    similar_columns = {k: v for k, v in similar_columns.items() if v}
    validation_results['similar_columns'] = similar_columns
    
    return validation_results
