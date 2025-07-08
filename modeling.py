"""
Data Modeling Module

This module handles data modeling tasks including primary key detection,
foreign key inference, and SQL DDL generation.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import defaultdict
from utils import get_column_similarity_score
from llm_analysis import (detect_relationships_with_llm,
                          analyze_primary_keys_with_ai,
                          get_ai_recommended_primary_key)

logger = logging.getLogger(__name__)


def detect_primary_keys(df: pd.DataFrame) -> Dict[str, float]:
    """
    Detect potential primary key columns in a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to confidence scores
    """
    primary_key_candidates = {}
    
    for col in df.columns:
        confidence = calculate_pk_confidence(df[col])
        if confidence > 0.5:  # Only include viable candidates
            primary_key_candidates[col] = confidence
    
    return primary_key_candidates


def detect_composite_keys(df: pd.DataFrame,
                          max_columns: int = 3) -> List[Dict[str, Any]]:
    """
    Detect potential composite primary keys when no single column is unique.
    
    Args:
        df: DataFrame to analyze
        max_columns: Maximum number of columns to consider for composite key
        
    Returns:
        List of composite key candidates with their confidence scores
    """
    if len(df) == 0:
        return []
    
    composite_candidates = []
    
    # Only consider if no single column is a good primary key
    single_pk_candidates = detect_primary_keys(df)
    best_single_confidence = (max(single_pk_candidates.values())
                              if single_pk_candidates else 0.0)
    
    if best_single_confidence > 0.9:  # Very high confidence required
        return []
    
    from itertools import combinations
    
    # Test combinations of 2 to max_columns
    for num_cols in range(2, min(max_columns + 1, len(df.columns) + 1)):
        for col_combination in combinations(df.columns, num_cols):
            confidence = calculate_composite_key_confidence(
                df, col_combination)
            
            if confidence > 0.7:  # High confidence threshold
                composite_candidates.append({
                    'columns': list(col_combination),
                    'confidence': confidence,
                    'uniqueness': (df[list(col_combination)]
                                   .drop_duplicates().shape[0] / len(df)),
                    'null_ratio': (df[list(col_combination)]
                                   .isnull().any().any())
                })
    
    # Sort by confidence and return top candidates
    return sorted(composite_candidates,
                  key=lambda x: x['confidence'], reverse=True)


def calculate_composite_key_confidence(df: pd.DataFrame,
                                       columns: tuple) -> float:
    """
    Calculate confidence score for a composite key.
    
    Args:
        df: DataFrame to analyze
        columns: Tuple of column names forming the composite key
        
    Returns:
        Confidence score between 0 and 1
    """
    if len(df) == 0:
        return 0.0
    
    # Check uniqueness of the combination
    unique_combinations = df[list(columns)].drop_duplicates().shape[0]
    uniqueness_ratio = unique_combinations / len(df)
    
    # Check for null values in any of the columns
    has_nulls_series = df[list(columns)].isnull().any()
    has_nulls = has_nulls_series.sum()
    null_penalty = has_nulls / len(df)
    
    # Base confidence on uniqueness
    confidence = uniqueness_ratio * (1 - null_penalty)
    
    # Bonus for columns that look like IDs
    id_bonus = sum(0.1 for col in columns
                   if isinstance(col, str) and 'id' in col.lower())
    confidence += id_bonus / len(columns)
    
    # Penalty for too many columns (prefer simpler keys)
    complexity_penalty = (len(columns) - 2) * 0.1
    confidence -= complexity_penalty
    
    # Bonus if this is a junction/bridge table pattern
    if (len(columns) == 2 and
        all(isinstance(col, str) and 'id' in col.lower()
            for col in columns)):
        confidence += 0.2
    
    return min(max(confidence, 0.0), 1.0)


def calculate_pk_confidence(series: pd.Series) -> float:
    """
    Calculate confidence score for a column being a primary key.
    
    Args:
        series: Column data
        
    Returns:
        Confidence score between 0 and 1
    """
    if len(series) == 0:
        return 0.0
    
    # Check for uniqueness
    unique_ratio = series.nunique() / len(series)
    
    # Check for null values
    null_ratio = series.isnull().sum() / len(series)
    
    # Base confidence on uniqueness and non-nullness
    confidence = unique_ratio * (1 - null_ratio)
    
    # Bonus for integer types (common for IDs)
    if pd.api.types.is_integer_dtype(series):
        confidence += 0.1
    
    # Bonus for columns with 'id' in the name
    if isinstance(series.name, str) and 'id' in series.name.lower():
        confidence += 0.2
    
    # Penalty for very long text values (unlikely to be IDs)
    if pd.api.types.is_string_dtype(series):
        avg_length = series.astype(str).str.len().mean()
        if avg_length > 50:  # Very long strings
            confidence *= 0.5
    
    return min(confidence, 1.0)


def infer_foreign_keys(
    dataframes: Dict[str, pd.DataFrame],
    similarity_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Infer foreign key relationships between DataFrames.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        similarity_threshold: Minimum similarity for column matching
        
    Returns:
        List of detected relationships
    """
    relationships = []
    
    # First, get primary key candidates for each table
    primary_keys = {}
    composite_keys = {}
    
    for filename, df in dataframes.items():
        pk_candidates = detect_primary_keys(df)
        if pk_candidates:
            # Take the highest confidence primary key
            best_pk = max(pk_candidates.items(), key=lambda x: x[1])
            primary_keys[filename] = best_pk[0]
        else:
            # Try to find composite keys
            composite_candidates = detect_composite_keys(df)
            if composite_candidates:
                best_composite = composite_candidates[0]
                composite_keys[filename] = best_composite['columns']
    
    # Look for foreign key relationships
    for source_file, source_df in dataframes.items():
        for target_file, target_df in dataframes.items():
            if source_file == target_file:
                continue
            
            # Check if target has a single column primary key
            if target_file in primary_keys:
                target_pk = primary_keys[target_file]
                
                # Look for columns in source that might reference target's PK
                for source_col in source_df.columns:
                    fk_confidence = calculate_fk_confidence(
                        source_df[source_col],
                        target_df[target_pk],
                        source_col,
                        target_pk
                    )
                    
                    if fk_confidence > 0.6:
                        relationship = {
                            'source_table': source_file,
                            'source_column': source_col,
                            'target_table': target_file,
                            'target_column': target_pk,
                            'type': determine_relationship_type(
                                source_df[source_col], 
                                target_df[target_pk]
                            ),
                            'confidence': fk_confidence,
                            'reasoning': (
                                f"Column '{source_col}' appears to reference "
                                f"'{target_pk}' in {target_file}"
                            )
                        }
                        relationships.append(relationship)
            
            # TODO: Add composite foreign key detection in future version
    
    # Use LLM for additional relationship detection
    try:
        llm_relationships = detect_relationships_with_llm(dataframes)
        for rel in llm_relationships:
            if rel.get('confidence', 0) > 0.7:  # High confidence only
                relationships.append(rel)
    except Exception as e:
        logger.warning(f"LLM relationship detection failed: {str(e)}")
    
    # Remove duplicates and sort by confidence
    unique_relationships = remove_duplicate_relationships(relationships)
    return sorted(unique_relationships,
                  key=lambda x: x['confidence'], reverse=True)


def calculate_fk_confidence(
    source_col: pd.Series,
    target_col: pd.Series,
    source_name: str,
    target_name: str
) -> float:
    """
    Calculate confidence that source_col is a foreign key to target_col.
    
    Args:
        source_col: Source column data
        target_col: Target column data (potential primary key)
        source_name: Source column name
        target_name: Target column name
        
    Returns:
        Confidence score between 0 and 1
    """
    confidence = 0.0
    
    # Check column name similarity
    name_similarity = get_column_similarity_score(source_name, target_name)
    confidence += name_similarity * 0.4
    
    # Check data type compatibility
    if are_types_compatible(str(source_col.dtype), str(target_col.dtype)):
        confidence += 0.2
    
    # Check value overlap
    source_values = set(source_col.dropna().astype(str))
    target_values = set(target_col.dropna().astype(str))
    
    if source_values and target_values:
        overlap_ratio = len(source_values.intersection(target_values)) / len(source_values)
        confidence += overlap_ratio * 0.4
    
    return min(confidence, 1.0)


def determine_relationship_type(source_col: pd.Series, target_col: pd.Series) -> str:
    """
    Determine the type of relationship between two columns.
    
    Args:
        source_col: Source column
        target_col: Target column
        
    Returns:
        Relationship type string
    """
    # Count unique values
    source_unique = source_col.nunique()
    target_unique = target_col.nunique()
    
    # Check for duplicate values in source
    source_has_duplicates = source_col.duplicated().any()
    
    if not source_has_duplicates and source_unique == target_unique:
        return "one-to-one"
    elif source_has_duplicates:
        return "many-to-one"
    else:
        return "one-to-many"


def are_types_compatible(type1: str, type2: str) -> bool:
    """
    Check if two data types are compatible for relationships.
    
    Args:
        type1: First data type
        type2: Second data type
        
    Returns:
        True if types are compatible
    """
    # Normalize type names
    type1_norm = type1.lower().replace('64', '').replace('32', '')
    type2_norm = type2.lower().replace('64', '').replace('32', '')
    
    # Define compatible type groups
    numeric_types = {'int', 'float', 'number'}
    string_types = {'object', 'string', 'category'}
    
    # Check compatibility
    if (any(t in type1_norm for t in numeric_types) and 
        any(t in type2_norm for t in numeric_types)):
        return True
    
    if (any(t in type1_norm for t in string_types) and 
        any(t in type2_norm for t in string_types)):
        return True
    
    return type1_norm == type2_norm


def remove_duplicate_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate relationship entries.
    
    Args:
        relationships: List of relationship dictionaries
        
    Returns:
        List with duplicates removed
    """
    seen = set()
    unique_relationships = []
    
    for rel in relationships:
        # Create a key to identify duplicates
        key = (
            rel.get('source_table', ''),
            rel.get('source_column', ''),
            rel.get('target_table', ''),
            rel.get('target_column', '')
        )
        
        if key not in seen:
            seen.add(key)
            unique_relationships.append(rel)
    
    return unique_relationships


def generate_ddl(
    dataframes: Dict[str, pd.DataFrame], 
    enable_normalization: bool = True
) -> str:
    """
    Generate SQL DDL statements for the data model.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        enable_normalization: Whether to apply normalization suggestions
        
    Returns:
        SQL DDL statements as string
    """
    ddl_statements = []
    
    # Header comment
    ddl_statements.append("-- Generated DDL for AI Data Model")
    ddl_statements.append("-- Created by AI Data Modeller")
    ddl_statements.append("")
    
    # Generate CREATE TABLE statements
    primary_keys = {}
    composite_keys = {}
    
    for filename, df in dataframes.items():
        table_name = clean_table_name(filename)
        create_table_sql = generate_create_table_sql(df, table_name)
        ddl_statements.append(create_table_sql)
        ddl_statements.append("")
        
        # Store primary key info
        pk_candidates = detect_primary_keys(df)
        if pk_candidates:
            best_pk = max(pk_candidates.items(), key=lambda x: x[1])
            primary_keys[table_name] = best_pk[0]
        else:
            # Check for composite keys
            composite_candidates = detect_composite_keys(df)
            if composite_candidates:
                best_composite = composite_candidates[0]
                composite_keys[table_name] = best_composite['columns']
    
    # Generate foreign key constraints
    relationships = infer_foreign_keys(dataframes)
    if relationships:
        ddl_statements.append("-- Add Foreign Key Constraints")
        
        for rel in relationships:
            if rel.get('confidence', 0) > 0.7:  # High confidence only
                source_table = clean_table_name(rel['source_table'])
                target_table = clean_table_name(rel['target_table'])
                
                fk_sql = generate_foreign_key_sql(
                    source_table,
                    rel['source_column'],
                    target_table,
                    rel['target_column']
                )
                ddl_statements.append(fk_sql)
        
        ddl_statements.append("")
    
    # Add indexes for better performance
    ddl_statements.append("-- Recommended Indexes")
    for table_name, pk_column in primary_keys.items():
        index_sql = (f"CREATE INDEX idx_{table_name}_{pk_column} "
                     f"ON {table_name}({pk_column});")
        ddl_statements.append(index_sql)
    
    # Add indexes for composite keys
    for table_name, composite_cols in composite_keys.items():
        cols_str = '_'.join(composite_cols)
        index_sql = (f"CREATE INDEX idx_{table_name}_{cols_str} "
                     f"ON {table_name}({', '.join(composite_cols)});")
        ddl_statements.append(index_sql)
    
    return '\n'.join(ddl_statements)


def generate_create_table_sql(df: pd.DataFrame, table_name: str) -> str:
    """
    Generate CREATE TABLE SQL statement for a DataFrame.
    
    Args:
        df: DataFrame to convert
        table_name: Name for the table
        
    Returns:
        CREATE TABLE SQL statement
    """
    lines = [f"CREATE TABLE {table_name} ("]
    
    # Generate column definitions
    column_definitions = []
    for col in df.columns:
        col_type = infer_sql_type(df[col])
        nullable = "NULL" if df[col].isnull().any() else "NOT NULL"
        column_definitions.append(f"    {col} {col_type} {nullable}")
    # Add primary key if detected
    pk_candidates = detect_primary_keys(df)
    composite_candidates = detect_composite_keys(df)
    
    # Prefer composite keys for junction tables or low single PK confidence
    if (composite_candidates and
            (not pk_candidates or max(pk_candidates.values()) < 0.85)):
        # Use composite key
        best_composite = composite_candidates[0]
        composite_cols = ', '.join(best_composite['columns'])
        column_definitions.append(
            f"    PRIMARY KEY ({composite_cols}) -- Composite Key"
        )
    elif pk_candidates:
        # Use single column primary key
        best_pk = max(pk_candidates.items(), key=lambda x: x[1])
        pk_column = best_pk[0]
        column_definitions.append(f"    PRIMARY KEY ({pk_column})")
    # If neither single nor composite keys found, no primary key constraint
    
    lines.append(',\n'.join(column_definitions))
    lines.append(");")
    
    return '\n'.join(lines)


def infer_sql_type(series: pd.Series) -> str:
    """
    Infer SQL data type from pandas Series.
    
    Args:
        series: Pandas Series
        
    Returns:
        SQL data type string
    """
    dtype = str(series.dtype)
    
    # Integer types
    if 'int' in dtype:
        max_val = series.max() if not series.empty else 0
        if max_val < 32767:
            return "SMALLINT"
        elif max_val < 2147483647:
            return "INTEGER"
        else:
            return "BIGINT"
    
    # Float types
    elif 'float' in dtype:
        return "DECIMAL(10,2)"
    
    # Boolean type
    elif dtype == 'bool':
        return "BOOLEAN"
    
    # DateTime types
    elif 'datetime' in dtype:
        return "TIMESTAMP"
    
    # String/Object types
    else:
        if not series.empty:
            max_length = series.astype(str).str.len().max()
            if max_length <= 50:
                return f"VARCHAR({min(max_length * 2, 255)})"
            elif max_length <= 255:
                return "VARCHAR(255)"
            else:
                return "TEXT"
        else:
            return "VARCHAR(255)"


def generate_foreign_key_sql(
    source_table: str,
    source_column: str,
    target_table: str,
    target_column: str
) -> str:
    """
    Generate foreign key constraint SQL.
    
    Args:
        source_table: Source table name
        source_column: Source column name
        target_table: Target table name
        target_column: Target column name
        
    Returns:
        ALTER TABLE SQL statement
    """
    constraint_name = f"fk_{source_table}_{source_column}"
    
    return (
        f"ALTER TABLE {source_table} "
        f"ADD CONSTRAINT {constraint_name} "
        f"FOREIGN KEY ({source_column}) "
        f"REFERENCES {target_table}({target_column});"
    )


def clean_table_name(filename: str) -> str:
    """
    Clean filename to create a valid SQL table name.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned table name
    """
    # Remove file extension
    name = filename.split('.')[0]
    
    # Replace invalid characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = f"table_{name}"
    
    # Provide default if empty
    if not name:
        name = "data_table"
    
    return name.lower()


def suggest_normalization(dataframes: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Suggest normalization improvements.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        List of normalization suggestions
    """
    suggestions = []
    
    for filename, df in dataframes.items():
        table_name = clean_table_name(filename)
        
        # Check for repeating groups (1NF violation)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for comma-separated values
                sample_values = df[col].dropna().head(100)
                if any(',' in str(val) for val in sample_values):
                    suggestions.append(
                        f"Table '{table_name}': Column '{col}' may contain "
                        f"comma-separated values. Consider normalizing to separate table."
                    )
        
        # Check for partial dependencies (2NF violation)
        pk_candidates = detect_primary_keys(df)
        if len(pk_candidates) > 1:
            suggestions.append(
                f"Table '{table_name}': Multiple potential primary keys detected. "
                f"Consider if composite key is needed or if table should be split."
            )
        
        # Check for transitive dependencies (3NF violation)
        # This is a simplified check - in practice would need more sophisticated analysis
        non_key_cols = [col for col in df.columns if col not in pk_candidates]
        if len(non_key_cols) > 5:  # Arbitrary threshold
            suggestions.append(
                f"Table '{table_name}': Large number of non-key columns. "
                f"Review for potential transitive dependencies."
            )
    
    return suggestions


def generate_ddl_with_ai(
    dataframes: Dict[str, pd.DataFrame], 
    enable_normalization: bool = True
) -> str:
    """
    Generate SQL DDL statements using AI analysis for better primary key detection.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        enable_normalization: Whether to apply normalization suggestions
        
    Returns:
        SQL DDL statements as string
    """
    ddl_statements = []
    
    # Header comment
    ddl_statements.append("-- Generated DDL for AI Data Model")
    ddl_statements.append("-- Created by AI Data Modeller with AI-Powered Key Detection")
    ddl_statements.append("")
    
    # Get AI analysis for primary keys
    logger.info("Getting AI analysis for primary key recommendations...")
    ai_analysis = analyze_primary_keys_with_ai(dataframes)
    logger.info(f"AI Analysis results: {ai_analysis}")
    
    # Generate CREATE TABLE statements with AI-recommended keys
    primary_keys = {}
    composite_keys = {}
    
    for filename, df in dataframes.items():
        table_name = clean_table_name(filename)
        create_table_sql = generate_create_table_sql_with_ai(
            df, table_name, ai_analysis
        )
        ddl_statements.append(create_table_sql)
        ddl_statements.append("")
        
        # Store primary key info from AI analysis
        ai_recommendation = get_ai_recommended_primary_key(table_name, ai_analysis)
        if ai_recommendation:
            if ai_recommendation.get('primary_key_type') == 'single':
                primary_keys[table_name] = ai_recommendation['recommended_key'][0]
            elif ai_recommendation.get('primary_key_type') == 'composite':
                composite_keys[table_name] = ai_recommendation['recommended_key']
        else:
            # Fallback to statistical analysis
            pk_candidates = detect_primary_keys(df)
            if pk_candidates:
                best_pk = max(pk_candidates.items(), key=lambda x: x[1])
                primary_keys[table_name] = best_pk[0]
            else:
                composite_candidates = detect_composite_keys(df)
                if composite_candidates:
                    best_composite = composite_candidates[0]
                    composite_keys[table_name] = best_composite['columns']
    
    # Generate foreign key constraints
    relationships = infer_foreign_keys(dataframes)
    if relationships:
        ddl_statements.append("-- Add Foreign Key Constraints")
        
        for rel in relationships:
            if rel.get('confidence', 0) > 0.7:  # High confidence only
                source_table = clean_table_name(rel['source_table'])
                target_table = clean_table_name(rel['target_table'])
                
                fk_sql = generate_foreign_key_sql(
                    source_table,
                    rel['source_column'],
                    target_table,
                    rel['target_column']
                )
                ddl_statements.append(fk_sql)
        
        ddl_statements.append("")
    
    # Add indexes for better performance
    ddl_statements.append("-- Recommended Indexes")
    for table_name, pk_column in primary_keys.items():
        index_sql = (f"CREATE INDEX idx_{table_name}_{pk_column} "
                     f"ON {table_name}({pk_column});")
        ddl_statements.append(index_sql)
    
    # Add indexes for composite keys
    for table_name, composite_cols in composite_keys.items():
        cols_str = '_'.join(composite_cols)
        index_sql = (f"CREATE INDEX idx_{table_name}_{cols_str} "
                     f"ON {table_name}({', '.join(composite_cols)});")
        ddl_statements.append(index_sql)
    
    return '\n'.join(ddl_statements)


def generate_create_table_sql_with_ai(
    df: pd.DataFrame, 
    table_name: str, 
    ai_analysis: Dict[str, Dict[str, Any]]
) -> str:
    """
    Generate CREATE TABLE SQL statement using AI recommendations.
    
    Args:
        df: DataFrame to convert
        table_name: Name for the table
        ai_analysis: AI analysis results
        
    Returns:
        CREATE TABLE SQL statement
    """
    lines = [f"CREATE TABLE {table_name} ("]
    
    # Generate column definitions
    column_definitions = []
    for col in df.columns:
        col_type = infer_sql_type(df[col])
        nullable = "NULL" if df[col].isnull().any() else "NOT NULL"
        column_definitions.append(f"    {col} {col_type} {nullable}")
    
    # Add primary key based on AI recommendation
    ai_recommendation = get_ai_recommended_primary_key(table_name, ai_analysis)
    
    if ai_recommendation:
        pk_type = ai_recommendation.get('primary_key_type')
        recommended_key = ai_recommendation.get('recommended_key', [])
        reasoning = ai_recommendation.get('reasoning', '')
        
        if pk_type == 'single' and len(recommended_key) == 1:
            # Single column primary key
            pk_column = recommended_key[0]
            column_definitions.append(
                f"    PRIMARY KEY ({pk_column}) -- AI Recommended: {reasoning[:50]}..."
            )
        elif pk_type == 'composite' and len(recommended_key) > 1:
            # Composite primary key
            composite_cols = ', '.join(recommended_key)
            column_definitions.append(
                f"    PRIMARY KEY ({composite_cols}) -- AI Composite Key: {reasoning[:40]}..."
            )
        # If pk_type is 'none', don't add primary key constraint
    else:
        # Fallback to statistical analysis
        pk_candidates = detect_primary_keys(df)
        composite_candidates = detect_composite_keys(df)
        
        # Prefer composite keys when single-column confidence is low
        if (composite_candidates and
                (not pk_candidates or max(pk_candidates.values()) < 0.85)):
            # Use composite key
            best_composite = composite_candidates[0]
            composite_cols = ', '.join(best_composite['columns'])
            column_definitions.append(
                f"    PRIMARY KEY ({composite_cols}) -- Statistical Composite Key"
            )
        elif pk_candidates:
            # Use single column primary key
            best_pk = max(pk_candidates.items(), key=lambda x: x[1])
            pk_column = best_pk[0]
            column_definitions.append(
                f"    PRIMARY KEY ({pk_column}) -- Statistical Analysis"
            )
    
    lines.append(',\n'.join(column_definitions))
    lines.append(");")
    
    return '\n'.join(lines)
