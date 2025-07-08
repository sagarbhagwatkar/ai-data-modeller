"""
LLM Analysis Module

This module handles OpenAI GPT-4 integration for intelligent data analysis,
schema understanding, and relationship detection.
"""

import openai
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)


def analyze_schema_relationships(profiles: Dict[str, Dict]) -> str:
    """
    Use LLM to analyze schema relationships across multiple data files.
    
    Args:
        profiles: Dictionary of filename -> profile data
        
    Returns:
        LLM analysis results as formatted text
    """
    try:
        # Prepare schema summary for LLM
        schema_summary = prepare_schema_summary(profiles)
        
        prompt = f"""
        You are a data modeling expert. Analyze the following database schemas and provide insights:

        {schema_summary}

        Please provide:
        1. Primary key recommendations for each table
        2. Potential foreign key relationships between tables
        3. Data normalization suggestions
        4. Any data quality concerns
        5. Recommended table names if current names are unclear

        Format your response in clear sections with specific recommendations.
        """
        
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert database designer and data modeler."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error in LLM schema analysis: {str(e)}")
        return f"Error analyzing schemas: {str(e)}"


def suggest_column_mappings(
    profiles: Dict[str, Dict], 
    similarity_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Use LLM to suggest column mappings between similar columns across files.
    
    Args:
        profiles: Dictionary of filename -> profile data
        similarity_threshold: Minimum similarity score for suggestions
        
    Returns:
        List of column mapping suggestions
    """
    try:
        # Find potentially similar columns
        similar_columns = find_similar_columns(profiles, similarity_threshold)
        
        if not similar_columns:
            return []
        
        mappings = []
        for group in similar_columns:
            # Use LLM to verify and explain the mapping
            mapping_analysis = analyze_column_mapping(group)
            if mapping_analysis:
                mappings.append(mapping_analysis)
        
        return mappings
        
    except Exception as e:
        logger.error(f"Error suggesting column mappings: {str(e)}")
        return []


def find_similar_columns(
    profiles: Dict[str, Dict], 
    threshold: float
) -> List[List[Dict]]:
    """
    Find groups of similar columns across different files.
    
    Args:
        profiles: Profile data for all files
        threshold: Similarity threshold
        
    Returns:
        List of column groups that are potentially similar
    """
    from fuzzywuzzy import fuzz
    from utils import get_column_similarity_score
    
    all_columns = []
    
    # Collect all columns with their metadata
    for filename, profile in profiles.items():
        for col_name in profile['basic_info']['column_names']:
            col_stats = profile['column_stats'].get(col_name, {})
            all_columns.append({
                'file': filename,
                'column': col_name,
                'stats': col_stats,
                'dtype': profile['data_types'].get(col_name, 'unknown')
            })
    
    # Group similar columns
    groups = []
    processed = set()
    
    for i, col1 in enumerate(all_columns):
        if i in processed:
            continue
        
        group = [col1]
        processed.add(i)
        
        for j, col2 in enumerate(all_columns[i+1:], i+1):
            if j in processed:
                continue
            
            # Check if columns are from different files
            if col1['file'] == col2['file']:
                continue
            
            # Calculate similarity
            similarity = get_column_similarity_score(
                col1['column'], 
                col2['column']
            )
            
            # Also check data type compatibility
            type_compatible = are_types_compatible(
                col1['dtype'], 
                col2['dtype']
            )
            
            if similarity >= threshold and type_compatible:
                group.append(col2)
                processed.add(j)
        
        # Only include groups with multiple columns
        if len(group) > 1:
            groups.append(group)
    
    return groups


def analyze_column_mapping(column_group: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Use LLM to analyze and validate a group of potentially similar columns.
    
    Args:
        column_group: List of column dictionaries
        
    Returns:
        Analysis result or None if not a valid mapping
    """
    try:
        # Prepare column information for LLM
        column_info = []
        for col in column_group:
            info = {
                'file': col['file'],
                'column_name': col['column'],
                'data_type': col['dtype'],
                'sample_values': col['stats'].get('sample_values', []),
                'unique_count': col['stats'].get('unique_count', 0),
                'null_percentage': col['stats'].get('null_percentage', 0)
            }
            column_info.append(info)
        
        prompt = f"""
        Analyze these columns from different data files to determine if they represent the same logical field:

        {json.dumps(column_info, indent=2, default=str)}

        Consider:
        1. Column name similarity
        2. Data type compatibility
        3. Sample values similarity
        4. Statistical properties

        Respond with a JSON object containing:
        {{
            "is_same_field": boolean,
            "confidence": float (0-1),
            "reasoning": "explanation",
            "suggested_name": "recommended_column_name",
            "relationship_type": "primary_key|foreign_key|attribute"
        }}
        """
        
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a data modeling expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        if analysis.get('is_same_field', False):
            return {
                'columns': column_group,
                'similarity': analysis.get('confidence', 0.0),
                'description': analysis.get('reasoning', ''),
                'suggested_name': analysis.get('suggested_name', ''),
                'relationship_type': analysis.get('relationship_type', 'attribute')
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error analyzing column mapping: {str(e)}")
        return None


def detect_relationships_with_llm(
    dataframes: Dict[str, pd.DataFrame]
) -> List[Dict[str, Any]]:
    """
    Use LLM to detect relationships between tables.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        List of detected relationships
    """
    try:
        # Prepare data summary for LLM
        table_summary = prepare_table_summary(dataframes)
        
        prompt = f"""
        Analyze these database tables and identify relationships:

        {table_summary}

        For each potential relationship, provide:
        1. Source table and column
        2. Target table and column
        3. Relationship type (one-to-one, one-to-many, many-to-many)
        4. Confidence level (0-1)
        5. Reasoning

        Respond with a JSON array of relationships:
        [
            {{
                "source_table": "table_name",
                "source_column": "column_name",
                "target_table": "table_name",
                "target_column": "column_name",
                "relationship_type": "one-to-many",
                "confidence": 0.9,
                "reasoning": "explanation"
            }}
        ]
        """
        
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a database expert. Respond only with valid JSON array."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        relationships = json.loads(response.choices[0].message.content)
        return relationships if isinstance(relationships, list) else []
        
    except Exception as e:
        logger.error(f"Error detecting relationships with LLM: {str(e)}")
        return []


def generate_normalization_suggestions(
    dataframes: Dict[str, pd.DataFrame]
) -> str:
    """
    Use LLM to suggest database normalization improvements.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        Normalization suggestions as formatted text
    """
    try:
        table_summary = prepare_table_summary(dataframes)
        
        prompt = f"""
        Analyze these database tables for normalization opportunities:

        {table_summary}

        Provide specific recommendations for:
        1. First Normal Form (1NF) compliance
        2. Second Normal Form (2NF) improvements
        3. Third Normal Form (3NF) optimizations
        4. Suggested table splits or merges
        5. Index recommendations

        Format your response with clear sections and specific SQL-like examples.
        """
        
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a database normalization expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating normalization suggestions: {str(e)}")
        return f"Error generating suggestions: {str(e)}"


def prepare_schema_summary(profiles: Dict[str, Dict]) -> str:
    """
    Prepare a formatted schema summary for LLM analysis.
    
    Args:
        profiles: Profile data for all files
        
    Returns:
        Formatted schema summary
    """
    summary_lines = []
    
    for filename, profile in profiles.items():
        summary_lines.append(f"\n## Table: {filename}")
        summary_lines.append(f"Rows: {profile['basic_info']['rows']:,}")
        summary_lines.append(f"Columns: {profile['basic_info']['columns']}")
        
        summary_lines.append("\n### Columns:")
        for col_name in profile['basic_info']['column_names']:
            col_stats = profile['column_stats'].get(col_name, {})
            dtype = profile['data_types'].get(col_name, 'unknown')
            
            unique_ratio = col_stats.get('unique_ratio', 0)
            null_pct = col_stats.get('null_percentage', 0)
            
            summary_lines.append(
                f"- {col_name} ({dtype}): "
                f"{unique_ratio:.1%} unique, {null_pct:.1f}% null"
            )
            
            # Add sample values if available
            samples = col_stats.get('sample_values', [])
            if samples:
                sample_str = ', '.join(str(s) for s in samples[:3])
                summary_lines.append(f"  Sample: {sample_str}")
        
        # Add potential keys
        if profile.get('potential_keys'):
            summary_lines.append("\n### Potential Keys:")
            for key in profile['potential_keys']:
                summary_lines.append(
                    f"- {key['column']} ({key['type']}, "
                    f"confidence: {key['confidence']:.2f})"
                )
    
    return '\n'.join(summary_lines)


def prepare_table_summary(dataframes: Dict[str, pd.DataFrame]) -> str:
    """
    Prepare a summary of table structures for LLM analysis.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        Formatted table summary
    """
    summary_lines = []
    
    for filename, df in dataframes.items():
        summary_lines.append(f"\n## Table: {filename}")
        summary_lines.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        summary_lines.append("\n### Schema:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            
            summary_lines.append(
                f"- {col} ({dtype}): "
                f"{unique_count:,} unique, {null_count:,} nulls"
            )
        
        # Add sample data
        summary_lines.append("\n### Sample Data:")
        sample_data = df.head(3).to_dict('records')
        for i, row in enumerate(sample_data, 1):
            row_str = ', '.join(f"{k}={v}" for k, v in row.items())
            summary_lines.append(f"Row {i}: {row_str}")
    
    return '\n'.join(summary_lines)


def are_types_compatible(type1: str, type2: str) -> bool:
    """
    Check if two data types are compatible for relationship detection.
    
    Args:
        type1: First data type
        type2: Second data type
        
    Returns:
        True if types are compatible
    """
    # Define type compatibility groups
    numeric_types = {'int64', 'int32', 'float64', 'float32', 'number'}
    string_types = {'object', 'string', 'category'}
    datetime_types = {'datetime64', 'datetime'}
    
    # Normalize type names
    type1_norm = type1.lower().replace('[ns]', '')
    type2_norm = type2.lower().replace('[ns]', '')
    
    # Check if both types are in the same compatibility group
    if (type1_norm in numeric_types and type2_norm in numeric_types) or \
       (type1_norm in string_types and type2_norm in string_types) or \
       (type1_norm in datetime_types and type2_norm in datetime_types):
        return True
    
    return False


def analyze_primary_keys_with_ai(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Use AI to analyze and recommend primary keys for each table.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        Dictionary with AI recommendations for primary keys
    """
    try:
        # Prepare data summary for AI analysis
        tables_info = {}
        for filename, df in dataframes.items():
            table_name = filename.replace('.csv', '').replace('.xlsx', '').replace('.json', '')
            
            # Get basic stats for each column
            column_info = []
            for col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df)
                null_count = df[col].isnull().sum()
                
                column_info.append({
                    'name': col,
                    'unique_values': unique_count,
                    'total_rows': total_count,
                    'uniqueness_ratio': unique_count / total_count if total_count > 0 else 0,
                    'null_count': null_count,
                    'has_nulls': null_count > 0,
                    'sample_values': df[col].dropna().head(3).tolist()
                })
            
            tables_info[table_name] = {
                'columns': column_info,
                'total_rows': len(df),
                'sample_data': df.head(3).to_dict('records')
            }
        
        # Create AI prompt
        prompt = f"""
Analyze the following database tables and recommend the best primary key strategy for each table.

Table Data Analysis:
{json.dumps(tables_info, indent=2, default=str)}

For each table, determine:
1. If a single column can serve as a primary key (uniqueness_ratio = 1.0 and no nulls)
2. If no single column is suitable, suggest a composite key using 2-3 columns that together would be unique
3. Explain why the recommended approach is best for that specific table

Guidelines:
- A primary key must uniquely identify each row
- Primary keys cannot have null values
- Prefer simple single-column keys when available
- For junction tables or many-to-many relationships, composite keys are often necessary
- Consider the semantic meaning of columns (e.g., order_id + product_id for order items)

Respond in valid JSON format:
{{
  "table_name": {{
    "primary_key_type": "single" | "composite" | "none",
    "recommended_key": ["column1"] or ["column1", "column2"],
    "confidence": 0.0-1.0,
    "reasoning": "explanation of why this approach is recommended",
    "uniqueness_check": "whether the recommended key would be unique"
  }}
}}
"""
        
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert database designer. Analyze data and recommend optimal primary key strategies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "3000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        )
        
        # Parse AI response
        ai_response = response.choices[0].message.content
        logger.info(f"AI Primary Key Analysis Response: {ai_response}")
        
        # Try to parse JSON response
        try:
            if not ai_response:
                logger.warning("Empty AI response")
                return {}
                
            # Clean the response to extract JSON
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                ai_recommendations = json.loads(json_str)
                return ai_recommendations
            else:
                logger.warning("Could not extract JSON from AI response")
                return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse AI response as JSON: {e}")
            return {}
            
    except Exception as e:
        logger.error(f"Error in AI primary key analysis: {str(e)}")
        return {}


def get_ai_recommended_primary_key(table_name: str, ai_analysis: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract AI recommendation for a specific table.
    
    Args:
        table_name: Name of the table
        ai_analysis: AI analysis results
        
    Returns:
        Primary key recommendation or None
    """
    # Try different variations of table name
    variations = [
        table_name,
        table_name.replace('.csv', '').replace('.xlsx', '').replace('.json', ''),
        table_name.replace('_', ''),
        table_name.lower(),
        table_name.upper()
    ]
    
    for variation in variations:
        if variation in ai_analysis:
            return ai_analysis[variation]
    
    return None
