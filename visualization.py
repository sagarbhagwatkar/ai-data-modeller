"""
Visualization Module

This module handles ERD diagram generation and data visualization
using Graphviz and other plotting libraries.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import streamlit as st
from modeling import detect_primary_keys, infer_foreign_keys, clean_table_name

logger = logging.getLogger(__name__)


def create_erd_diagram(dataframes: Dict[str, pd.DataFrame]) -> Optional[str]:
    """
    Create an Entity Relationship Diagram using Graphviz with AI-powered primary key detection.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        HTML string containing the ERD or None if generation fails
    """
    try:
        import graphviz
        from llm_analysis import analyze_primary_keys_with_ai
        
        # Create directed graph
        dot = graphviz.Digraph(comment='ERD Diagram')
        dot.attr(rankdir='TB', size='12,8')
        dot.attr('node', shape='record', style='filled', fillcolor='lightblue')
        
        # Get AI-powered primary key recommendations
        ai_pk_analysis = analyze_primary_keys_with_ai(dataframes)
        
        # Add tables as nodes
        primary_keys = {}
        for filename, df in dataframes.items():
            table_name = clean_table_name(filename)
            
            # Get AI-recommended primary key
            table_key = table_name.replace('.csv', '').replace('.xlsx', '')
            table_key = table_key.replace('.json', '')
            ai_recommendation = ai_pk_analysis.get(table_key, {})
            recommended_pk = ai_recommendation.get('recommended_key', [])
            pk_type = ai_recommendation.get('primary_key_type', 'none')
            
            # Store primary key info for relationships
            if recommended_pk:
                primary_keys[table_name] = recommended_pk
            
            # Create table node with columns
            table_label = create_table_label_with_ai(
                df, table_name, recommended_pk, pk_type
            )
            dot.node(table_name, table_label)
        
        # Add relationships as edges
        relationships = infer_foreign_keys(dataframes)
        for rel in relationships:
            if rel.get('confidence', 0) > 0.7:  # High confidence only
                source_table = clean_table_name(rel['source_table'])
                target_table = clean_table_name(rel['target_table'])
                
                # Determine edge style based on relationship type
                edge_style = get_edge_style(rel.get('type', 'many-to-one'))
                
                dot.edge(
                    target_table, 
                    source_table,
                    label=f"{rel.get('confidence', 0):.2f}",
                    **edge_style
                )
        
        # Render to SVG
        svg_content = dot.pipe(format='svg', encoding='utf-8')
        
        # Convert to HTML for display
        html_content = f"""
        <div style="text-align: center; padding: 20px;">
            <h3>Entity Relationship Diagram</h3>
            {svg_content}
        </div>
        """
        
        return html_content
        
    except ImportError:
        logger.error("Graphviz not installed. Cannot generate ERD.")
        return create_fallback_erd(dataframes)
    except Exception as e:
        logger.error(f"Error creating ERD: {str(e)}")
        return create_fallback_erd(dataframes)


def create_table_label_with_ai(
    df: pd.DataFrame, 
    table_name: str, 
    primary_key_columns: List[str],
    pk_type: str
) -> str:
    """
    Create a formatted label for a table node in the ERD using AI recommendations.
    
    Args:
        df: DataFrame representing the table
        table_name: Name of the table
        primary_key_columns: List of primary key column names
        pk_type: Type of primary key ('single', 'composite', 'none')
        
    Returns:
        Formatted label string
    """
    # Table header
    label_parts = [f"{{<table_name> {table_name.upper()}|"]
    
    # Add columns
    for col in df.columns:
        col_type = str(df[col].dtype)
        
        # Mark primary key columns
        if col in primary_key_columns:
            if pk_type == 'composite' and len(primary_key_columns) > 1:
                # Show composite key with different symbol
                label_parts.append(f"<{col}> 🔗 {col} ({col_type})|")
            else:
                # Single primary key
                label_parts.append(f"<{col}> 🔑 {col} ({col_type})|")
        else:
            label_parts.append(f"<{col}> {col} ({col_type})|")
    
    # Remove the last separator and close the label
    label = ''.join(label_parts[:-1]) + label_parts[-1][:-1] + "}}"
    
    return label


def create_table_label(
    df: pd.DataFrame, 
    table_name: str, 
    primary_key: Optional[str]
) -> str:
    """
    Create a formatted label for a table node in the ERD.
    
    Args:
        df: DataFrame representing the table
        table_name: Name of the table
        primary_key: Primary key column name
        
    Returns:
        Formatted label string
    """
    # Table header
    label_parts = [f"{{<table_name> {table_name.upper()}|"]
    
    # Add columns
    for col in df.columns:
        col_type = str(df[col].dtype)
        
        # Mark primary key
        if col == primary_key:
            label_parts.append(f"<{col}> 🔑 {col} ({col_type})|")
        else:
            label_parts.append(f"<{col}> {col} ({col_type})|")
    
    # Remove the last separator and close the label
    label = ''.join(label_parts[:-1]) + label_parts[-1][:-1] + "}}"
    
    return label


def get_edge_style(relationship_type: str) -> Dict[str, str]:
    """
    Get edge styling based on relationship type.
    
    Args:
        relationship_type: Type of relationship
        
    Returns:
        Dictionary of edge attributes
    """
    styles = {
        'one-to-one': {
            'arrowhead': 'normal',
            'arrowtail': 'normal',
            'dir': 'both',
            'color': 'blue'
        },
        'one-to-many': {
            'arrowhead': 'crow',
            'arrowtail': 'normal',
            'dir': 'both',
            'color': 'green'
        },
        'many-to-one': {
            'arrowhead': 'normal',
            'arrowtail': 'crow',
            'dir': 'both',
            'color': 'orange'
        },
        'many-to-many': {
            'arrowhead': 'crow',
            'arrowtail': 'crow',
            'dir': 'both',
            'color': 'red'
        }
    }
    
    return styles.get(relationship_type, styles['many-to-one'])


def create_fallback_erd(dataframes: Dict[str, pd.DataFrame]) -> str:
    """
    Create a text-based fallback ERD when Graphviz is not available.
    Uses AI-powered primary key detection.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        HTML string with text-based ERD
    """
    try:
        from llm_analysis import analyze_primary_keys_with_ai
        ai_pk_analysis = analyze_primary_keys_with_ai(dataframes)
    except Exception:
        ai_pk_analysis = {}
    
    html_parts = [
        "<div style='font-family: monospace; padding: 20px;'>",
        "<h3>Entity Relationship Diagram (Text Format)</h3>",
        "<p><em>Install Graphviz for visual diagrams</em></p>",
        "<pre>"
    ]
    
    # Show tables
    for filename, df in dataframes.items():
        table_name = clean_table_name(filename)
        
        # Get AI-recommended primary key
        table_key = table_name.replace('.csv', '').replace('.xlsx', '')
        table_key = table_key.replace('.json', '')
        ai_recommendation = ai_pk_analysis.get(table_key, {})
        recommended_pk = ai_recommendation.get('recommended_key', [])
        pk_type = ai_recommendation.get('primary_key_type', 'none')
        
        html_parts.append(f"\n┌─ {table_name.upper()} ─┐")
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            if col in recommended_pk:
                if pk_type == 'composite' and len(recommended_pk) > 1:
                    html_parts.append(f"│ 🔗 {col} ({col_type}) [PK]")
                else:
                    html_parts.append(f"│ 🔑 {col} ({col_type}) [PK]")
            else:
                html_parts.append(f"│   {col} ({col_type})")
        
        html_parts.append("└─────────────┘")
    
    # Show relationships
    relationships = infer_foreign_keys(dataframes)
    if relationships:
        html_parts.append("\n\nRelationships:")
        for rel in relationships:
            if rel.get('confidence', 0) > 0.7:
                html_parts.append(
                    f"  {rel['source_table']}.{rel['source_column']} → "
                    f"{rel['target_table']}.{rel['target_column']} "
                    f"({rel.get('type', 'unknown')})"
                )
    
    html_parts.extend(["</pre>", "</div>"])
    
    return ''.join(html_parts)


def display_relationship_summary(dataframes: Dict[str, pd.DataFrame]) -> None:
    """
    Display a summary of detected relationships in Streamlit.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
    """
    st.subheader("📋 Relationship Summary")
    
    # Get all relationships
    relationships = infer_foreign_keys(dataframes)
    
    if not relationships:
        st.info("No relationships detected between the uploaded files.")
        return
    
    # Create summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Relationships", len(relationships))
    
    with col2:
        high_confidence = sum(1 for r in relationships if r.get('confidence', 0) > 0.8)
        st.metric("High Confidence", high_confidence)
    
    with col3:
        unique_tables = len(set(
            [r['source_table'] for r in relationships] + 
            [r['target_table'] for r in relationships]
        ))
        st.metric("Connected Tables", unique_tables)
    
    # Display relationship details
    st.subheader("Detected Relationships")
    
    for i, rel in enumerate(relationships, 1):
        confidence = rel.get('confidence', 0)
        confidence_color = get_confidence_color(confidence)
        
        with st.expander(f"Relationship {i} (Confidence: {confidence:.2f})", 
                        expanded=confidence > 0.8):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Source:**")
                st.write(f"Table: `{rel['source_table']}`")
                st.write(f"Column: `{rel['source_column']}`")
            
            with col2:
                st.markdown("**Target:**")
                st.write(f"Table: `{rel['target_table']}`")
                st.write(f"Column: `{rel['target_column']}`")
            
            st.markdown(f"**Type:** {rel.get('type', 'Unknown')}")
            
            if 'reasoning' in rel:
                st.markdown(f"**Reasoning:** {rel['reasoning']}")
            
            # Confidence indicator
            st.progress(confidence)


def get_confidence_color(confidence: float) -> str:
    """
    Get color based on confidence level.
    
    Args:
        confidence: Confidence score 0-1
        
    Returns:
        Color string
    """
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"


def create_data_profile_charts(dataframes: Dict[str, pd.DataFrame]) -> None:
    """
    Create data profiling charts using Plotly.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        st.subheader("📊 Data Profiling Charts")
        
        # Data quality overview
        fig = create_data_quality_chart(dataframes)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Column type distribution
        fig = create_column_type_chart(dataframes)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.warning("Plotly not available. Install it for enhanced visualizations.")
    except Exception as e:
        logger.error(f"Error creating charts: {str(e)}")
        st.error(f"Error creating charts: {str(e)}")


def create_data_quality_chart(dataframes: Dict[str, pd.DataFrame]):
    """
    Create a data quality overview chart.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        Plotly figure or None
    """
    try:
        import plotly.express as px
        
        # Collect data quality metrics
        quality_data = []
        
        for filename, df in dataframes.items():
            total_cells = df.shape[0] * df.shape[1]
            null_cells = df.isnull().sum().sum()
            complete_ratio = (total_cells - null_cells) / total_cells if total_cells > 0 else 0
            
            quality_data.append({
                'File': filename,
                'Completeness': complete_ratio * 100,
                'Missing %': (null_cells / total_cells) * 100 if total_cells > 0 else 0,
                'Total Rows': df.shape[0],
                'Total Columns': df.shape[1]
            })
        
        # Create bar chart
        fig = px.bar(
            quality_data,
            x='File',
            y='Completeness',
            title='Data Completeness by File',
            labels={'Completeness': 'Completeness (%)'},
            color='Completeness',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=400)
        return fig
        
    except Exception as e:
        logger.error(f"Error creating data quality chart: {str(e)}")
        return None


def create_column_type_chart(dataframes: Dict[str, pd.DataFrame]):
    """
    Create a column type distribution chart.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        
    Returns:
        Plotly figure or None
    """
    try:
        import plotly.express as px
        
        # Collect column type data
        type_counts = {}
        
        for filename, df in dataframes.items():
            for dtype in df.dtypes:
                dtype_str = str(dtype)
                if 'int' in dtype_str:
                    category = 'Integer'
                elif 'float' in dtype_str:
                    category = 'Float'
                elif 'object' in dtype_str:
                    category = 'Text'
                elif 'datetime' in dtype_str:
                    category = 'DateTime'
                elif 'bool' in dtype_str:
                    category = 'Boolean'
                else:
                    category = 'Other'
                
                type_counts[category] = type_counts.get(category, 0) + 1
        
        # Create pie chart
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title='Column Type Distribution'
        )
        
        fig.update_layout(height=400)
        return fig
        
    except Exception as e:
        logger.error(f"Error creating column type chart: {str(e)}")
        return None


def export_erd_as_image(dataframes: Dict[str, pd.DataFrame], format: str = 'png') -> bytes:
    """
    Export ERD diagram as image bytes.
    
    Args:
        dataframes: Dictionary of filename -> DataFrame
        format: Image format ('png', 'pdf', 'svg')
        
    Returns:
        Image bytes or None if export fails
    """
    try:
        import graphviz
        
        # Create the same diagram as in create_erd_diagram
        dot = graphviz.Digraph(comment='ERD Diagram')
        dot.attr(rankdir='TB', size='12,8')
        dot.attr('node', shape='record', style='filled', fillcolor='lightblue')
        
        # Add tables
        primary_keys = {}
        for filename, df in dataframes.items():
            table_name = clean_table_name(filename)
            pk_candidates = detect_primary_keys(df)
            pk_column = None
            if pk_candidates:
                pk_column = max(pk_candidates.items(), key=lambda x: x[1])[0]
                primary_keys[table_name] = pk_column
            
            table_label = create_table_label(df, table_name, pk_column)
            dot.node(table_name, table_label)
        
        # Add relationships
        relationships = infer_foreign_keys(dataframes)
        for rel in relationships:
            if rel.get('confidence', 0) > 0.7:
                source_table = clean_table_name(rel['source_table'])
                target_table = clean_table_name(rel['target_table'])
                edge_style = get_edge_style(rel.get('type', 'many-to-one'))
                
                dot.edge(target_table, source_table, **edge_style)
        
        # Render to specified format
        return dot.pipe(format=format)
        
    except Exception as e:
        logger.error(f"Error exporting ERD: {str(e)}")
        return None
