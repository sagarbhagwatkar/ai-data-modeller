# 🤖 AI Data Modeller

A powerful Streamlit-based web application that uses Large Language Models (LLMs) to analyze uploaded data files and automatically generate intelligent data models with relationship detection and ERD visualization.

## ✨ Features

- **Multi-format Support**: Upload CSV, Excel (XLSX), and JSON files
- **AI-Powered Analysis**: Uses OpenAI GPT-4 to analyze schema relationships and detect patterns
- **Intelligent Relationship Detection**: Automatically identifies primary keys, foreign keys, and table relationships
- **Composite Key Support**: Detects and suggests composite primary keys when single-column keys are insufficient
- **Column Mapping**: Detects similar columns across files with different naming conventions
- **Interactive ERD**: Generates beautiful Entity Relationship Diagrams using Graphviz
- **SQL DDL Generation**: Exports clean, normalized SQL CREATE TABLE statements
- **Data Profiling**: Comprehensive data quality analysis and visualization
- **Normalization Suggestions**: AI-powered recommendations for database normalization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- System-level Graphviz installation

### Installation

1. **Install system dependencies**:
   ```bash
   # macOS (using Homebrew)
   brew install graphviz
   
   # Ubuntu/Debian
   sudo apt-get install graphviz graphviz-dev
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** - The `.env` file should contain your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🎯 How to Use

### 1. Upload Data Files
- Click "Choose data files" and select your CSV, XLSX, or JSON files
- The app works best with related files (e.g., customers, orders, products)
- Maximum 5 files recommended for optimal performance

### 2. Explore Your Data
- **File Preview**: View basic statistics and sample data
- **Schema Analysis**: See AI-detected primary keys and column mappings
- **Relationships**: Review detected foreign key relationships
- **ERD Diagram**: Visualize your data model

### 3. Export Results
- **SQL DDL**: Download CREATE TABLE statements with constraints
- **Metadata JSON**: Export complete data model metadata

## 📊 Sample Data

The `sample_data/` directory contains example files:

- `customers.csv` - Customer master data
- `orders.csv` - Order transactions 
- `products.csv` - Product catalog
- `order_items.csv` - Order line items

These demonstrate:
- Primary key detection (`customer_id`, `order_id`, `product_id`)
- Composite key detection (`(order_id, product_id)` for junction tables)
- Foreign key relationships (`customer_ref` → `customer_id`)
- Column name variations (`prod_id` ≈ `product_id`)

## 🧠 AI Capabilities

### Primary Key Detection
The AI can identify both single-column and composite primary keys:
- **Single Column**: `customer_id`, `order_id`, `product_id`
- **Composite Keys**: `(order_id, product_id)` for junction tables
- **Smart Selection**: Prefers composite keys when single-column confidence is low

### Column Mapping Detection
The AI can identify semantically similar columns:
- `customer_id` ≈ `cust_id` ≈ `customer_ref`
- `product_id` ≈ `prod_id` ≈ `item_id`

### Relationship Detection
- **One-to-One**: Unique relationships
- **One-to-Many**: Parent-child relationships
- **Many-to-Many**: Complex relationships with bridge tables

## ⚙️ Configuration

### Environment Variables (.env)
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.1
```

## 🛡️ Error Handling

The application handles:
- File format issues
- API failures
- Data quality problems
- Missing dependencies

## 🔧 Troubleshooting

1. **Graphviz not found**: Install system-level Graphviz first
2. **OpenAI API errors**: Verify API key and quota
3. **Memory issues**: Reduce file size or sample rows
4. **Column detection**: Lower similarity threshold

---

**Built with ❤️ using Python, Streamlit, and AI**
