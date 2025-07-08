
The primary goal of this APP to use large language model to do complete data modelling, so make sure to use output from LLM to give user the DDLs and relationship between tables. 

Try to make app as simple as possible, do not use too heavy coding.

Build a Streamlit-based web application that allows users to upload multiple sample data files (in .csv, .xlsx, or .json formats). The app should use a Large Language Model (LLM) to:

Analyze schema and sample data in the uploaded files.

Detect relationships between tables (primary keys, foreign keys, one-to-many, many-to-many).

Infer if two or more columns are actually the same field, despite having different names in different files (e.g., customer_id vs cust_id) using:

Schema similarity (column names, data types, nulls, etc.)

Data similarity (overlapping values, value distributions)

Contextual inference using LLM

Generate and display a clean, interactive ERD diagram using tools like graphviz, pygraphviz, or plotly.

Show a downloadable metadata summary (e.g., inferred data model in JSON or SQL CREATE TABLE statements).

⚙️ Implementation Requirements
Frontend:

Use Streamlit for a simple and reactive UI.

Add a file uploader (st.file_uploader) with support for multiple files.

Add tabs or expandable sections to show:

Parsed file preview

Column mapping suggestions

ERD diagram

JSON/SQL export

Backend:

Use Pandas for file parsing.

Use OpenAI GPT-4 or GPT-4o via API to:

Compare column name semantics (e.g., embeddings, cosine similarity)

Interpret data relationships using sample data

Suggest normalization (1NF, 2NF, 3NF if possible)

Use fuzzy matching (like fuzzywuzzy, difflib, Levenshtein) for column name similarity.

Use data profiling (via pandas-profiling, ydata-profiling, or manual analysis) to support inference.

ERD Generation:

Use tools like graphviz, pygraphviz, mermaid for interactive diagrams.

Show table nodes, columns, keys, and relationships with tooltips or legends.

✅ Coding Standards to Follow
Use PEP8-compliant code.

Structure code in modular form: e.g., utils.py, llm_analysis.py, modeling.py, visualization.py.

Add docstrings and inline comments for every function.

Use type hints (List[str], Dict[str, Any], etc.).

Include error handling (for corrupt files, missing headers, empty tables).

Implement logging using the logging module.

Use environment variables (e.g., via .env) for storing LLM API keys.

✅ Data Modeling Standards
Detect and mark:

Primary Keys (unique, non-null columns)

Foreign Keys (columns matching PKs in other tables)

Composite Keys where needed

Normalize data where possible (1NF, 2NF, 3NF) — avoid repeating groups and partial dependencies.

Detect and document:

One-to-one, One-to-many, Many-to-many relationships.

Surrogate vs Natural keys.

Candidate keys.

Resolve naming collisions or inconsistent naming using LLM suggestions.

⚠️ Edge Cases to Handle
Different column names with same semantics (e.g., cust_id, customerid, client_id).

No clear primary key in some tables.

Same data in multiple columns across files but slightly different formats (e.g., dates, phone numbers).

Circular foreign key references.

Missing values or inconsistent data types.

Empty or sample-only files where data is limited.

Many-to-many bridge tables (e.g., orders-products).

Redundant tables (with duplicated data across multiple files).

Case sensitivity in column names and values.

Ambiguous relationships (more than one possible FK match).

🧠 Example Workflow
User uploads 5 CSV files: customers.csv, orders.csv, products.csv, order_items.csv, shipping.csv.

App parses and profiles data.

LLM infers:

customers.customer_id = PK

orders.customer_ref ≈ customers.customer_id → FK

order_items.order_id ≈ orders.order_id → FK

ERD shows relationships.

App shows mappings: customer_id ~ cust_id, product_id ~ prod_id, etc.

App generates ERD and SQL script:

sql
Copy
Edit
CREATE TABLE customers (...);
CREATE TABLE orders (...);
ALTER TABLE orders ADD FOREIGN KEY (customer_id) REFERENCES customers(customer_id);
