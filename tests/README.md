# Tests Directory

This directory contains all test scripts for the AI Data Modeller project. These tests verify the functionality of AI-powered composite key detection, DDL generation, and ERD visualization.

## Test Files

### Core Functionality Tests

- **`test_composite_keys.py`** - Basic composite key detection test
  - Tests detection of composite primary keys in junction tables
  - Verifies DDL generation with composite keys
  - Uses sample data: order_items, enrollments, user_permissions

- **`test_composite_advanced.py`** - Advanced composite key scenarios
  - Tests complex multi-table relationships
  - Verifies edge cases in composite key detection
  - Tests with more realistic business data

- **`test_low_confidence.py`** - Low confidence scenario testing
  - Tests behavior when no clear primary keys exist
  - Verifies fallback mechanisms
  - Tests AI recommendations for ambiguous cases

### AI-Powered Features Tests

- **`test_ai_ddl.py`** - AI-powered DDL generation test
  - Tests `generate_ddl_with_ai()` function
  - Verifies AI primary key recommendations in SQL output
  - Ensures composite keys are correctly represented in DDL

- **`test_erd_composite.py`** - ERD composite key visualization test
  - Tests ERD generation with AI-detected composite keys
  - Verifies 🔗 symbol for composite keys and 🔑 for single keys
  - Tests both Graphviz and fallback ERD generation

## Running Tests

### Run Individual Tests
```bash
# From project root directory
python3 tests/test_composite_keys.py
python3 tests/test_ai_ddl.py
python3 tests/test_erd_composite.py
```

### Run All Tests
```bash
# From project root directory
python3 -m pytest tests/ -v
```

Or run them individually:
```bash
for test in tests/test_*.py; do
    echo "Running $test..."
    python3 "$test"
    echo "---"
done
```

## Test Data

The tests create their own sample data internally, including:
- Customer tables with single primary keys
- Order item tables requiring composite keys
- User permission tables with many-to-many relationships
- Various edge cases for comprehensive testing

## Expected Outputs

All tests should:
- ✅ Complete without errors
- ✅ Generate valid SQL DDL statements
- ✅ Create proper ERD visualizations
- ✅ Correctly identify single and composite primary keys
- ✅ Show appropriate confidence levels for AI recommendations

## Dependencies

Tests require the same dependencies as the main application:
- pandas
- openai
- python-dotenv
- streamlit (for some tests)
- graphviz (optional, for ERD tests)

Make sure your `.env` file contains a valid `OPENAI_API_KEY` for AI-powered tests to work properly.
