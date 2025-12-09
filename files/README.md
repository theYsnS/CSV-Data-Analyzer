# CSV Data Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

> **Day 1 of 100 Days of AI/ML Challenge**

A professional-grade CSV analysis tool with data cleaning, statistical analysis, and reporting capabilities.

## Features

- **Load & Validate**: Read CSV files with automatic encoding detection
- **Clean Data**: Remove duplicates, handle missing values (drop/fill/forward-fill)
- **Analyze**: Generate comprehensive statistics (mean, median, std, quartiles, skewness, kurtosis)
- **Filter**: Query data with intuitive conditions
- **Group**: Calculate aggregated statistics by categories
- **Export**: Generate detailed text reports
- **CLI**: Full command-line interface for quick analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/day01-csv-data-analyzer.git
cd day01-csv-data-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

##  Quick Start

### Python API

```python
from src.analyzer import CSVDataAnalyzer

# Load and analyze
analyzer = CSVDataAnalyzer("data/sample_employees.csv")
analyzer.load()

# Clean data
analyzer.clean(drop_duplicates=True, handle_missing='fill')

# Get analysis results
result = analyzer.analyze()
print(f"Shape: {result.shape}")
print(f"Statistics: {result.statistics}")

# Filter data
high_salary = analyzer.filter('salary', '>', 80000)
print(high_salary)

# Group statistics
by_dept = analyzer.group_stats('department', 'salary', ['mean', 'count'])
print(by_dept)

# Export report
analyzer.export_report("output/analysis_report.txt")
```

### Command Line

```bash
# Basic analysis
python -m src.cli data/sample_employees.csv

# Clean data and show first 5 rows
python -m src.cli data/sample_employees.csv --clean --head 5

# Statistics for specific column
python -m src.cli data/sample_employees.csv --stats salary

# Show correlation matrix
python -m src.cli data/sample_employees.csv --corr

# Export report
python -m src.cli data/sample_employees.csv --output report.txt

# Get help
python -m src.cli --help
```

## Project Structure

```
day01-csv-data-analyzer/
├── src/
│   ├── __init__.py       # Package initialization
│   ├── analyzer.py       # Main analyzer class
│   └── cli.py            # Command-line interface
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py  # Unit tests
├── data/
│   └── sample_employees.csv  # Sample data
├── docs/
│   └── API.md            # API documentation
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## API Reference

### CSVDataAnalyzer

| Method | Description |
|--------|-------------|
| `load(filepath)` | Load CSV file |
| `from_dataframe(df)` | Initialize from pandas DataFrame |
| `info()` | Get dataset information |
| `clean(...)` | Clean data (duplicates, missing values) |
| `analyze()` | Run comprehensive analysis |
| `get_summary()` | Get pandas describe() output |
| `get_correlations()` | Calculate correlation matrix |
| `filter(column, condition, value)` | Filter data |
| `group_stats(group_by, agg_column)` | Grouped statistics |
| `export_report(path)` | Export text report |
| `reset()` | Reset to original data |

### AnalysisResult

```python
@dataclass
class AnalysisResult:
    shape: tuple           # (rows, columns)
    columns: List[str]     # Column names
    dtypes: Dict           # Data types
    missing_values: Dict   # Missing value counts
    statistics: Dict       # Numeric statistics
    memory_usage: float    # Memory in MB
    duplicates: int        # Duplicate row count
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test
pytest tests/test_analyzer.py::TestCSVDataAnalyzer::test_analyze -v
```

## Example Output

```
==================================================
CSV DATA ANALYSIS REPORT
==================================================

File: sample_employees.csv
Shape: 20 rows × 7 columns
Memory: 0.01 MB
Duplicates: 0

Numeric Column Summary:
------------------------------------------

  salary:
    mean: 75,052.63 | std: 14,891.23 | range: [52,000.00, 105,000.00]

  age:
    mean: 34.05 | std: 5.98 | range: [25.00, 45.00]

  performance_score:
    mean: 4.06 | std: 0.37 | range: [3.50, 4.80]

==================================================
```

## Roadmap

This is **Day 1** of the 100 Days of AI/ML Challenge:

- [x] Day 1: CSV Data Analyzer (You are here!)
- [ ] Day 2: JSON API Fetcher
- [ ] Day 3: Data Visualizer
- [ ] Day 4: Web Scraper
- [ ] ...
- [ ] Day 100: Portfolio Website

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Yasin SARIGÜL**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---
Star this repo if you find it helpful!
