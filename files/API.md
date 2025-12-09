# API Documentation

## CSVDataAnalyzer Class

The main class for CSV file analysis.

### Initialization

```python
from src.analyzer import CSVDataAnalyzer

# Initialize empty
analyzer = CSVDataAnalyzer()

# Initialize with file path
analyzer = CSVDataAnalyzer("path/to/file.csv")
```

### Methods

#### `load(filepath=None, encoding='utf-8', sep=',', **kwargs)`

Load a CSV file into the analyzer.

**Parameters:**
- `filepath` (str, optional): Path to CSV file
- `encoding` (str): File encoding (default: 'utf-8')
- `sep` (str): Column separator (default: ',')
- `**kwargs`: Additional pandas read_csv arguments

**Returns:** Self (for method chaining)

**Example:**
```python
analyzer.load("data.csv")
analyzer.load("data.csv", encoding='latin-1', sep=';')
```

---

#### `from_dataframe(df)`

Initialize analyzer from an existing pandas DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to analyze

**Returns:** Self (for method chaining)

**Example:**
```python
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
analyzer.from_dataframe(df)
```

---

#### `info()`

Get basic information about the dataset.

**Returns:** Dictionary with:
- `rows`: Number of rows
- `columns`: Number of columns
- `column_names`: List of column names
- `dtypes`: Dictionary of column data types
- `memory_mb`: Memory usage in megabytes

**Example:**
```python
info = analyzer.info()
print(f"Dataset has {info['rows']} rows")
```

---

#### `clean(drop_duplicates=True, handle_missing='keep', fill_value=None, subset=None)`

Clean the dataset by handling duplicates and missing values.

**Parameters:**
- `drop_duplicates` (bool): Remove duplicate rows (default: True)
- `handle_missing` (str): Strategy for missing values
  - `'keep'`: Keep missing values as is
  - `'drop'`: Drop rows with missing values
  - `'fill'`: Fill with `fill_value` or column mean/mode
  - `'ffill'`: Forward fill
  - `'bfill'`: Backward fill
- `fill_value` (any, optional): Value to use when `handle_missing='fill'`
- `subset` (list, optional): Columns to consider for operations

**Returns:** Self (for method chaining)

**Example:**
```python
# Remove duplicates and fill missing with mean/mode
analyzer.clean(drop_duplicates=True, handle_missing='fill')

# Fill missing with specific value
analyzer.clean(handle_missing='fill', fill_value=0)
```

---

#### `analyze()`

Perform comprehensive analysis on the dataset.

**Returns:** `AnalysisResult` dataclass with:
- `shape`: Tuple (rows, columns)
- `columns`: List of column names
- `dtypes`: Dictionary of column data types
- `missing_values`: Dictionary of missing value counts per column
- `statistics`: Dictionary of statistics for numeric columns
- `memory_usage`: Memory in megabytes
- `duplicates`: Number of duplicate rows

**Statistics included:**
- mean, median, std
- min, max
- q25 (25th percentile), q75 (75th percentile)
- skewness, kurtosis

**Example:**
```python
result = analyzer.analyze()
print(f"Mean salary: {result.statistics['salary']['mean']}")
```

---

#### `filter(column, condition, value)`

Filter data based on a condition.

**Parameters:**
- `column` (str): Column name to filter on
- `condition` (str): One of:
  - `'=='`, `'!='`: Equality/inequality
  - `'>'`, `'<'`, `'>='`, `'<='`: Comparisons
  - `'contains'`: String contains
  - `'isin'`: Value in list
- `value` (any): Value to compare against

**Returns:** Filtered pandas DataFrame

**Example:**
```python
# Get high earners
high_salary = analyzer.filter('salary', '>', 80000)

# Get IT department
it_dept = analyzer.filter('department', '==', 'IT')

# Get names containing 'John'
johns = analyzer.filter('name', 'contains', 'John')

# Get specific departments
depts = analyzer.filter('department', 'isin', ['IT', 'HR'])
```

---

#### `group_stats(group_by, agg_column, agg_funcs=['mean', 'sum', 'count'])`

Calculate grouped statistics.

**Parameters:**
- `group_by` (str or list): Column(s) to group by
- `agg_column` (str): Column to aggregate
- `agg_funcs` (list): Aggregation functions to apply

**Returns:** pandas DataFrame with grouped statistics

**Example:**
```python
# Average salary by department
by_dept = analyzer.group_stats('department', 'salary', ['mean', 'count'])
```

---

#### `get_summary()`

Get pandas describe() output for quick summary.

**Returns:** pandas DataFrame with summary statistics

---

#### `get_correlations()`

Calculate correlation matrix for numeric columns.

**Returns:** pandas DataFrame with correlation matrix

---

#### `export_report(output_path, include_data=False)`

Export analysis report to a text file.

**Parameters:**
- `output_path` (str): Path for the output file
- `include_data` (bool): Include sample of raw data

**Example:**
```python
analyzer.export_report("report.txt", include_data=True)
```

---

#### `reset()`

Reset DataFrame to original state (undo all cleaning).

**Returns:** Self (for method chaining)

---

## AnalysisResult Dataclass

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AnalysisResult:
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    statistics: Dict[str, Dict[str, float]]
    memory_usage: float
    duplicates: int
```

## Complete Example

```python
from src.analyzer import CSVDataAnalyzer

# Create analyzer and load data
analyzer = CSVDataAnalyzer("employees.csv")
analyzer.load()

# Get initial info
print("Initial info:", analyzer.info())

# Clean data
analyzer.clean(
    drop_duplicates=True,
    handle_missing='fill'
)

# Run analysis
result = analyzer.analyze()
print(f"Shape: {result.shape}")
print(f"Missing values: {result.missing_values}")

# Get statistics for salary
if 'salary' in result.statistics:
    stats = result.statistics['salary']
    print(f"Salary - Mean: ${stats['mean']:,.2f}, Median: ${stats['median']:,.2f}")

# Filter high performers
top_performers = analyzer.filter('performance_score', '>=', 4.0)
print(f"Top performers: {len(top_performers)}")

# Group by department
dept_stats = analyzer.group_stats('department', 'salary', ['mean', 'count', 'std'])
print(dept_stats)

# Export report
analyzer.export_report("analysis_report.txt")
```
