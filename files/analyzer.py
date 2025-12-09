"""
CSV Data Analyzer - Day 1 of 100 Days of AI/ML
A professional CSV analysis tool with data cleaning, statistics, and reporting.

Author: Yasin SARIGÜL
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Data class to store analysis results."""
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    statistics: Dict[str, Dict[str, float]]
    memory_usage: float
    duplicates: int


class CSVDataAnalyzer:
    """
    A comprehensive CSV data analyzer with cleaning and statistical capabilities.
    
    Features:
        - Load and validate CSV files
        - Clean data (handle missing values, duplicates)
        - Generate descriptive statistics
        - Export analysis reports
    
    Example:
        >>> analyzer = CSVDataAnalyzer("data.csv")
        >>> analyzer.load()
        >>> result = analyzer.analyze()
        >>> print(result.statistics)
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            filepath: Path to the CSV file (optional, can be set later)
        """
        self.filepath = Path(filepath) if filepath else None
        self.df: Optional[pd.DataFrame] = None
        self._original_df: Optional[pd.DataFrame] = None
        logger.info("CSVDataAnalyzer initialized")
    
    def load(
        self,
        filepath: Optional[str] = None,
        encoding: str = 'utf-8',
        sep: str = ',',
        **kwargs
    ) -> 'CSVDataAnalyzer':
        """
        Load a CSV file into the analyzer.
        
        Args:
            filepath: Path to CSV file (uses init path if not provided)
            encoding: File encoding (default: utf-8)
            sep: Column separator (default: comma)
            **kwargs: Additional pandas read_csv arguments
            
        Returns:
            Self for method chaining
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If no filepath is provided
        """
        if filepath:
            self.filepath = Path(filepath)
        
        if not self.filepath:
            raise ValueError("No filepath provided")
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        logger.info(f"Loading CSV from: {self.filepath}")
        
        self.df = pd.read_csv(
            self.filepath,
            encoding=encoding,
            sep=sep,
            **kwargs
        )
        self._original_df = self.df.copy()
        
        logger.info(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        return self
    
    def from_dataframe(self, df: pd.DataFrame) -> 'CSVDataAnalyzer':
        """
        Initialize analyzer from an existing DataFrame.
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Self for method chaining
        """
        self.df = df.copy()
        self._original_df = df.copy()
        logger.info(f"Loaded DataFrame with {len(df)} rows")
        return self
    
    def info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        self._check_loaded()
        
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def clean(
        self,
        drop_duplicates: bool = True,
        handle_missing: str = 'keep',
        fill_value: Any = None,
        subset: Optional[List[str]] = None
    ) -> 'CSVDataAnalyzer':
        """
        Clean the dataset.
        
        Args:
            drop_duplicates: Remove duplicate rows
            handle_missing: Strategy for missing values
                - 'keep': Keep as is
                - 'drop': Drop rows with missing values
                - 'fill': Fill with fill_value or column mean/mode
                - 'ffill': Forward fill
                - 'bfill': Backward fill
            fill_value: Value to use when handle_missing='fill'
            subset: Columns to consider for operations
            
        Returns:
            Self for method chaining
        """
        self._check_loaded()
        
        initial_rows = len(self.df)
        
        # Handle duplicates
        if drop_duplicates:
            self.df = self.df.drop_duplicates(subset=subset)
            dropped = initial_rows - len(self.df)
            if dropped > 0:
                logger.info(f"Removed {dropped} duplicate rows")
        
        # Handle missing values
        if handle_missing == 'drop':
            self.df = self.df.dropna(subset=subset)
            dropped = initial_rows - len(self.df)
            logger.info(f"Dropped {dropped} rows with missing values")
            
        elif handle_missing == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                # Smart fill: mean for numeric, mode for categorical
                for col in self.df.columns:
                    if subset and col not in subset:
                        continue
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    else:
                        mode = self.df[col].mode()
                        if len(mode) > 0:
                            self.df[col].fillna(mode[0], inplace=True)
            logger.info("Filled missing values")
            
        elif handle_missing == 'ffill':
            self.df = self.df.ffill()
            logger.info("Forward filled missing values")
            
        elif handle_missing == 'bfill':
            self.df = self.df.bfill()
            logger.info("Backward filled missing values")
        
        return self
    
    def analyze(self) -> AnalysisResult:
        """
        Perform comprehensive analysis on the dataset.
        
        Returns:
            AnalysisResult object with all statistics
        """
        self._check_loaded()
        
        logger.info("Running comprehensive analysis...")
        
        # Calculate statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        statistics = {}
        
        for col in numeric_cols:
            statistics[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'q25': float(self.df[col].quantile(0.25)),
                'q75': float(self.df[col].quantile(0.75)),
                'skewness': float(self.df[col].skew()),
                'kurtosis': float(self.df[col].kurtosis())
            }
        
        result = AnalysisResult(
            shape=self.df.shape,
            columns=list(self.df.columns),
            dtypes={col: str(dtype) for col, dtype in self.df.dtypes.items()},
            missing_values=self.df.isnull().sum().to_dict(),
            statistics=statistics,
            memory_usage=self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            duplicates=self.df.duplicated().sum()
        )
        
        logger.info("Analysis complete")
        return result
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get pandas describe() output for quick summary.
        
        Returns:
            Summary statistics DataFrame
        """
        self._check_loaded()
        return self.df.describe(include='all')
    
    def get_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.
        
        Returns:
            Correlation matrix DataFrame
        """
        self._check_loaded()
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()
    
    def filter(
        self,
        column: str,
        condition: str,
        value: Any
    ) -> pd.DataFrame:
        """
        Filter data based on condition.
        
        Args:
            column: Column name to filter on
            condition: One of '==', '!=', '>', '<', '>=', '<=', 'contains', 'isin'
            value: Value to compare against
            
        Returns:
            Filtered DataFrame
        """
        self._check_loaded()
        
        conditions = {
            '==': lambda: self.df[column] == value,
            '!=': lambda: self.df[column] != value,
            '>': lambda: self.df[column] > value,
            '<': lambda: self.df[column] < value,
            '>=': lambda: self.df[column] >= value,
            '<=': lambda: self.df[column] <= value,
            'contains': lambda: self.df[column].str.contains(value, na=False),
            'isin': lambda: self.df[column].isin(value)
        }
        
        if condition not in conditions:
            raise ValueError(f"Invalid condition: {condition}")
        
        return self.df[conditions[condition]()]
    
    def group_stats(
        self,
        group_by: Union[str, List[str]],
        agg_column: str,
        agg_funcs: List[str] = ['mean', 'sum', 'count']
    ) -> pd.DataFrame:
        """
        Calculate grouped statistics.
        
        Args:
            group_by: Column(s) to group by
            agg_column: Column to aggregate
            agg_funcs: Aggregation functions to apply
            
        Returns:
            Grouped statistics DataFrame
        """
        self._check_loaded()
        return self.df.groupby(group_by)[agg_column].agg(agg_funcs)
    
    def export_report(
        self,
        output_path: str,
        include_data: bool = False
    ) -> None:
        """
        Export analysis report to a text file.
        
        Args:
            output_path: Path for the output report
            include_data: Whether to include raw data sample
        """
        self._check_loaded()
        
        result = self.analyze()
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CSV DATA ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"File: {self.filepath}\n")
            f.write(f"Shape: {result.shape[0]} rows × {result.shape[1]} columns\n")
            f.write(f"Memory Usage: {result.memory_usage:.2f} MB\n")
            f.write(f"Duplicate Rows: {result.duplicates}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("COLUMNS & DATA TYPES\n")
            f.write("-" * 40 + "\n")
            for col, dtype in result.dtypes.items():
                missing = result.missing_values.get(col, 0)
                f.write(f"  {col}: {dtype} (missing: {missing})\n")
            
            f.write("\n" + "-" * 40 + "\n")
            f.write("NUMERIC STATISTICS\n")
            f.write("-" * 40 + "\n")
            for col, stats in result.statistics.items():
                f.write(f"\n  {col}:\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"    {stat_name}: {stat_value:.4f}\n")
            
            if include_data:
                f.write("\n" + "-" * 40 + "\n")
                f.write("DATA SAMPLE (first 10 rows)\n")
                f.write("-" * 40 + "\n")
                f.write(self.df.head(10).to_string())
            
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Report exported to: {output_path}")
    
    def reset(self) -> 'CSVDataAnalyzer':
        """
        Reset DataFrame to original state (undo all cleaning).
        
        Returns:
            Self for method chaining
        """
        if self._original_df is not None:
            self.df = self._original_df.copy()
            logger.info("DataFrame reset to original state")
        return self
    
    def _check_loaded(self) -> None:
        """Check if data is loaded, raise error if not."""
        if self.df is None:
            raise RuntimeError("No data loaded. Call load() or from_dataframe() first.")
    
    def __repr__(self) -> str:
        if self.df is not None:
            return f"CSVDataAnalyzer(rows={len(self.df)}, cols={len(self.df.columns)})"
        return "CSVDataAnalyzer(no data loaded)"


def main():
    """Example usage of CSVDataAnalyzer."""
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice'],
        'age': [25, 30, None, 35, 28, 25],
        'salary': [50000, 60000, 55000, 70000, None, 50000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'IT'],
        'experience': [2, 5, 3, 8, 4, 2]
    })
    
    # Initialize and analyze
    analyzer = CSVDataAnalyzer()
    analyzer.from_dataframe(sample_data)
    
    # Get basic info
    print("Dataset Info:")
    print(analyzer.info())
    print()
    
    # Clean data
    analyzer.clean(drop_duplicates=True, handle_missing='fill')
    
    # Run analysis
    result = analyzer.analyze()
    print(f"Shape: {result.shape}")
    print(f"Duplicates removed: {result.duplicates}")
    print(f"\nStatistics for 'salary':")
    print(result.statistics.get('salary', {}))
    
    # Group statistics
    print("\nSalary by Department:")
    print(analyzer.group_stats('department', 'salary'))


if __name__ == "__main__":
    main()
