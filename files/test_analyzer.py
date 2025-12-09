"""
Unit tests for CSV Data Analyzer.
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.analyzer import CSVDataAnalyzer, AnalysisResult


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, None, 35, 28],
        'salary': [50000, 60000, 55000, 70000, None],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
    })


@pytest.fixture
def sample_csv(sample_df):
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_df.to_csv(f, index=False)
        filepath = f.name
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def analyzer(sample_df):
    """Create analyzer with sample data."""
    analyzer = CSVDataAnalyzer()
    analyzer.from_dataframe(sample_df)
    return analyzer


class TestCSVDataAnalyzer:
    """Test suite for CSVDataAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = CSVDataAnalyzer()
        assert analyzer.df is None
        assert analyzer.filepath is None
    
    def test_init_with_path(self):
        """Test initialization with filepath."""
        analyzer = CSVDataAnalyzer("test.csv")
        assert analyzer.filepath == Path("test.csv")
    
    def test_load_csv(self, sample_csv):
        """Test loading CSV file."""
        analyzer = CSVDataAnalyzer()
        analyzer.load(sample_csv)
        
        assert analyzer.df is not None
        assert len(analyzer.df) == 5
        assert 'name' in analyzer.df.columns
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        analyzer = CSVDataAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.load("nonexistent.csv")
    
    def test_load_no_filepath(self):
        """Test loading without filepath raises error."""
        analyzer = CSVDataAnalyzer()
        with pytest.raises(ValueError):
            analyzer.load()
    
    def test_from_dataframe(self, sample_df):
        """Test initializing from DataFrame."""
        analyzer = CSVDataAnalyzer()
        analyzer.from_dataframe(sample_df)
        
        assert analyzer.df is not None
        assert len(analyzer.df) == len(sample_df)
    
    def test_info(self, analyzer):
        """Test getting dataset info."""
        info = analyzer.info()
        
        assert info['rows'] == 5
        assert info['columns'] == 4
        assert 'name' in info['column_names']
        assert 'memory_mb' in info
    
    def test_clean_duplicates(self):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'a': [1, 1, 2, 3],
            'b': ['x', 'x', 'y', 'z']
        })
        analyzer = CSVDataAnalyzer()
        analyzer.from_dataframe(df)
        analyzer.clean(drop_duplicates=True)
        
        assert len(analyzer.df) == 3
    
    def test_clean_missing_drop(self, analyzer):
        """Test dropping missing values."""
        initial_len = len(analyzer.df)
        analyzer.clean(handle_missing='drop')
        
        assert len(analyzer.df) < initial_len
        assert analyzer.df.isnull().sum().sum() == 0
    
    def test_clean_missing_fill(self, analyzer):
        """Test filling missing values."""
        analyzer.clean(handle_missing='fill')
        
        assert analyzer.df['age'].isnull().sum() == 0
        assert analyzer.df['salary'].isnull().sum() == 0
    
    def test_analyze(self, analyzer):
        """Test comprehensive analysis."""
        result = analyzer.analyze()
        
        assert isinstance(result, AnalysisResult)
        assert result.shape == (5, 4)
        assert 'age' in result.statistics
        assert 'mean' in result.statistics['age']
    
    def test_get_summary(self, analyzer):
        """Test summary statistics."""
        summary = analyzer.get_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'age' in summary.columns
    
    def test_get_correlations(self, analyzer):
        """Test correlation matrix."""
        corr = analyzer.get_correlations()
        
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1]  # Square matrix
    
    def test_filter_equals(self, analyzer):
        """Test filtering with equality."""
        filtered = analyzer.filter('department', '==', 'IT')
        
        assert len(filtered) == 3
        assert all(filtered['department'] == 'IT')
    
    def test_filter_greater_than(self, analyzer):
        """Test filtering with greater than."""
        filtered = analyzer.filter('age', '>', 28)
        
        assert all(filtered['age'] > 28)
    
    def test_filter_contains(self, analyzer):
        """Test filtering with string contains."""
        filtered = analyzer.filter('name', 'contains', 'a')
        
        assert 'Charlie' in filtered['name'].values
    
    def test_filter_invalid_condition(self, analyzer):
        """Test filter with invalid condition."""
        with pytest.raises(ValueError):
            analyzer.filter('age', 'invalid', 25)
    
    def test_group_stats(self, analyzer):
        """Test grouped statistics."""
        stats = analyzer.group_stats('department', 'salary')
        
        assert isinstance(stats, pd.DataFrame)
        assert 'IT' in stats.index
    
    def test_reset(self, analyzer):
        """Test resetting to original state."""
        original_len = len(analyzer.df)
        
        # Make changes
        analyzer.clean(handle_missing='drop')
        modified_len = len(analyzer.df)
        
        # Reset
        analyzer.reset()
        
        assert len(analyzer.df) == original_len
        assert modified_len < original_len
    
    def test_export_report(self, analyzer, tmp_path):
        """Test exporting report."""
        output_path = tmp_path / "report.txt"
        analyzer.export_report(str(output_path))
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "CSV DATA ANALYSIS REPORT" in content
    
    def test_method_chaining(self, sample_df):
        """Test method chaining works."""
        result = (
            CSVDataAnalyzer()
            .from_dataframe(sample_df)
            .clean(drop_duplicates=True)
            .analyze()
        )
        
        assert isinstance(result, AnalysisResult)
    
    def test_repr(self, analyzer):
        """Test string representation."""
        repr_str = repr(analyzer)
        
        assert "CSVDataAnalyzer" in repr_str
        assert "rows=5" in repr_str
    
    def test_repr_empty(self):
        """Test repr for empty analyzer."""
        analyzer = CSVDataAnalyzer()
        assert "no data loaded" in repr(analyzer)


class TestAnalysisResult:
    """Test suite for AnalysisResult dataclass."""
    
    def test_analysis_result_fields(self, analyzer):
        """Test AnalysisResult has all expected fields."""
        result = analyzer.analyze()
        
        assert hasattr(result, 'shape')
        assert hasattr(result, 'columns')
        assert hasattr(result, 'dtypes')
        assert hasattr(result, 'missing_values')
        assert hasattr(result, 'statistics')
        assert hasattr(result, 'memory_usage')
        assert hasattr(result, 'duplicates')
    
    def test_statistics_content(self, analyzer):
        """Test statistics contain expected metrics."""
        result = analyzer.analyze()
        
        expected_stats = ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'skewness', 'kurtosis']
        
        for stat in expected_stats:
            assert stat in result.statistics.get('age', {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
