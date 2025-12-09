#!/usr/bin/env python3
"""
Command Line Interface for CSV Data Analyzer.
Run: python -m src.cli --help
"""

import argparse
import sys
from pathlib import Path

from .analyzer import CSVDataAnalyzer


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='csv-analyzer',
        description='Analyze CSV files with ease - Day 1 of 100 Days AI/ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  csv-analyzer data.csv                    # Basic analysis
  csv-analyzer data.csv --clean            # Clean and analyze
  csv-analyzer data.csv --output report.txt  # Save report
  csv-analyzer data.csv --stats salary     # Stats for specific column
        '''
    )
    
    parser.add_argument(
        'file',
        type=str,
        help='Path to CSV file to analyze'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for the report'
    )
    
    parser.add_argument(
        '-c', '--clean',
        action='store_true',
        help='Clean data (remove duplicates, handle missing values)'
    )
    
    parser.add_argument(
        '--missing',
        choices=['keep', 'drop', 'fill', 'ffill', 'bfill'],
        default='keep',
        help='How to handle missing values (default: keep)'
    )
    
    parser.add_argument(
        '-s', '--stats',
        type=str,
        help='Show detailed statistics for specific column'
    )
    
    parser.add_argument(
        '--head',
        type=int,
        default=0,
        help='Show first N rows of data'
    )
    
    parser.add_argument(
        '--corr',
        action='store_true',
        help='Show correlation matrix'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show basic dataset information only'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser


def main(args=None):
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(args)
    
    # Validate file exists
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Initialize analyzer
        analyzer = CSVDataAnalyzer(str(filepath))
        analyzer.load()
        
        # Clean if requested
        if args.clean:
            analyzer.clean(
                drop_duplicates=True,
                handle_missing=args.missing
            )
            if args.verbose:
                print("✓ Data cleaned")
        
        # Info only mode
        if args.info:
            info = analyzer.info()
            print("\n📊 Dataset Information")
            print("=" * 40)
            print(f"  Rows: {info['rows']:,}")
            print(f"  Columns: {info['columns']}")
            print(f"  Memory: {info['memory_mb']:.2f} MB")
            print(f"\n  Columns: {', '.join(info['column_names'])}")
            return
        
        # Show head
        if args.head > 0:
            print(f"\n📋 First {args.head} rows:")
            print(analyzer.df.head(args.head).to_string())
            print()
        
        # Full analysis
        result = analyzer.analyze()
        
        print("\n" + "=" * 50)
        print("📈 CSV DATA ANALYSIS REPORT")
        print("=" * 50)
        
        print(f"\n📁 File: {filepath.name}")
        print(f"📊 Shape: {result.shape[0]:,} rows × {result.shape[1]} columns")
        print(f"💾 Memory: {result.memory_usage:.2f} MB")
        print(f"🔄 Duplicates: {result.duplicates}")
        
        # Missing values summary
        total_missing = sum(result.missing_values.values())
        if total_missing > 0:
            print(f"\n⚠️  Missing Values: {total_missing}")
            for col, count in result.missing_values.items():
                if count > 0:
                    print(f"   - {col}: {count}")
        
        # Column-specific stats
        if args.stats:
            if args.stats in result.statistics:
                print(f"\n📊 Statistics for '{args.stats}':")
                print("-" * 30)
                for stat, value in result.statistics[args.stats].items():
                    print(f"  {stat:12}: {value:,.4f}")
            else:
                print(f"\n⚠️  Column '{args.stats}' not found or not numeric")
                print(f"   Available numeric columns: {list(result.statistics.keys())}")
        else:
            # Show summary for all numeric columns
            if result.statistics:
                print("\n📊 Numeric Column Summary:")
                print("-" * 40)
                for col, stats in result.statistics.items():
                    print(f"\n  {col}:")
                    print(f"    mean: {stats['mean']:,.2f} | "
                          f"std: {stats['std']:,.2f} | "
                          f"range: [{stats['min']:,.2f}, {stats['max']:,.2f}]")
        
        # Correlation matrix
        if args.corr:
            print("\n🔗 Correlation Matrix:")
            print("-" * 40)
            corr = analyzer.get_correlations()
            print(corr.round(3).to_string())
        
        # Export report
        if args.output:
            analyzer.export_report(args.output, include_data=args.head > 0)
            print(f"\n✅ Report saved to: {args.output}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
