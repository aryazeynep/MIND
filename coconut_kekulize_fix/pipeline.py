#!/usr/bin/env python3
"""
COCONUT Kekulize Pipeline

This script provides a complete pipeline for analyzing, fixing, and filtering
the COCONUT dataset to resolve kekulize issues.

Author: AI Assistant
Date: September 2024
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import traceback
from typing import Dict, Any


class COCONUTKekulizePipeline:
    """Complete pipeline for COCONUT kekulize problem solving"""
    
    def __init__(self, input_sdf: str, output_dir: str, max_molecules: int = None):
        """
        Initialize pipeline
        
        Args:
            input_sdf: Path to input COCONUT SDF file
            output_dir: Directory to save all outputs
            max_molecules: Maximum molecules to process (for testing)
        """
        self.input_sdf = input_sdf
        self.output_dir = Path(output_dir)
        self.max_molecules = max_molecules
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "fixed").mkdir(exist_ok=True)
        (self.output_dir / "filtered").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Define file paths
        self.analysis_json = self.output_dir / "analysis" / "kekulize_analysis.json"
        self.analysis_report = self.output_dir / "reports" / "analysis_report.txt"
        self.fixed_sdf = self.output_dir / "fixed" / "coconut_fixed.sdf"
        self.fix_results = self.output_dir / "fixed" / "fix_results.json"
        self.fix_report = self.output_dir / "reports" / "fix_report.txt"
        self.filtered_sdf = self.output_dir / "filtered" / "coconut_filtered.sdf"
        self.filter_stats = self.output_dir / "filtered" / "filter_statistics.json"
        self.stats_dir = self.output_dir / "statistics"
        self.test_results = self.output_dir / "analysis" / "test_results.json"
        
        print(f"🚀 COCONUT Kekulize Pipeline Initialized")
        print(f"   Input: {self.input_sdf}")
        print(f"   Output Directory: {self.output_dir}")
        if self.max_molecules:
            print(f"   Max Molecules: {self.max_molecules}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        results = {}
        
        try:
            print(f"\n{'='*60}")
            print(f"🔍 STEP 1: ANALYZING KEKULIZE ISSUES")
            print(f"{'='*60}")
            results['analysis'] = self._run_analysis()
            
            print(f"\n{'='*60}")
            print(f"🔧 STEP 2: ATTEMPTING TO FIX ISSUES")
            print(f"{'='*60}")
            results['fixing'] = self._run_fixing()
            
            print(f"\n{'='*60}")
            print(f"🧹 STEP 3: FILTERING DATASET")
            print(f"{'='*60}")
            results['filtering'] = self._run_filtering()
            
            print(f"\n{'='*60}")
            print(f"📊 STEP 4: GENERATING STATISTICS")
            print(f"{'='*60}")
            results['statistics'] = self._run_statistics()
            
            print(f"\n{'='*60}")
            print(f"🧪 STEP 5: TESTING FIXES")
            print(f"{'='*60}")
            results['testing'] = self._run_testing()
            
            print(f"\n{'='*60}")
            print(f"📋 STEP 6: GENERATING FINAL REPORT")
            print(f"{'='*60}")
            self._generate_final_report(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _run_analysis(self) -> Dict[str, Any]:
        """Step 1: Analyze kekulize issues"""
        cmd = [
            sys.executable, "kekulize_analyzer.py",
            "--input", str(self.input_sdf),
            "--output", str(self.analysis_json),
            "--report", str(self.analysis_report)
        ]
        
        if self.max_molecules:
            cmd.extend(["--max-molecules", str(self.max_molecules)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            raise RuntimeError(f"Analysis failed: {result.stderr}")
        
        # Load results
        with open(self.analysis_json, 'r') as f:
            analysis_data = json.load(f)
        
        return {
            "success": True,
            "output_files": [str(self.analysis_json), str(self.analysis_report)],
            "statistics": analysis_data.get("statistics", {})
        }
    
    def _run_fixing(self) -> Dict[str, Any]:
        """Step 2: Attempt to fix kekulize issues"""
        cmd = [
            sys.executable, "kekulize_fixer.py",
            "--input", str(self.input_sdf),
            "--output", str(self.fixed_sdf),
            "--results", str(self.fix_results),
            "--report", str(self.fix_report)
        ]
        
        if self.max_molecules:
            cmd.extend(["--max-molecules", str(self.max_molecules)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            raise RuntimeError(f"Fixing failed: {result.stderr}")
        
        # Load results
        with open(self.fix_results, 'r') as f:
            fix_data = json.load(f)
        
        return {
            "success": True,
            "output_files": [str(self.fixed_sdf), str(self.fix_results), str(self.fix_report)],
            "statistics": fix_data.get("statistics", {})
        }
    
    def _run_filtering(self) -> Dict[str, Any]:
        """Step 3: Filter dataset to remove problematic molecules"""
        cmd = [
            sys.executable, "coconut_filter.py",
            "--input", str(self.input_sdf),
            "--output", str(self.filtered_sdf),
            "--statistics", str(self.filter_stats),
            "--max-atoms", "150",
            "--min-atoms", "5",
            "--max-weight", "1000.0",
            "--min-weight", "50.0"
        ]
        
        if self.max_molecules:
            cmd.extend(["--max-molecules", str(self.max_molecules)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            raise RuntimeError(f"Filtering failed: {result.stderr}")
        
        # Load results
        with open(self.filter_stats, 'r') as f:
            filter_data = json.load(f)
        
        return {
            "success": True,
            "output_files": [str(self.filtered_sdf), str(self.filter_stats)],
            "statistics": filter_data
        }
    
    def _run_statistics(self) -> Dict[str, Any]:
        """Step 4: Generate comprehensive statistics"""
        cmd = [
            sys.executable, "statistics_reporter.py",
            "--input", str(self.filtered_sdf),  # Use filtered dataset
            "--output-dir", str(self.stats_dir),
            "--generate-plots",
            "--export-csv"
        ]
        
        if self.max_molecules:
            cmd.extend(["--max-molecules", str(min(self.max_molecules, 10000))])  # Limit for stats
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print(f"Warning: Statistics generation failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        return {
            "success": True,
            "output_dir": str(self.stats_dir)
        }
    
    def _run_testing(self) -> Dict[str, Any]:
        """Step 5: Test fix strategies"""
        cmd = [
            sys.executable, "test_kekulize_fixes.py",
            "--mode", "real",
            "--input", str(self.input_sdf),
            "--output", str(self.test_results),
            "--max-molecules", str(min(self.max_molecules or 1000, 1000))
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print(f"Warning: Testing failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        # Load results
        try:
            with open(self.test_results, 'r') as f:
                test_data = json.load(f)
            
            return {
                "success": True,
                "output_file": str(self.test_results),
                "results": test_data
            }
        except:
            return {"success": False, "error": "Could not load test results"}
    
    def _generate_final_report(self, results: Dict[str, Any]) -> None:
        """Generate final comprehensive report"""
        report_path = self.output_dir / "reports" / "final_pipeline_report.txt"
        
        report = f"""
🧬 COCONUT Kekulize Problem Solving - Final Pipeline Report
{'='*70}

📋 PIPELINE OVERVIEW
Input dataset: {self.input_sdf}
Output directory: {self.output_dir}
Processing date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Max molecules processed: {self.max_molecules or 'All'}

{'='*70}
🔍 ANALYSIS RESULTS
{'='*70}
"""
        
        if results.get('analysis', {}).get('success'):
            analysis_stats = results['analysis']['statistics']
            total_mols = analysis_stats.get('total_molecules', 0)
            success_rate = analysis_stats.get('success_rates', {}).get('both_success', 0)
            
            report += f"""
Total molecules analyzed: {total_mols:,}
Kekulize + Sanitize success rate: {success_rate:.1%}
Problematic molecules: {(1-success_rate)*100:.1f}%

Error Distribution:
"""
            for error_type, count in analysis_stats.get('error_distribution', {}).items():
                percentage = (count / total_mols) * 100 if total_mols > 0 else 0
                report += f"  {error_type}: {count:,} ({percentage:.1f}%)\n"
        else:
            report += "Analysis step failed or incomplete.\n"
        
        report += f"""
{'='*70}
🔧 FIXING RESULTS
{'='*70}
"""
        
        if results.get('fixing', {}).get('success'):
            fix_stats = results['fixing']['statistics']
            report += f"""
Total molecules processed: {fix_stats.get('total_molecules', 0):,}
Successfully fixed: {fix_stats.get('successful_fixes', 0):,}
Fix success rate: {fix_stats.get('success_rate', 0):.1%}

Most effective strategies:
"""
            for strategy, count in fix_stats.get('strategy_effectiveness', {}).items():
                report += f"  {strategy}: {count:,} fixes\n"
        else:
            report += "Fixing step failed or incomplete.\n"
        
        report += f"""
{'='*70}
🧹 FILTERING RESULTS  
{'='*70}
"""
        
        if results.get('filtering', {}).get('success'):
            filter_stats = results['filtering']['statistics']
            total_processed = filter_stats.get('total_molecules', 0)
            passed = filter_stats.get('passed_molecules', 0)
            
            report += f"""
Total molecules processed: {total_processed:,}
Molecules passed filters: {passed:,} ({passed/total_processed:.1%} if total_processed > 0 else 0)
Molecules filtered out: {total_processed-passed:,}

Filter reasons:
"""
            for reason, count in filter_stats.get('filter_reasons', {}).items():
                percentage = (count / total_processed) * 100 if total_processed > 0 else 0
                report += f"  {reason}: {count:,} ({percentage:.1f}%)\n"
        else:
            report += "Filtering step failed or incomplete.\n"
        
        report += f"""
{'='*70}
🧪 TESTING RESULTS
{'='*70}
"""
        
        if results.get('testing', {}).get('success'):
            test_data = results['testing']['results']
            report += f"""
Fix testing completed successfully.
Problematic molecules tested: {test_data.get('problematic_molecules_tested', 0):,}
Fix success rate on problematic molecules: {test_data.get('fix_success_rate', 0):.1%}
"""
        else:
            report += "Testing step failed or incomplete.\n"
        
        report += f"""
{'='*70}
📁 OUTPUT FILES
{'='*70}

Analysis:
  - {self.analysis_json}
  - {self.analysis_report}

Fixing:
  - {self.fixed_sdf}
  - {self.fix_results}
  - {self.fix_report}

Filtering:
  - {self.filtered_sdf} (RECOMMENDED FOR TRAINING)
  - {self.filter_stats}

Statistics:
  - {self.stats_dir}/

Testing:
  - {self.test_results}

{'='*70}
🎯 RECOMMENDATIONS
{'='*70}

1. Use the filtered dataset ({self.filtered_sdf}) for training as it contains only 
   molecules that can be properly kekulized and sanitized.

2. Consider the fix success rate when deciding whether to include fixed molecules
   in your training dataset.

3. Review the error patterns to understand the types of chemical issues present
   in the original COCONUT dataset.

4. The statistics directory contains comprehensive molecular property analysis
   that can guide model architecture decisions.

5. Test your model training pipeline with a small subset first to validate
   compatibility with the processed dataset.
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Final report saved to {report_path}")
        
        # Print summary to console
        print(f"""
✅ PIPELINE COMPLETE!

📁 Key Output Files:
  • Filtered Dataset: {self.filtered_sdf}
  • Final Report: {report_path}
  • Statistics: {self.stats_dir}/

🎯 Next Steps:
  1. Review the final report
  2. Use the filtered dataset for training
  3. Check statistics for dataset insights
""")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Complete pipeline for solving COCONUT kekulize issues"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input COCONUT SDF file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory to save all pipeline outputs"
    )
    parser.add_argument(
        "--max-molecules", "-n",
        type=int,
        help="Maximum number of molecules to process (for testing the pipeline)"
    )
    parser.add_argument(
        "--step",
        choices=["analysis", "fixing", "filtering", "statistics", "testing", "all"],
        default="all",
        help="Run specific step only (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        # Check input file exists
        if not os.path.exists(args.input):
            print(f"❌ Error: Input SDF file not found: {args.input}")
            sys.exit(1)
        
        # Initialize pipeline
        pipeline = COCONUTKekulizePipeline(
            input_sdf=args.input,
            output_dir=args.output_dir,
            max_molecules=args.max_molecules
        )
        
        # Run pipeline
        if args.step == "all":
            results = pipeline.run_complete_pipeline()
        else:
            # Run individual step
            step_methods = {
                "analysis": pipeline._run_analysis,
                "fixing": pipeline._run_fixing,
                "filtering": pipeline._run_filtering,
                "statistics": pipeline._run_statistics,
                "testing": pipeline._run_testing
            }
            
            results = {args.step: step_methods[args.step]()}
            print(f"✅ {args.step.title()} step completed successfully")
        
        if "error" in results:
            print(f"❌ Pipeline failed: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if not available
    main()