#!/bin/bash
"""
Example Usage Scripts for COCONUT Kekulize Fix Tools

This script demonstrates how to use the various tools in the coconut_kekulize_fix package.
"""

# Make scripts executable
chmod +x *.py

echo "🧬 COCONUT Kekulize Fix Tools - Usage Examples"
echo "=============================================="

echo ""
echo "📋 AVAILABLE TOOLS:"
echo "1. kekulize_analyzer.py  - Analyze kekulize failures"
echo "2. kekulize_fixer.py     - Attempt to fix kekulize issues" 
echo "3. coconut_filter.py     - Filter out problematic molecules"
echo "4. statistics_reporter.py - Generate dataset statistics"
echo "5. test_kekulize_fixes.py - Test fix strategies"
echo "6. pipeline.py           - Run complete pipeline"

echo ""
echo "💡 EXAMPLE USAGE:"
echo ""

echo "# 1. Quick analysis of a small subset (for testing):"
echo "python kekulize_analyzer.py --input /path/to/coconut.sdf --output analysis.json --max-molecules 1000"
echo ""

echo "# 2. Fix kekulize issues:"
echo "python kekulize_fixer.py --input /path/to/coconut.sdf --output fixed_coconut.sdf --max-molecules 1000"
echo ""

echo "# 3. Filter dataset to remove problematic molecules:"
echo "python coconut_filter.py --input /path/to/coconut.sdf --output clean_coconut.sdf --max-atoms 150"
echo ""

echo "# 4. Generate comprehensive statistics:"
echo "python statistics_reporter.py --input /path/to/coconut.sdf --output-dir stats/ --generate-plots"
echo ""

echo "# 5. Test fix strategies:"
echo "python test_kekulize_fixes.py --mode real --input /path/to/coconut.sdf --max-molecules 100"
echo ""

echo "# 6. Run complete pipeline (RECOMMENDED):"
echo "python pipeline.py --input /path/to/coconut.sdf --output-dir results/ --max-molecules 5000"
echo ""

echo "🎯 FOR YOUR CURRENT SETUP:"
echo ""
echo "Based on your COCONUT adapter, your SDF file should be at:"
echo "/path/to/data/COCONUT/coconut_sdf_3d-09-2025.sdf"
echo ""
echo "To run a complete test with 5000 molecules:"
echo "python pipeline.py \\"
echo "  --input /path/to/data/COCONUT/coconut_sdf_3d-09-2025.sdf \\"
echo "  --output-dir ../coconut_kekulize_results/ \\"
echo "  --max-molecules 5000"
echo ""

echo "✨ The pipeline will:"
echo "  • Analyze kekulize issues"
echo "  • Attempt to fix problems"
echo "  • Filter out unfixable molecules"
echo "  • Generate statistics and plots"
echo "  • Test fix effectiveness"
echo "  • Provide a clean dataset for training"

echo ""
echo "📄 Check README.md for more detailed information!"