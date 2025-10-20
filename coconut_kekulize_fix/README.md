# COCONUT Kekulize Problem Solver

This folder contains tools and scripts to address kekulize and aromaticity issues in the COCONUT (Collection of Open Natural Products) dataset.

## Problem Description

The COCONUT dataset contains natural product molecules that often have complex aromatic systems. When processing these molecules with RDKit, kekulization failures occur due to:

1. **Invalid valences**: Atoms with incorrect formal charges or valence states
2. **Aromaticity perception issues**: Complex polycyclic aromatic systems that RDKit cannot kekulize
3. **Sanitization failures**: Molecules that fail RDKit's sanitization process
4. **Radical species**: Molecules with unpaired electrons that cause valence issues

## Tools Provided

- `kekulize_analyzer.py` - Analyze kekulize failures in COCONUT dataset
- `kekulize_fixer.py` - Attempt to fix kekulize issues with various strategies
- `coconut_filter.py` - Filter COCONUT dataset to remove problematic molecules
- `test_kekulize_fixes.py` - Test suite for validating fixes
- `statistics_reporter.py` - Generate detailed statistics on dataset quality

## Usage

1. First, analyze the dataset for kekulize issues:
```bash
python kekulize_analyzer.py --input /path/to/coconut.sdf --output analysis_report.json
```

2. Attempt to fix the issues:
```bash
python kekulize_fixer.py --input /path/to/coconut.sdf --output fixed_coconut.sdf
```

3. Filter out unfixable molecules:
```bash
python coconut_filter.py --input /path/to/coconut.sdf --output clean_coconut.sdf
```

## Strategies Implemented

- Formal charge correction
- Explicit hydrogen addition/removal
- Alternative aromaticity models
- Valence adjustment
- Radical neutralization
- Sanitization alternatives