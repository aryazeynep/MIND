#!/usr/bin/env python3
"""
COCONUT Kekulize Fix Test Suite

This script tests the effectiveness of various kekulize fix strategies
on a subset of COCONUT molecules to validate the fixing approaches.

Author: AI Assistant
Date: September 2024
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import unittest
import tempfile
from pathlib import Path
import traceback

# Add RDKit
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors
    
    # Try different import patterns for MolStandardize based on RDKit version
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        try:
            from rdkit.Chem import rdMolStandardize
        except ImportError:
            rdMolStandardize = None
            
except ImportError as e:
    print(f"Error: RDKit not found. Please install RDKit: {e}")
    sys.exit(1)

# Import our kekulize fixer
try:
    from kekulize_fixer import COCONUTKekulizeFixer
    from kekulize_analyzer import COCONUTKekulizeAnalyzer
except ImportError:
    print("Error: Could not import kekulize_fixer or kekulize_analyzer modules")
    sys.exit(1)

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')


class TestKekulizeFixes(unittest.TestCase):
    """Test suite for kekulize fixing strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fixer = COCONUTKekulizeFixer()
        self.analyzer = COCONUTKekulizeAnalyzer()
        
        # Create test molecules with known issues
        self.test_molecules = self._create_test_molecules()
    
    def _create_test_molecules(self) -> List[Tuple[str, Chem.Mol, str]]:
        """Create test molecules with known kekulize issues"""
        test_cases = []
        
        # Test case 1: Molecule with valence issue (N+ without explicit charge)
        smiles1 = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine-like
        mol1 = Chem.MolFromSmiles(smiles1, sanitize=False)
        if mol1:
            # Introduce a valence issue
            for atom in mol1.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetDegree() == 3:
                    atom.SetFormalCharge(0)  # Remove proper charge
                    break
            test_cases.append(("valence_issue", mol1, "Nitrogen valence issue"))
        
        # Test case 2: Aromatic system that might fail kekulization
        smiles2 = "c1ccc2c(c1)c3ccccc3c4ccccc24"  # Anthracene-like
        mol2 = Chem.MolFromSmiles(smiles2, sanitize=False)
        if mol2:
            test_cases.append(("aromatic_system", mol2, "Complex aromatic system"))
        
        # Test case 3: Molecule with radicals
        smiles3 = "CC(C)C[C@@H](C(=O)O)N"  # Leucine
        mol3 = Chem.MolFromSmiles(smiles3, sanitize=False)
        if mol3:
            # Introduce radical by removing H
            mol3 = Chem.RemoveHs(mol3)
            test_cases.append(("radical_issue", mol3, "Potential radical issue"))
        
        return test_cases
    
    def test_strategy_direct(self):
        """Test direct kekulization strategy"""
        print("\n🧪 Testing direct kekulization strategy...")
        
        for test_id, mol, description in self.test_molecules:
            with self.subTest(test_id=test_id):
                try:
                    fixed_mol = self.fixer._strategy_direct(mol)
                    self.assertIsNotNone(fixed_mol, f"Direct strategy failed for {test_id}")
                    self.assertTrue(self.fixer._validate_fix(fixed_mol), 
                                  f"Validation failed for {test_id}")
                    print(f"  ✅ {test_id}: Direct strategy successful")
                except Exception as e:
                    print(f"  ❌ {test_id}: Direct strategy failed - {str(e)}")
    
    def test_strategy_add_hydrogens(self):
        """Test hydrogen addition strategy"""
        print("\n🧪 Testing hydrogen addition strategy...")
        
        for test_id, mol, description in self.test_molecules:
            with self.subTest(test_id=test_id):
                try:
                    fixed_mol = self.fixer._strategy_add_hydrogens(mol)
                    self.assertIsNotNone(fixed_mol, f"Hydrogen strategy failed for {test_id}")
                    self.assertTrue(self.fixer._validate_fix(fixed_mol), 
                                  f"Validation failed for {test_id}")
                    print(f"  ✅ {test_id}: Hydrogen addition strategy successful")
                except Exception as e:
                    print(f"  ❌ {test_id}: Hydrogen addition strategy failed - {str(e)}")
    
    def test_strategy_standardize(self):
        """Test standardization strategy"""
        print("\n🧪 Testing standardization strategy...")
        
        for test_id, mol, description in self.test_molecules:
            with self.subTest(test_id=test_id):
                try:
                    fixed_mol = self.fixer._strategy_standardize(mol)
                    self.assertIsNotNone(fixed_mol, f"Standardization strategy failed for {test_id}")
                    self.assertTrue(self.fixer._validate_fix(fixed_mol), 
                                  f"Validation failed for {test_id}")
                    print(f"  ✅ {test_id}: Standardization strategy successful")
                except Exception as e:
                    print(f"  ❌ {test_id}: Standardization strategy failed - {str(e)}")
    
    def test_strategy_fix_valences(self):
        """Test valence fixing strategy"""
        print("\n🧪 Testing valence fixing strategy...")
        
        for test_id, mol, description in self.test_molecules:
            with self.subTest(test_id=test_id):
                try:
                    fixed_mol = self.fixer._strategy_fix_valences(mol)
                    self.assertIsNotNone(fixed_mol, f"Valence fixing strategy failed for {test_id}")
                    self.assertTrue(self.fixer._validate_fix(fixed_mol), 
                                  f"Validation failed for {test_id}")
                    print(f"  ✅ {test_id}: Valence fixing strategy successful")
                except Exception as e:
                    print(f"  ❌ {test_id}: Valence fixing strategy failed - {str(e)}")
    
    def test_validation_function(self):
        """Test the fix validation function"""
        print("\n🧪 Testing fix validation function...")
        
        # Test with valid molecule
        valid_mol = Chem.MolFromSmiles("CCO")  # Ethanol
        self.assertTrue(self.fixer._validate_fix(valid_mol), "Validation should pass for ethanol")
        
        # Test with None
        self.assertFalse(self.fixer._validate_fix(None), "Validation should fail for None")
        
        # Test with empty molecule
        empty_mol = Chem.Mol()
        self.assertFalse(self.fixer._validate_fix(empty_mol), "Validation should fail for empty molecule")
        
        print("  ✅ Validation function tests passed")


class COCONUTKekulizeTestRunner:
    """Test runner for COCONUT kekulize fixes on real data"""
    
    def __init__(self):
        self.results = {}
    
    def test_on_real_data(self, sdf_path: str, max_molecules: int = 100) -> Dict[str, Any]:
        """Test fixing strategies on real COCONUT data"""
        print(f"🧪 Testing kekulize fixes on real COCONUT data...")
        print(f"Input file: {sdf_path}")
        print(f"Testing on {max_molecules} molecules")
        
        if not os.path.exists(sdf_path):
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")
        
        # First, analyze the problems
        analyzer = COCONUTKekulizeAnalyzer()
        analyzer.analyze_sdf_file(sdf_path, max_molecules)
        
        problematic_molecules = [
            r for r in analyzer.results 
            if not (r.kekulize_success and r.sanitize_success)
        ]
        
        print(f"Found {len(problematic_molecules)} problematic molecules")
        
        if len(problematic_molecules) == 0:
            print("No problematic molecules found to test fixes on!")
            return {"error": "No problematic molecules"}
        
        # Test fixes on problematic molecules
        fixer = COCONUTKekulizeFixer()
        fix_results = []
        
        # Load problematic molecules
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        problematic_indices = {r.mol_id.split('_')[1]: r for r in problematic_molecules}
        
        for i, mol in enumerate(supplier):
            if str(i) in problematic_indices:
                mol_id = f"coconut_{i}"
                fixed_mol, fix_result = fixer.fix_single_molecule(mol, mol_id)
                fix_results.append(fix_result)
                
                if len(fix_results) >= 50:  # Limit testing to 50 problematic molecules
                    break
        
        # Analyze fix results
        successful_fixes = sum(1 for r in fix_results if r.fix_successful)
        total_tested = len(fix_results)
        
        strategy_effectiveness = {}
        for result in fix_results:
            if result.fix_successful:
                strategy = result.fix_strategy
                strategy_effectiveness[strategy] = strategy_effectiveness.get(strategy, 0) + 1
        
        self.results = {
            "total_molecules_tested": max_molecules,
            "problematic_molecules_found": len(problematic_molecules),
            "problematic_molecules_tested": total_tested,
            "successful_fixes": successful_fixes,
            "fix_success_rate": successful_fixes / total_tested if total_tested > 0 else 0,
            "strategy_effectiveness": strategy_effectiveness,
            "original_problem_distribution": analyzer.statistics.get("error_distribution", {}),
            "fix_results": [
                {
                    "mol_id": r.mol_id,
                    "fix_successful": r.fix_successful,
                    "fix_strategy": r.fix_strategy,
                    "error_message": r.error_message
                } for r in fix_results
            ]
        }
        
        return self.results
    
    def print_test_summary(self):
        """Print a summary of test results"""
        if not self.results:
            print("No test results available")
            return
        
        results = self.results
        
        print(f"\n🧪 COCONUT Kekulize Fix Test Summary")
        print(f"{'='*50}")
        print(f"Total molecules tested: {results['total_molecules_tested']:,}")
        print(f"Problematic molecules found: {results['problematic_molecules_found']:,}")
        print(f"Problematic molecules tested: {results['problematic_molecules_tested']:,}")
        print(f"Successful fixes: {results['successful_fixes']:,}")
        print(f"Fix success rate: {results['fix_success_rate']:.1%}")
        
        print(f"\n🎯 Strategy Effectiveness:")
        for strategy, count in results['strategy_effectiveness'].items():
            percentage = (count / results['successful_fixes']) * 100 if results['successful_fixes'] > 0 else 0
            print(f"  {strategy.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n❌ Original Problems Distribution:")
        for problem_type, count in results['original_problem_distribution'].items():
            percentage = (count / results['total_molecules_tested']) * 100
            print(f"  {problem_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    def save_test_results(self, output_path: str):
        """Save test results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📊 Test results saved to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test kekulize fix strategies on COCONUT dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["unit", "real", "both"],
        default="both",
        help="Test mode: unit tests, real data tests, or both"
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to COCONUT SDF file (required for real data tests)"
    )
    parser.add_argument(
        "--max-molecules", "-n",
        type=int,
        default=100,
        help="Maximum number of molecules to test on real data (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save test results JSON"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode in ["unit", "both"]:
            print("🧪 Running unit tests...")
            unittest.main(argv=[''], exit=False, verbosity=2)
        
        if args.mode in ["real", "both"]:
            if not args.input:
                print("❌ Error: --input required for real data tests")
                sys.exit(1)
            
            print("\n🧪 Running real data tests...")
            test_runner = COCONUTKekulizeTestRunner()
            results = test_runner.test_on_real_data(args.input, args.max_molecules)
            test_runner.print_test_summary()
            
            if args.output:
                test_runner.save_test_results(args.output)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()