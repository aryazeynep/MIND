#!/usr/bin/env python3
"""
COCONUT Kekulize Problem Analyzer

This script analyzes the COCONUT dataset to identify and categorize kekulize failures,
providing detailed statistics and problem classification for natural product molecules.

Author: AI Assistant
Date: September 2024
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
import traceback

# Add RDKit and other dependencies
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen
    
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

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')


@dataclass
class KekulizeAnalysis:
    """Data class to store analysis results for a molecule"""
    mol_id: str
    smiles_original: Optional[str]
    smiles_sanitized: Optional[str]
    kekulize_success: bool
    sanitize_success: bool
    error_type: str
    error_message: str
    num_atoms: int
    num_heavy_atoms: int
    num_aromatic_atoms: int
    num_aromatic_rings: int
    molecular_formula: str
    molecular_weight: float
    logp: Optional[float]
    fix_strategy: Optional[str] = None
    fix_successful: bool = False


class COCONUTKekulizeAnalyzer:
    """Comprehensive analyzer for kekulize issues in COCONUT dataset"""
    
    def __init__(self):
        self.results: List[KekulizeAnalysis] = []
        self.statistics: Dict[str, Any] = {}
        
    def analyze_sdf_file(self, sdf_path: str, max_molecules: Optional[int] = None) -> None:
        """Analyze an entire SDF file for kekulize issues"""
        print(f"📊 Starting COCONUT kekulize analysis...")
        print(f"Input file: {sdf_path}")
        
        if not os.path.exists(sdf_path):
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")
        
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        total_molecules = 0
        processed_molecules = 0
        
        # Estimate total molecules for progress tracking
        print("Estimating total molecules...")
        temp_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        total_count = sum(1 for mol in temp_supplier if mol is not None)
        print(f"Found {total_count} molecules to process")
        
        if max_molecules:
            total_count = min(total_count, max_molecules)
            print(f"Limiting analysis to {max_molecules} molecules")
        
        for i, mol in enumerate(supplier):
            if max_molecules and processed_molecules >= max_molecules:
                break
                
            total_molecules += 1
            
            if mol is None:
                print(f"Warning: Molecule {i} could not be loaded from SDF")
                continue
            
            # Progress update
            if processed_molecules % 1000 == 0:
                print(f"Processing molecule {processed_molecules}/{total_count}...")
            
            analysis = self.analyze_single_molecule(mol, f"coconut_{i}")
            self.results.append(analysis)
            processed_molecules += 1
        
        print(f"✅ Analysis complete. Processed {processed_molecules} molecules")
        self._compute_statistics()
    
    def analyze_single_molecule(self, mol: Chem.Mol, mol_id: str) -> KekulizeAnalysis:
        """Analyze a single molecule for kekulize issues"""
        
        # Initialize analysis object
        analysis = KekulizeAnalysis(
            mol_id=mol_id,
            smiles_original=None,
            smiles_sanitized=None,
            kekulize_success=False,
            sanitize_success=False,
            error_type="unknown",
            error_message="",
            num_atoms=mol.GetNumAtoms(),
            num_heavy_atoms=mol.GetNumHeavyAtoms(),
            num_aromatic_atoms=0,
            num_aromatic_rings=0,
            molecular_formula="",
            molecular_weight=0.0,
            logp=None
        )
        
        try:
            # Get original SMILES (before any processing)
            try:
                analysis.smiles_original = Chem.MolToSmiles(mol, canonical=True)
            except:
                analysis.smiles_original = "Could not generate"
            
            # Basic molecular properties
            analysis.molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            analysis.molecular_weight = Descriptors.MolWt(mol)
            
            # Count aromatic atoms and rings before sanitization
            try:
                analysis.num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                analysis.num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            except:
                pass
            
            # Test kekulization
            kekulize_success, kekulize_error = self._test_kekulization(mol)
            analysis.kekulize_success = kekulize_success
            
            # Test sanitization
            sanitize_success, sanitize_error = self._test_sanitization(mol)
            analysis.sanitize_success = sanitize_success
            
            # Determine primary error type
            if not kekulize_success and not sanitize_success:
                analysis.error_type = "both_kekulize_and_sanitize"
                analysis.error_message = f"Kekulize: {kekulize_error}; Sanitize: {sanitize_error}"
            elif not kekulize_success:
                analysis.error_type = "kekulize_only"
                analysis.error_message = kekulize_error
            elif not sanitize_success:
                analysis.error_type = "sanitize_only" 
                analysis.error_message = sanitize_error
            else:
                analysis.error_type = "success"
                analysis.error_message = ""
                
                # If successful, get sanitized SMILES and LogP
                try:
                    mol_copy = Chem.Mol(mol)
                    Chem.SanitizeMol(mol_copy)
                    analysis.smiles_sanitized = Chem.MolToSmiles(mol_copy, canonical=True)
                    analysis.logp = Crippen.MolLogP(mol_copy)
                except:
                    pass
        
        except Exception as e:
            analysis.error_type = "analysis_error"
            analysis.error_message = f"Analysis failed: {str(e)}"
        
        return analysis
    
    def _test_kekulization(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Test if a molecule can be kekulized"""
        try:
            mol_copy = Chem.Mol(mol)
            Chem.Kekulize(mol_copy, clearAromaticFlags=True)
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def _test_sanitization(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Test if a molecule can be sanitized"""
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def _compute_statistics(self) -> None:
        """Compute comprehensive statistics from analysis results"""
        total_molecules = len(self.results)
        
        if total_molecules == 0:
            self.statistics = {"error": "No molecules analyzed"}
            return
        
        # Basic success rates
        kekulize_success_count = sum(1 for r in self.results if r.kekulize_success)
        sanitize_success_count = sum(1 for r in self.results if r.sanitize_success)
        both_success_count = sum(1 for r in self.results if r.kekulize_success and r.sanitize_success)
        
        # Error type distribution
        error_types = Counter([r.error_type for r in self.results])
        
        # Molecular property statistics
        num_atoms_stats = [r.num_atoms for r in self.results]
        num_heavy_atoms_stats = [r.num_heavy_atoms for r in self.results]
        aromatic_atoms_stats = [r.num_aromatic_atoms for r in self.results]
        aromatic_rings_stats = [r.num_aromatic_rings for r in self.results]
        mol_weights = [r.molecular_weight for r in self.results if r.molecular_weight > 0]
        
        self.statistics = {
            "total_molecules": total_molecules,
            "success_rates": {
                "kekulize_success": kekulize_success_count / total_molecules,
                "sanitize_success": sanitize_success_count / total_molecules,  
                "both_success": both_success_count / total_molecules,
                "total_problematic": (total_molecules - both_success_count) / total_molecules
            },
            "error_distribution": dict(error_types),
            "molecular_properties": {
                "avg_atoms": sum(num_atoms_stats) / len(num_atoms_stats) if num_atoms_stats else 0,
                "avg_heavy_atoms": sum(num_heavy_atoms_stats) / len(num_heavy_atoms_stats) if num_heavy_atoms_stats else 0,
                "avg_aromatic_atoms": sum(aromatic_atoms_stats) / len(aromatic_atoms_stats) if aromatic_atoms_stats else 0,
                "avg_aromatic_rings": sum(aromatic_rings_stats) / len(aromatic_rings_stats) if aromatic_rings_stats else 0,
                "avg_molecular_weight": sum(mol_weights) / len(mol_weights) if mol_weights else 0,
                "max_atoms": max(num_atoms_stats) if num_atoms_stats else 0,
                "min_atoms": min(num_atoms_stats) if num_atoms_stats else 0
            }
        }
        
        # Add detailed error analysis
        self._analyze_error_patterns()
    
    def _analyze_error_patterns(self) -> None:
        """Analyze common error patterns in detail"""
        error_patterns = defaultdict(list)
        
        for result in self.results:
            if result.error_type != "success":
                # Extract key phrases from error messages
                error_msg = result.error_message.lower()
                
                if "valence" in error_msg:
                    error_patterns["valence_errors"].append(result.mol_id)
                if "kekulize" in error_msg:
                    error_patterns["kekulize_errors"].append(result.mol_id)
                if "aromatic" in error_msg:
                    error_patterns["aromatic_errors"].append(result.mol_id)
                if "radical" in error_msg:
                    error_patterns["radical_errors"].append(result.mol_id)
                if "sanitiz" in error_msg:
                    error_patterns["sanitization_errors"].append(result.mol_id)
        
        self.statistics["error_patterns"] = {
            pattern: len(mol_ids) for pattern, mol_ids in error_patterns.items()
        }
    
    def save_results(self, output_path: str) -> None:
        """Save analysis results to JSON file"""
        output_data = {
            "statistics": self.statistics,
            "detailed_results": [
                {
                    "mol_id": r.mol_id,
                    "smiles_original": r.smiles_original,
                    "smiles_sanitized": r.smiles_sanitized,
                    "kekulize_success": r.kekulize_success,
                    "sanitize_success": r.sanitize_success,
                    "error_type": r.error_type,
                    "error_message": r.error_message,
                    "num_atoms": r.num_atoms,
                    "num_heavy_atoms": r.num_heavy_atoms,
                    "num_aromatic_atoms": r.num_aromatic_atoms,
                    "num_aromatic_rings": r.num_aromatic_rings,
                    "molecular_formula": r.molecular_formula,
                    "molecular_weight": r.molecular_weight,
                    "logp": r.logp
                } for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Results saved to {output_path}")
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        if not self.statistics:
            return "No analysis results available."
        
        stats = self.statistics
        
        report = f"""
🧪 COCONUT Kekulize Analysis Summary Report
{'='*50}

📊 OVERVIEW
Total molecules analyzed: {stats['total_molecules']:,}
Overall success rate: {stats['success_rates']['both_success']:.1%}
Problematic molecules: {stats['success_rates']['total_problematic']:.1%}

🔍 SUCCESS RATES  
✅ Kekulize success: {stats['success_rates']['kekulize_success']:.1%}
✅ Sanitize success: {stats['success_rates']['sanitize_success']:.1%}
✅ Both successful: {stats['success_rates']['both_success']:.1%}

❌ ERROR DISTRIBUTION
"""
        
        for error_type, count in stats['error_distribution'].items():
            percentage = (count / stats['total_molecules']) * 100
            report += f"  {error_type}: {count:,} ({percentage:.1f}%)\n"
        
        report += f"""
🧬 MOLECULAR PROPERTIES
Average atoms: {stats['molecular_properties']['avg_atoms']:.1f}
Average heavy atoms: {stats['molecular_properties']['avg_heavy_atoms']:.1f}  
Average aromatic atoms: {stats['molecular_properties']['avg_aromatic_atoms']:.1f}
Average aromatic rings: {stats['molecular_properties']['avg_aromatic_rings']:.1f}
Average molecular weight: {stats['molecular_properties']['avg_molecular_weight']:.1f}
Size range: {stats['molecular_properties']['min_atoms']} - {stats['molecular_properties']['max_atoms']} atoms

🔍 ERROR PATTERNS
"""
        
        if 'error_patterns' in stats:
            for pattern, count in stats['error_patterns'].items():
                percentage = (count / stats['total_molecules']) * 100
                report += f"  {pattern.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)\n"
        
        return report


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze kekulize issues in COCONUT dataset"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to COCONUT SDF file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True, 
        help="Path to output JSON analysis file"
    )
    parser.add_argument(
        "--max-molecules", "-n",
        type=int,
        help="Maximum number of molecules to analyze (for testing)"
    )
    parser.add_argument(
        "--report", "-r",
        help="Path to save human-readable summary report"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = COCONUTKekulizeAnalyzer()
        
        # Run analysis
        analyzer.analyze_sdf_file(args.input, args.max_molecules)
        
        # Save results
        analyzer.save_results(args.output)
        
        # Generate and save summary report
        summary = analyzer.generate_summary_report()
        print(summary)
        
        if args.report:
            with open(args.report, 'w') as f:
                f.write(summary)
            print(f"📄 Summary report saved to {args.report}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()