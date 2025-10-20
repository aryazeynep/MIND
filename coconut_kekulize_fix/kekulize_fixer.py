#!/usr/bin/env python3
"""
COCONUT Kekulize Problem Fixer

This script attempts to fix kekulize and sanitization issues in COCONUT molecules
using various chemical standardization and correction strategies.

Author: AI Assistant  
Date: September 2024
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
import traceback
import copy

# Add RDKit and other dependencies
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
    from rdkit.Chem import rdmolops
    
    # Try different import patterns for MolStandardize based on RDKit version
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.MolStandardize.rdMolStandardize import Standardizer, Uncharger, TautomerEnumerator
    except (ImportError, AttributeError):
        try:
            from rdkit.Chem import rdMolStandardize
            from rdkit.Chem.rdMolStandardize import Standardizer, Uncharger, TautomerEnumerator
        except (ImportError, AttributeError):
            # Fallback for older RDKit versions
            rdMolStandardize = None
            Standardizer = None
            Uncharger = None
            TautomerEnumerator = None
            
except ImportError as e:
    print(f"Error: RDKit not found. Please install RDKit: {e}")
    sys.exit(1)

# Suppress RDKit warnings for cleaner output  
RDLogger.DisableLog('rdApp.*')


@dataclass 
class FixResult:
    """Data class to store fix attempt results"""
    mol_id: str
    original_smiles: Optional[str]
    fixed_smiles: Optional[str] 
    fix_successful: bool
    fix_strategy: str
    error_message: str
    kekulize_success: bool
    sanitize_success: bool


class COCONUTKekulizeFixer:
    """Comprehensive fixer for kekulize issues in COCONUT molecules"""
    
    def __init__(self):
        self.fix_results: List[FixResult] = []
        self.statistics: Dict[str, Any] = {}
        
        # Initialize RDKit standardization tools (with fallback for different versions)
        try:
            if Standardizer is not None:
                self.standardizer = Standardizer()
            else:
                self.standardizer = None
                
            if Uncharger is not None:
                self.uncharger = Uncharger()
            else:
                self.uncharger = None
                
            if TautomerEnumerator is not None:
                self.tautomer_enumerator = TautomerEnumerator()
            else:
                self.tautomer_enumerator = None
                
            print("✅ Initialized standardization tools")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize some standardization tools: {e}")
            self.standardizer = None
            self.uncharger = None
            self.tautomer_enumerator = None
        
    def fix_sdf_file(self, input_path: str, output_path: str, max_molecules: Optional[int] = None) -> None:
        """Fix kekulize issues in an entire SDF file"""
        print(f"🔧 Starting COCONUT kekulize fixing...")
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"SDF file not found: {input_path}")
        
        supplier = Chem.SDMolSupplier(input_path, removeHs=False, sanitize=False)
        writer = Chem.SDWriter(output_path)
        
        total_molecules = 0
        processed_molecules = 0
        fixed_molecules = 0
        
        try:
            for i, mol in enumerate(supplier):
                if max_molecules and processed_molecules >= max_molecules:
                    break
                    
                total_molecules += 1
                
                if mol is None:
                    print(f"Warning: Molecule {i} could not be loaded from SDF")
                    continue
                
                # Progress update
                if processed_molecules % 1000 == 0:
                    print(f"Processing molecule {processed_molecules}...")
                
                fixed_mol, fix_result = self.fix_single_molecule(mol, f"coconut_{i}")
                self.fix_results.append(fix_result)
                
                # Write to output SDF if fix was successful
                if fixed_mol is not None:
                    writer.write(fixed_mol)
                    fixed_molecules += 1
                
                processed_molecules += 1
        
        finally:
            writer.close()
        
        print(f"✅ Fixing complete!")
        print(f"   Processed: {processed_molecules} molecules")
        print(f"   Fixed: {fixed_molecules} molecules")
        print(f"   Success rate: {fixed_molecules/processed_molecules:.1%}")
        
        self._compute_statistics()
    
    def fix_single_molecule(self, mol: Chem.Mol, mol_id: str) -> Tuple[Optional[Chem.Mol], FixResult]:
        """Attempt to fix a single molecule using various strategies"""
        
        # Initialize fix result
        fix_result = FixResult(
            mol_id=mol_id,
            original_smiles=None,
            fixed_smiles=None,
            fix_successful=False,
            fix_strategy="none",
            error_message="",
            kekulize_success=False,
            sanitize_success=False
        )
        
        try:
            # Get original SMILES
            try:
                fix_result.original_smiles = Chem.MolToSmiles(mol, canonical=True)
            except:
                fix_result.original_smiles = "Could not generate"
            
            # Try different fixing strategies in order of preference
            fixing_strategies = [
                self._strategy_direct,
                self._strategy_add_hydrogens,
                self._strategy_standardize,
                self._strategy_uncharge,
                self._strategy_fix_valences,
                self._strategy_tautomer_canonical,
                self._strategy_remove_salts,
                self._strategy_neutralize_charges,
                self._strategy_kekulize_alternative,
                self._strategy_sanitize_alternative
            ]
            
            for strategy in fixing_strategies:
                try:
                    fixed_mol = strategy(mol)
                    if self._validate_fix(fixed_mol):
                        fix_result.fix_successful = True
                        fix_result.fix_strategy = strategy.__name__.replace('_strategy_', '')
                        fix_result.fixed_smiles = Chem.MolToSmiles(fixed_mol, canonical=True)
                        fix_result.kekulize_success = True
                        fix_result.sanitize_success = True
                        return fixed_mol, fix_result
                except Exception as e:
                    # Continue to next strategy if this one fails
                    fix_result.error_message += f"{strategy.__name__}: {str(e)}; "
                    continue
            
            # If all strategies failed
            fix_result.fix_successful = False
            fix_result.fix_strategy = "all_failed"
            
        except Exception as e:
            fix_result.error_message = f"Fix attempt failed: {str(e)}"
        
        return None, fix_result
    
    def _strategy_direct(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 1: Try direct sanitization and kekulization"""
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        return mol_copy
    
    def _strategy_add_hydrogens(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 2: Add explicit hydrogens then sanitize"""
        mol_copy = Chem.Mol(mol)
        mol_copy = Chem.AddHs(mol_copy, explicitOnly=False)
        Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        return mol_copy
    
    def _strategy_standardize(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 3: Use RDKit standardizer"""
        mol_copy = Chem.Mol(mol)
        if self.standardizer is not None:
            mol_copy = self.standardizer.standardize(mol_copy)
        else:
            # Fallback: basic sanitization
            Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        return mol_copy
    
    def _strategy_uncharge(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 4: Remove formal charges"""
        mol_copy = Chem.Mol(mol)
        if self.uncharger is not None:
            mol_copy = self.uncharger.uncharge(mol_copy)
        else:
            # Fallback: manually set all formal charges to 0
            for atom in mol_copy.GetAtoms():
                atom.SetFormalCharge(0)
        Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        return mol_copy
    
    def _strategy_fix_valences(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 5: Fix obvious valence issues"""
        mol_copy = Chem.Mol(mol)
        
        # Fix common valence issues
        for atom in mol_copy.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            
            # Fix nitrogen valence issues (common in natural products)
            if atomic_num == 7:  # Nitrogen
                if atom.GetFormalCharge() == 0 and atom.GetExplicitValence() == 4:
                    atom.SetFormalCharge(1)
                elif atom.GetFormalCharge() == 0 and atom.GetExplicitValence() > 3:
                    atom.SetFormalCharge(atom.GetExplicitValence() - 3)
            
            # Fix oxygen valence issues
            elif atomic_num == 8:  # Oxygen
                if atom.GetFormalCharge() == 0 and atom.GetExplicitValence() > 2:
                    atom.SetFormalCharge(atom.GetExplicitValence() - 2)
            
            # Fix carbon valence issues
            elif atomic_num == 6:  # Carbon
                if atom.GetFormalCharge() == 0 and atom.GetExplicitValence() > 4:
                    # This shouldn't happen, but handle it
                    atom.SetFormalCharge(atom.GetExplicitValence() - 4)
        
        Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        return mol_copy
    
    def _strategy_tautomer_canonical(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 6: Use canonical tautomer"""
        mol_copy = Chem.Mol(mol)
        if self.tautomer_enumerator is not None:
            mol_copy = self.tautomer_enumerator.Canonicalize(mol_copy)
        else:
            # Fallback: just sanitize
            Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        return mol_copy
    
    def _strategy_remove_salts(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 7: Remove salt/solvent components"""
        mol_copy = Chem.Mol(mol)
        
        # Use RDKit's salt remover if available
        try:
            if rdMolStandardize is not None:
                # Try different ways to access SaltRemover
                try:
                    remover = rdMolStandardize.SaltRemover()
                except AttributeError:
                    try:
                        from rdkit.Chem.MolStandardize.rdMolStandardize import SaltRemover
                        remover = SaltRemover()
                    except:
                        remover = None
                
                if remover is not None:
                    mol_copy = remover.StripMol(mol_copy)
        except:
            pass  # Continue without salt removal
        
        Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        return mol_copy
    
    def _strategy_neutralize_charges(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 8: Neutralize all formal charges"""
        mol_copy = Chem.Mol(mol)
        
        # Set all formal charges to 0
        for atom in mol_copy.GetAtoms():
            atom.SetFormalCharge(0)
        
        Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        return mol_copy
    
    def _strategy_kekulize_alternative(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 9: Alternative kekulization approach"""
        mol_copy = Chem.Mol(mol)
        
        # Try without clearing aromatic flags first
        try:
            Chem.Kekulize(mol_copy, clearAromaticFlags=False)
        except:
            # If that fails, try with clearing flags
            Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        
        return mol_copy
    
    def _strategy_sanitize_alternative(self, mol: Chem.Mol) -> Chem.Mol:
        """Strategy 10: Alternative sanitization with specific operations"""
        mol_copy = Chem.Mol(mol)
        
        # Try partial sanitization
        sanitize_ops = (
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
            Chem.SanitizeFlags.SANITIZE_KEKULIZE |
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        )
        
        Chem.SanitizeMol(mol_copy, sanitizeOps=sanitize_ops)
        return mol_copy
    
    def _validate_fix(self, mol: Chem.Mol) -> bool:
        """Validate that a fix was successful"""
        if mol is None:
            return False
        
        try:
            # Test kekulization
            mol_test = Chem.Mol(mol)
            Chem.Kekulize(mol_test, clearAromaticFlags=True)
            
            # Test sanitization
            mol_test2 = Chem.Mol(mol)
            Chem.SanitizeMol(mol_test2)
            
            # Test SMILES generation
            smiles = Chem.MolToSmiles(mol)
            
            # Additional validation: make sure molecule makes chemical sense
            if mol.GetNumAtoms() == 0:
                return False
            
            # Check for reasonable valences
            for atom in mol.GetAtoms():
                if atom.GetTotalValence() > 8:  # Unreasonably high valence
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _compute_statistics(self) -> None:
        """Compute statistics from fix results"""
        total_molecules = len(self.fix_results)
        
        if total_molecules == 0:
            self.statistics = {"error": "No molecules processed"}
            return
        
        # Success rates
        successful_fixes = sum(1 for r in self.fix_results if r.fix_successful)
        
        # Strategy effectiveness
        strategy_counts = Counter([r.fix_strategy for r in self.fix_results if r.fix_successful])
        
        self.statistics = {
            "total_molecules": total_molecules,
            "successful_fixes": successful_fixes,
            "success_rate": successful_fixes / total_molecules,
            "strategy_effectiveness": dict(strategy_counts),
            "failed_molecules": total_molecules - successful_fixes,
            "failure_rate": (total_molecules - successful_fixes) / total_molecules
        }
    
    def save_results(self, output_path: str) -> None:
        """Save fix results to JSON file"""
        output_data = {
            "statistics": self.statistics,
            "fix_results": [
                {
                    "mol_id": r.mol_id,
                    "original_smiles": r.original_smiles,
                    "fixed_smiles": r.fixed_smiles,
                    "fix_successful": r.fix_successful,
                    "fix_strategy": r.fix_strategy,
                    "error_message": r.error_message,
                    "kekulize_success": r.kekulize_success,
                    "sanitize_success": r.sanitize_success
                } for r in self.fix_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Fix results saved to {output_path}")
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        if not self.statistics:
            return "No fix results available."
        
        stats = self.statistics
        
        report = f"""
🔧 COCONUT Kekulize Fixer Summary Report
{'='*50}

📊 OVERVIEW
Total molecules processed: {stats['total_molecules']:,}
Successful fixes: {stats['successful_fixes']:,} ({stats['success_rate']:.1%})
Failed fixes: {stats['failed_molecules']:,} ({stats['failure_rate']:.1%})

🎯 STRATEGY EFFECTIVENESS
"""
        
        for strategy, count in stats['strategy_effectiveness'].items():
            percentage = (count / stats['successful_fixes']) * 100 if stats['successful_fixes'] > 0 else 0
            report += f"  {strategy.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)\n"
        
        return report


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fix kekulize issues in COCONUT dataset"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input COCONUT SDF file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True, 
        help="Path to output fixed SDF file"
    )
    parser.add_argument(
        "--max-molecules", "-n",
        type=int,
        help="Maximum number of molecules to process (for testing)"
    )
    parser.add_argument(
        "--results", "-r",
        help="Path to save detailed fix results JSON"
    )
    parser.add_argument(
        "--report", 
        help="Path to save human-readable summary report"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize fixer
        fixer = COCONUTKekulizeFixer()
        
        # Run fixing process
        fixer.fix_sdf_file(args.input, args.output, args.max_molecules)
        
        # Save detailed results if requested
        if args.results:
            fixer.save_results(args.results)
        
        # Generate and save summary report
        summary = fixer.generate_summary_report()
        print(summary)
        
        if args.report:
            with open(args.report, 'w') as f:
                f.write(summary)
            print(f"📄 Summary report saved to {args.report}")
        
    except Exception as e:
        print(f"❌ Error during fixing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()