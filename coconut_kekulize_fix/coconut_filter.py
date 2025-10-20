#!/usr/bin/env python3
"""
COCONUT Dataset Filter

This script filters the COCONUT dataset to remove molecules that cannot be 
kekulized or sanitized, producing a clean dataset for training.

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
import traceback

# Add RDKit and other dependencies
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem
    
    # Try different import patterns for MolStandardize based on RDKit version
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.MolStandardize.rdMolStandardize import Standardizer
    except (ImportError, AttributeError):
        try:
            from rdkit.Chem import rdMolStandardize
            from rdkit.Chem.rdMolStandardize import Standardizer
        except (ImportError, AttributeError):
            # Fallback for older RDKit versions
            rdMolStandardize = None
            Standardizer = None
            
except ImportError as e:
    print(f"Error: RDKit not found. Please install RDKit: {e}")
    sys.exit(1)

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')


class COCONUTDatasetFilter:
    """Filter COCONUT dataset to remove problematic molecules"""
    
    def __init__(self, 
                 max_atoms: int = 150, 
                 min_atoms: int = 5,
                 max_molecular_weight: float = 1000.0,
                 min_molecular_weight: float = 50.0):
        """
        Initialize filter with quality criteria
        
        Args:
            max_atoms: Maximum number of atoms allowed
            min_atoms: Minimum number of atoms required  
            max_molecular_weight: Maximum molecular weight allowed
            min_molecular_weight: Minimum molecular weight required
        """
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.max_molecular_weight = max_molecular_weight
        self.min_molecular_weight = min_molecular_weight
        
        self.filter_statistics = {
            "total_molecules": 0,
            "passed_molecules": 0,
            "failed_molecules": 0,
            "filter_reasons": defaultdict(int)
        }
        
        # Initialize RDKit standardization tools (with fallback for different versions)
        try:
            if Standardizer is not None:
                self.standardizer = Standardizer()
            else:
                self.standardizer = None
            print("✅ Initialized standardization tools")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize standardization tools: {e}")
            self.standardizer = None
        
    def filter_sdf_file(self, input_path: str, output_path: str, max_molecules: Optional[int] = None) -> None:
        """Filter an entire SDF file"""
        print(f"🧹 Starting COCONUT dataset filtering...")
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print(f"Quality criteria:")
        print(f"  - Atom count: {self.min_atoms} - {self.max_atoms}")
        print(f"  - Molecular weight: {self.min_molecular_weight} - {self.max_molecular_weight}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"SDF file not found: {input_path}")
        
        supplier = Chem.SDMolSupplier(input_path, removeHs=False, sanitize=False)
        writer = Chem.SDWriter(output_path)
        
        try:
            for i, mol in enumerate(supplier):
                if max_molecules and i >= max_molecules:
                    break
                
                self.filter_statistics["total_molecules"] += 1
                
                if mol is None:
                    self.filter_statistics["filter_reasons"]["failed_to_load"] += 1
                    continue
                
                # Progress update
                if i % 1000 == 0:
                    print(f"Processing molecule {i}...")
                
                passed, reason = self._evaluate_molecule(mol)
                
                if passed:
                    # Standardize and write the molecule
                    try:
                        standardized_mol = self._standardize_molecule(mol)
                        if standardized_mol is not None:
                            writer.write(standardized_mol)
                            self.filter_statistics["passed_molecules"] += 1
                        else:
                            self.filter_statistics["filter_reasons"]["standardization_failed"] += 1
                    except Exception as e:
                        self.filter_statistics["filter_reasons"]["standardization_error"] += 1
                else:
                    self.filter_statistics["filter_reasons"][reason] += 1
        
        finally:
            writer.close()
        
        self.filter_statistics["failed_molecules"] = (
            self.filter_statistics["total_molecules"] - 
            self.filter_statistics["passed_molecules"]
        )
        
        self._print_summary()
    
    def _evaluate_molecule(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Evaluate if a molecule passes quality filters"""
        
        # Check basic molecular properties first (fast checks)
        num_atoms = mol.GetNumAtoms()
        if num_atoms < self.min_atoms:
            return False, "too_few_atoms"
        if num_atoms > self.max_atoms:
            return False, "too_many_atoms"
        
        # Check molecular weight
        try:
            mol_weight = Descriptors.MolWt(mol)
            if mol_weight < self.min_molecular_weight:
                return False, "molecular_weight_too_low"
            if mol_weight > self.max_molecular_weight:
                return False, "molecular_weight_too_high"
        except:
            return False, "molecular_weight_calculation_failed"
        
        # Check for obvious problematic structures
        if self._has_problematic_atoms(mol):
            return False, "problematic_atoms"
        
        if self._has_unusual_bonds(mol):
            return False, "unusual_bonds"
        
        # Test kekulization (most important chemical validity test)
        if not self._test_kekulization(mol):
            return False, "kekulization_failed"
        
        # Test sanitization  
        if not self._test_sanitization(mol):
            return False, "sanitization_failed"
        
        # Additional chemical validity checks
        if not self._test_chemical_validity(mol):
            return False, "chemical_invalidity"
        
        return True, "passed"
    
    def _has_problematic_atoms(self, mol: Chem.Mol) -> bool:
        """Check for atoms that commonly cause issues"""
        problematic_elements = {'*', 'R', 'Q', 'X', 'L', 'A', 'G', 'M'}
        
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            
            # Check for wildcard or dummy atoms
            if symbol in problematic_elements:
                return True
            
            # Check for unreasonable formal charges
            formal_charge = atom.GetFormalCharge()
            if abs(formal_charge) > 3:  # Very high formal charges are suspicious
                return True
            
            # Check for unreasonable valence
            try:
                total_valence = atom.GetTotalValence()
                if total_valence > 8:  # Beyond reasonable valence
                    return True
            except:
                return True  # If we can't calculate valence, it's problematic
        
        return False
    
    def _has_unusual_bonds(self, mol: Chem.Mol) -> bool:
        """Check for unusual bond types that might cause issues"""
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            
            # Check for very high bond orders (quadruple bonds are rare)
            if bond_type == Chem.BondType.OTHER or str(bond_type) not in [
                'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'
            ]:
                return True
        
        return False
    
    def _test_kekulization(self, mol: Chem.Mol) -> bool:
        """Test if molecule can be kekulized"""
        try:
            mol_copy = Chem.Mol(mol)
            Chem.Kekulize(mol_copy, clearAromaticFlags=True)
            return True
        except:
            return False
    
    def _test_sanitization(self, mol: Chem.Mol) -> bool:
        """Test if molecule can be sanitized"""
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            return True
        except:
            return False
    
    def _test_chemical_validity(self, mol: Chem.Mol) -> bool:
        """Additional chemical validity tests"""
        try:
            # Test SMILES generation
            smiles = Chem.MolToSmiles(mol)
            if not smiles or len(smiles) < 3:
                return False
            
            # Test that we can recreate molecule from SMILES
            mol_from_smiles = Chem.MolFromSmiles(smiles)
            if mol_from_smiles is None:
                return False
            
            # Check for disconnected fragments (salts, solvents)
            fragments = Chem.GetMolFrags(mol, asMols=True)
            if len(fragments) > 1:
                # If multiple fragments, keep only if largest is significant
                fragment_sizes = [f.GetNumAtoms() for f in fragments]
                largest_size = max(fragment_sizes)
                if largest_size < self.min_atoms:  # All fragments too small
                    return False
            
            return True
            
        except:
            return False
    
    def _standardize_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Standardize molecule before writing to output"""
        try:
            mol_copy = Chem.Mol(mol)
            
            # Basic standardization (with fallback if standardizer not available)
            if self.standardizer is not None:
                mol_copy = self.standardizer.standardize(mol_copy)
            else:
                # Fallback: basic sanitization
                Chem.SanitizeMol(mol_copy)
            
            # Handle multiple fragments - keep largest
            fragments = Chem.GetMolFrags(mol_copy, asMols=True)
            if len(fragments) > 1:
                # Keep the largest fragment
                largest_fragment = max(fragments, key=lambda x: x.GetNumAtoms())
                mol_copy = largest_fragment
            
            # Final sanitization check
            Chem.SanitizeMol(mol_copy)
            
            # Add some basic properties as SDF data
            mol_copy.SetProp("NumAtoms", str(mol_copy.GetNumAtoms()))
            mol_copy.SetProp("NumHeavyAtoms", str(mol_copy.GetNumHeavyAtoms()))
            mol_copy.SetProp("MolecularWeight", str(Descriptors.MolWt(mol_copy)))
            mol_copy.SetProp("SMILES", Chem.MolToSmiles(mol_copy))
            
            return mol_copy
            
        except Exception as e:
            print(f"Warning: Standardization failed: {e}")
            return None
    
    def _print_summary(self) -> None:
        """Print filtering summary"""
        stats = self.filter_statistics
        
        print(f"\n✅ COCONUT Filtering Complete!")
        print(f"{'='*50}")
        print(f"Total molecules processed: {stats['total_molecules']:,}")
        print(f"Passed molecules: {stats['passed_molecules']:,} ({stats['passed_molecules']/stats['total_molecules']:.1%})")
        print(f"Failed molecules: {stats['failed_molecules']:,} ({stats['failed_molecules']/stats['total_molecules']:.1%})")
        
        print(f"\n📊 Filter Breakdown:")
        for reason, count in stats['filter_reasons'].items():
            percentage = (count / stats['total_molecules']) * 100
            print(f"  {reason.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
    
    def save_statistics(self, output_path: str) -> None:
        """Save filtering statistics to JSON"""
        with open(output_path, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            stats_copy = dict(self.filter_statistics)
            stats_copy['filter_reasons'] = dict(stats_copy['filter_reasons'])
            json.dump(stats_copy, f, indent=2)
        
        print(f"📊 Statistics saved to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Filter COCONUT dataset to remove problematic molecules"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input COCONUT SDF file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True, 
        help="Path to output filtered SDF file"
    )
    parser.add_argument(
        "--max-atoms", 
        type=int,
        default=150,
        help="Maximum number of atoms allowed (default: 150)"
    )
    parser.add_argument(
        "--min-atoms", 
        type=int,
        default=5,
        help="Minimum number of atoms required (default: 5)"
    )
    parser.add_argument(
        "--max-weight", 
        type=float,
        default=1000.0,
        help="Maximum molecular weight allowed (default: 1000.0)"
    )
    parser.add_argument(
        "--min-weight", 
        type=float,
        default=50.0,
        help="Minimum molecular weight required (default: 50.0)"
    )
    parser.add_argument(
        "--max-molecules", "-n",
        type=int,
        help="Maximum number of molecules to process (for testing)"
    )
    parser.add_argument(
        "--statistics", "-s",
        help="Path to save filtering statistics JSON"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize filter
        filter_tool = COCONUTDatasetFilter(
            max_atoms=args.max_atoms,
            min_atoms=args.min_atoms,
            max_molecular_weight=args.max_weight,
            min_molecular_weight=args.min_weight
        )
        
        # Run filtering
        filter_tool.filter_sdf_file(args.input, args.output, args.max_molecules)
        
        # Save statistics if requested
        if args.statistics:
            filter_tool.save_statistics(args.statistics)
        
    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()