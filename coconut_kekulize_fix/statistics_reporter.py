#!/usr/bin/env python3
"""
COCONUT Dataset Statistics Reporter

This script generates comprehensive statistics and reports about the COCONUT dataset
quality, including molecular properties, kekulize success rates, and filtering outcomes.

Author: AI Assistant
Date: September 2024
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import traceback

# Add RDKit
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcNumHBD, CalcNumHBA
except ImportError as e:
    print(f"Error: RDKit not found. Please install RDKit: {e}")
    sys.exit(1)

# Add matplotlib and seaborn with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")
    HAS_PLOTTING = False

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


class COCONUTStatisticsReporter:
    """Comprehensive statistics reporter for COCONUT dataset"""
    
    def __init__(self):
        self.molecular_properties = []
        self.statistics = {}
        
    def analyze_sdf_file(self, sdf_path: str, max_molecules: Optional[int] = None) -> None:
        """Analyze molecular properties in SDF file"""
        print(f"📊 Analyzing COCONUT dataset statistics...")
        print(f"Input file: {sdf_path}")
        
        if not os.path.exists(sdf_path):
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")
        
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        processed_count = 0
        
        for i, mol in enumerate(supplier):
            if max_molecules and processed_count >= max_molecules:
                break
            
            if mol is None:
                continue
            
            if processed_count % 1000 == 0:
                print(f"Processing molecule {processed_count}...")
            
            properties = self._calculate_molecular_properties(mol, f"coconut_{i}")
            if properties:
                self.molecular_properties.append(properties)
                processed_count += 1
        
        print(f"✅ Analyzed {processed_count} molecules")
        self._compute_statistics()
    
    def _calculate_molecular_properties(self, mol: Chem.Mol, mol_id: str) -> Optional[Dict[str, Any]]:
        """Calculate comprehensive molecular properties"""
        try:
            properties = {
                'mol_id': mol_id,
                'smiles': Chem.MolToSmiles(mol),
                'molecular_formula': rdMolDescriptors.CalcMolFormula(mol),
                
                # Basic properties
                'num_atoms': mol.GetNumAtoms(),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                'molecular_weight': Descriptors.MolWt(mol),
                
                # Chemical properties
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                'num_rotatable_bonds': CalcNumRotatableBonds(mol),
                'num_hbd': CalcNumHBD(mol),  # Hydrogen bond donors
                'num_hba': CalcNumHBA(mol),  # Hydrogen bond acceptors
                
                # Lipophilicity and drug-likeness
                'logp': Crippen.MolLogP(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),  # Topological polar surface area
                
                # Complexity measures
                'bertz_complexity': rdMolDescriptors.BertzCT(mol),
                'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                'num_stereocenters': rdMolDescriptors.CalcNumAtomStereoCenters(mol),
                
                # Aromatic properties
                'fraction_aromatic': self._calc_aromatic_fraction(mol),
                'num_aromatic_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
                
                # Element composition
                'element_counts': self._count_elements(mol),
                
                # Drug-likeness indicators
                'lipinski_violations': self._count_lipinski_violations(mol),
                'qed_score': None,  # Will calculate if rdkit version supports it
            }
            
            # Try to calculate QED score if available
            try:
                from rdkit.Chem import QED
                properties['qed_score'] = QED.qed(mol)
            except:
                pass
            
            return properties
            
        except Exception as e:
            print(f"Warning: Failed to calculate properties for {mol_id}: {e}")
            return None
    
    def _calc_aromatic_fraction(self, mol: Chem.Mol) -> float:
        """Calculate fraction of aromatic atoms"""
        total_atoms = mol.GetNumAtoms()
        if total_atoms == 0:
            return 0.0
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        return aromatic_atoms / total_atoms
    
    def _count_elements(self, mol: Chem.Mol) -> Dict[str, int]:
        """Count different elements in molecule"""
        element_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        return element_counts
    
    def _count_lipinski_violations(self, mol: Chem.Mol) -> int:
        """Count Lipinski rule violations"""
        violations = 0
        
        # Rule 1: Molecular weight <= 500 Da
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        
        # Rule 2: LogP <= 5
        if Crippen.MolLogP(mol) > 5:
            violations += 1
        
        # Rule 3: Hydrogen bond donors <= 5
        if CalcNumHBD(mol) > 5:
            violations += 1
        
        # Rule 4: Hydrogen bond acceptors <= 10
        if CalcNumHBA(mol) > 10:
            violations += 1
        
        return violations
    
    def _compute_statistics(self) -> None:
        """Compute comprehensive statistics"""
        if not self.molecular_properties:
            self.statistics = {"error": "No molecular properties calculated"}
            return
        
        df = pd.DataFrame(self.molecular_properties)
        
        # Basic statistics
        numeric_columns = ['num_atoms', 'num_heavy_atoms', 'molecular_weight', 'num_rings',
                          'num_aromatic_rings', 'num_rotatable_bonds', 'logp', 'tpsa',
                          'bertz_complexity', 'num_heteroatoms', 'fraction_aromatic']
        
        basic_stats = {}
        for col in numeric_columns:
            if col in df.columns:
                basic_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }
        
        # Element distribution
        all_elements = {}
        for props in self.molecular_properties:
            for element, count in props['element_counts'].items():
                all_elements[element] = all_elements.get(element, 0) + count
        
        # Drug-likeness statistics
        lipinski_stats = df['lipinski_violations'].value_counts().to_dict()
        
        # Size distributions
        size_distributions = {
            'small_molecules': len(df[df['num_heavy_atoms'] <= 20]),
            'medium_molecules': len(df[(df['num_heavy_atoms'] > 20) & (df['num_heavy_atoms'] <= 50)]),
            'large_molecules': len(df[df['num_heavy_atoms'] > 50])
        }
        
        # Complexity analysis
        complexity_stats = {
            'simple': len(df[df['bertz_complexity'] <= 300]),
            'moderate': len(df[(df['bertz_complexity'] > 300) & (df['bertz_complexity'] <= 600)]),
            'complex': len(df[df['bertz_complexity'] > 600])
        }
        
        self.statistics = {
            'total_molecules': len(self.molecular_properties),
            'basic_statistics': basic_stats,
            'element_distribution': all_elements,
            'lipinski_violations_distribution': lipinski_stats,
            'size_distribution': size_distributions,
            'complexity_distribution': complexity_stats,
            'aromatic_content': {
                'highly_aromatic': len(df[df['fraction_aromatic'] > 0.5]),
                'moderately_aromatic': len(df[(df['fraction_aromatic'] > 0.2) & (df['fraction_aromatic'] <= 0.5)]),
                'low_aromatic': len(df[df['fraction_aromatic'] <= 0.2])
            }
        }
    
    def generate_plots(self, output_dir: str) -> None:
        """Generate comprehensive plots"""
        if not HAS_PLOTTING:
            print("⚠️ Plotting libraries not available. Skipping plot generation.")
            return
            
        if not self.molecular_properties:
            print("No data available for plotting")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.molecular_properties)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style
        
        try:
            sns.set_palette("husl")
        except:
            pass  # Continue without seaborn styling
        
        # 1. Molecular weight distribution
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        plt.hist(df['molecular_weight'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Molecular Weight (Da)')
        plt.ylabel('Frequency')
        plt.title('Molecular Weight Distribution')
        plt.axvline(df['molecular_weight'].mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        
        # 2. Number of atoms distribution
        plt.subplot(2, 3, 2)
        plt.hist(df['num_heavy_atoms'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Heavy Atoms')
        plt.ylabel('Frequency')
        plt.title('Heavy Atom Count Distribution')
        
        # 3. LogP distribution
        plt.subplot(2, 3, 3)
        plt.hist(df['logp'], bins=40, alpha=0.7, edgecolor='black')
        plt.xlabel('LogP')
        plt.ylabel('Frequency')
        plt.title('Lipophilicity Distribution')
        
        # 4. Ring count distribution
        plt.subplot(2, 3, 4)
        plt.hist(df['num_rings'], bins=range(0, max(df['num_rings'])+2), alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Rings')
        plt.ylabel('Frequency')
        plt.title('Ring Count Distribution')
        
        # 5. Aromatic fraction
        plt.subplot(2, 3, 5)
        plt.hist(df['fraction_aromatic'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Fraction Aromatic')
        plt.ylabel('Frequency')
        plt.title('Aromatic Content Distribution')
        
        # 6. Complexity vs Size
        plt.subplot(2, 3, 6)
        plt.scatter(df['num_heavy_atoms'], df['bertz_complexity'], alpha=0.5, s=10)
        plt.xlabel('Number of Heavy Atoms')
        plt.ylabel('Bertz Complexity')
        plt.title('Complexity vs Molecular Size')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'molecular_properties_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Drug-likeness analysis
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        violation_counts = df['lipinski_violations'].value_counts().sort_index()
        plt.bar(violation_counts.index, violation_counts.values, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Lipinski Violations')
        plt.ylabel('Frequency')
        plt.title('Lipinski Rule Violations')
        
        # Element distribution (top 10)
        plt.subplot(1, 2, 2)
        all_elements = {}
        for props in self.molecular_properties:
            for element, count in props['element_counts'].items():
                all_elements[element] = all_elements.get(element, 0) + count
        
        top_elements = sorted(all_elements.items(), key=lambda x: x[1], reverse=True)[:10]
        elements, counts = zip(*top_elements)
        plt.bar(elements, counts, alpha=0.7, edgecolor='black')
        plt.xlabel('Element')
        plt.ylabel('Total Count')
        plt.title('Top 10 Elements in Dataset')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drug_likeness_and_elements.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to {output_dir}")
    
    def generate_report(self) -> str:
        """Generate comprehensive text report"""
        if not self.statistics:
            return "No statistics available"
        
        stats = self.statistics
        
        # Handle case where statistics computation failed
        if "error" in stats:
            return f"Statistics generation failed: {stats['error']}"
        
        # Get total molecules with fallback
        total_molecules = stats.get('total_molecules', len(self.molecular_properties))
        
        report = f"""
🧬 COCONUT Dataset Comprehensive Statistics Report
{'='*60}

📊 DATASET OVERVIEW
Total molecules analyzed: {total_molecules:,}
"""
        
        # Only add size distribution if it exists
        if 'size_distribution' in stats:
            size_dist = stats['size_distribution']
            report += f"""
🔬 MOLECULAR SIZE DISTRIBUTION
Small molecules (≤20 heavy atoms): {size_dist.get('small_molecules', 0):,} ({100*size_dist.get('small_molecules', 0)/total_molecules:.1f}%)
Medium molecules (21-50 heavy atoms): {size_dist.get('medium_molecules', 0):,} ({100*size_dist.get('medium_molecules', 0)/total_molecules:.1f}%)
Large molecules (>50 heavy atoms): {size_dist.get('large_molecules', 0):,} ({100*size_dist.get('large_molecules', 0)/total_molecules:.1f}%)
"""
        
        report += "\n🧮 KEY MOLECULAR PROPERTIES\n"
        
        # Add basic statistics if available
        if 'basic_statistics' in stats:
            for prop_name, prop_stats in stats['basic_statistics'].items():
                report += f"\n{prop_name.replace('_', ' ').title()}:\n"
                report += f"  Mean: {prop_stats.get('mean', 0):.2f} ± {prop_stats.get('std', 0):.2f}\n"
                report += f"  Range: {prop_stats.get('min', 0):.2f} - {prop_stats.get('max', 0):.2f}\n"
                report += f"  Median (Q1-Q3): {prop_stats.get('median', 0):.2f} ({prop_stats.get('q25', 0):.2f} - {prop_stats.get('q75', 0):.2f})\n"
        
        # Element distribution
        if 'element_distribution' in stats:
            report += f"\n🧪 ELEMENT COMPOSITION\n"
            element_dist = stats['element_distribution']
            top_elements = sorted(element_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            total_elements = sum(element_dist.values())
            for element, count in top_elements:
                percentage = 100 * count / total_elements if total_elements > 0 else 0
                report += f"  {element}: {count:,} ({percentage:.1f}%)\n"
        
        # Drug-likeness
        if 'lipinski_violations_distribution' in stats:
            report += f"\n💊 DRUG-LIKENESS (Lipinski Rules)\n"
            lipinski_dist = stats['lipinski_violations_distribution']
            for violations, count in sorted(lipinski_dist.items()):
                percentage = 100 * count / total_molecules if total_molecules > 0 else 0
                report += f"  {violations} violations: {count:,} ({percentage:.1f}%)\n"
        
        # Complexity
        if 'complexity_distribution' in stats:
            report += f"\n🧩 MOLECULAR COMPLEXITY\n"
            complex_stats = stats['complexity_distribution']
            report += f"  Simple (≤300): {complex_stats.get('simple', 0):,} ({100*complex_stats.get('simple', 0)/total_molecules:.1f}%)\n"
            report += f"  Moderate (301-600): {complex_stats.get('moderate', 0):,} ({100*complex_stats.get('moderate', 0)/total_molecules:.1f}%)\n"
            report += f"  Complex (>600): {complex_stats.get('complex', 0):,} ({100*complex_stats.get('complex', 0)/total_molecules:.1f}%)\n"
        
        # Aromatic content
        if 'aromatic_content' in stats:
            report += f"\n🌋 AROMATIC CONTENT\n"
            aromatic_stats = stats['aromatic_content']
            report += f"  Highly aromatic (>50%): {aromatic_stats.get('highly_aromatic', 0):,} ({100*aromatic_stats.get('highly_aromatic', 0)/total_molecules:.1f}%)\n"
            report += f"  Moderately aromatic (20-50%): {aromatic_stats.get('moderately_aromatic', 0):,} ({100*aromatic_stats.get('moderately_aromatic', 0)/total_molecules:.1f}%)\n"
            report += f"  Low aromatic (≤20%): {aromatic_stats.get('low_aromatic', 0):,} ({100*aromatic_stats.get('low_aromatic', 0)/total_molecules:.1f}%)\n"
        
        return report
    
    def save_statistics(self, output_path: str) -> None:
        """Save statistics to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        print(f"📊 Statistics saved to {output_path}")
    
    def export_csv(self, output_path: str) -> None:
        """Export molecular properties to CSV"""
        if not self.molecular_properties:
            print("No data to export")
            return
        
        # Flatten the data for CSV export
        flattened_data = []
        for props in self.molecular_properties:
            flat_props = props.copy()
            # Convert element_counts dict to individual columns
            element_counts = flat_props.pop('element_counts')
            for element, count in element_counts.items():
                flat_props[f'element_{element}'] = count
            flattened_data.append(flat_props)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(output_path, index=False)
        print(f"📄 Data exported to CSV: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive statistics for COCONUT dataset"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to COCONUT SDF file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True, 
        help="Directory to save all output files"
    )
    parser.add_argument(
        "--max-molecules", "-n",
        type=int,
        help="Maximum number of molecules to analyze (for testing)"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export molecular properties to CSV"
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize reporter
        reporter = COCONUTStatisticsReporter()
        
        # Analyze dataset
        reporter.analyze_sdf_file(args.input, args.max_molecules)
        
        # Generate and save report
        report = reporter.generate_report()
        print(report)
        
        report_path = os.path.join(args.output_dir, "coconut_statistics_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {report_path}")
        
        # Save statistics JSON
        stats_path = os.path.join(args.output_dir, "coconut_statistics.json")
        reporter.save_statistics(stats_path)
        
        # Generate plots if requested
        if args.generate_plots:
            plots_dir = os.path.join(args.output_dir, "plots")
            reporter.generate_plots(plots_dir)
        
        # Export CSV if requested
        if args.export_csv:
            csv_path = os.path.join(args.output_dir, "coconut_molecular_properties.csv")
            reporter.export_csv(csv_path)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()