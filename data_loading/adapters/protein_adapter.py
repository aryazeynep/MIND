# data_loading/adapters/protein_adapter.py
# source /opt/anaconda3/bin/activate

import sys
import os
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

# Add project root to sys.path to allow imports like 'data_loading.adapters'
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_loading.adapters.base_adapter import BaseAdapter
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule

# BioPython for reading PDB and CIF files
from Bio.PDB import PDBParser, MMCIFParser, is_aa

class ProteinAdapter(BaseAdapter):
    """
    Adapter to parse raw PDB/CIF files into the UniversalMolecule format.
    Can be configured to handle proteins only or protein-heteroatom complexes.
    """

    def __init__(self, include_hetatms: bool = False): # include_hetatms=False -> only protein, include_hetatms=True -> protein + heteroatoms
        super().__init__("protein")
        self.include_hetatms = include_hetatms
        # Initialize parsers with QUIET=True to suppress standard warnings
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)

    def load_raw_data(self, data_path: str, max_samples: int = None, manifest_file: str = None) -> List[Path]:
        """
        Load protein structure files, optionally filtering by manifest file.
        
        Args:
            data_path: Directory containing .pdb/.cif files
            max_samples: Limit number of files (applied after manifest filtering)
            manifest_file: Optional CSV file with 'repId' column for filtering
            
        Returns:
            List of Path objects to protein structure files
        """
        print(f"Scanning protein structures: {data_path}")
        if not Path(data_path).is_dir():
            raise FileNotFoundError(f"Specified data_path is not a directory: {data_path}")

        # If manifest provided, use it to filter specific files
        if manifest_file:
            import pandas as pd
            print(f"📋 Loading manifest: {manifest_file}")
            
            try:
                manifest_df = pd.read_csv(manifest_file)
            except Exception as e:
                raise ValueError(f"Failed to load manifest file: {e}")
            
            if 'repId' not in manifest_df.columns:
                raise ValueError(f"Manifest file must have 'repId' column, found: {manifest_df.columns.tolist()}")
            
            # Extract protein IDs
            protein_ids = manifest_df['repId'].tolist()
            print(f"📋 Manifest contains {len(protein_ids):,} protein IDs")
            
            # Build file paths based on manifest (AlphaFold naming convention)
            files = []
            missing_files = []
            
            for protein_id in protein_ids:
                # Try both .pdb and .cif extensions with AlphaFold naming
                pdb_path = Path(data_path) / f"AF-{protein_id}-F1-model_v4.pdb"
                cif_path = Path(data_path) / f"AF-{protein_id}-F1-model_v4.cif"
                
                # Also try direct filename match
                direct_pdb = Path(data_path) / f"{protein_id}.pdb"
                direct_cif = Path(data_path) / f"{protein_id}.cif"
                
                if pdb_path.exists():
                    files.append(pdb_path)
                elif cif_path.exists():
                    files.append(cif_path)
                elif direct_pdb.exists():
                    files.append(direct_pdb)
                elif direct_cif.exists():
                    files.append(direct_cif)
                else:
                    missing_files.append(protein_id)
            
            if missing_files:
                print(f"⚠️  {len(missing_files):,} files not found (will be skipped)")
                if len(missing_files) <= 5:
                    print(f"   Missing: {missing_files}")
                else:
                    print(f"   First 5 missing: {missing_files[:5]}")
            
            print(f"✅ Found {len(files):,}/{len(protein_ids):,} files from manifest")
            
        else:
            # Original behavior: scan directory
            files = sorted(list(Path(data_path).glob("*.pdb")))
            files.extend(sorted(list(Path(data_path).glob("*.cif"))))
            
            if not files:
                raise FileNotFoundError(f"No .pdb or .cif files found in directory: {data_path}")
            
            print(f"Found {len(files):,} structure files in total.")
        
        # Apply max_samples limit
        if max_samples:
            print(f"Processing limited to {max_samples:,} samples.")
            return files[:max_samples]
        
        return files

    def create_blocks(self, raw_item: Path) -> List[UniversalBlock]:
        """
        Parses a single PDB/CIF file into a list of UniversalBlock objects.
        By default, each amino acid residue is treated as one block. If `include_hetatms`
        is True, non-standard residues are also treated as blocks.
        """
        try:
            suffix = raw_item.suffix.lower()
            if suffix == ".pdb":
                structure = self.pdb_parser.get_structure(id=raw_item.stem, file=str(raw_item))
            elif suffix == ".cif":
                structure = self.cif_parser.get_structure(structure_id=raw_item.stem, file=str(raw_item))
            else:
                return [] # Unsupported format
        # More informative error logging.
        except Exception as e:
            print(f"ERROR: File cannot be read {raw_item.name}: {type(e).__name__} - {e}")
            return []

        blocks = []
        
        # Iterate through the hierarchy: model -> chain -> residue
        for model in structure:
            for chain in model:
                for residue in chain:
                    
                    # --- Filtering Section ---
                    # Check if the residue is a standard amino acid
                    is_standard_aa = residue.get_id()[0] == ' ' or is_aa(residue, standard=True) # standard=True -> only standard amino acids
                    
                    # Conditionally skip heteroatoms based on the `include_hetatms` flag.
                    if not self.include_hetatms and not is_standard_aa:
                        continue
                    
                    # --- Block Creation Section ---
                    block_atoms = []
                    res_name = residue.get_resname()

                    for atom in residue:
                        # Skip hydrogen atoms to reduce complexity
                        element = (atom.element or "C").strip().upper()
                        if element == 'H':
                            continue
                        
                        # Handle alternate locations, keeping only the primary or 'A' location
                        altloc = atom.get_altloc()
                        if altloc not in (" ", "A"):
                            continue
                        
                        # Determine entity_idx based on residue type.
                        # 0 for protein, 1 for heteroatoms (ligands, ions, etc.).
                        current_entity_idx = 0 if is_standard_aa else 1

                        uni_atom = UniversalAtom(
                            element=element,
                            position=tuple(atom.get_coord().tolist()),
                            pos_code=atom.get_name(), # e.g., 'CA', 'CB', 'N'
                            block_idx=len(blocks),
                            atom_idx_in_block=len(block_atoms),
                            entity_idx=current_entity_idx # Dynamically set entity_idx
                        )
                        block_atoms.append(uni_atom)
                    
                    if block_atoms:
                        # Create a UniversalBlock for the residue if it contains any atoms after filtering.
                        block = UniversalBlock(
                            symbol=res_name, # e.g., 'ALA', 'LYS', or 'ZN', 'HEM' for heteroatoms
                            atoms=block_atoms
                        )
                        blocks.append(block)
        
        return blocks

    def convert_to_universal(self, raw_item: Path) -> UniversalMolecule:
        """
        Wraps the created blocks into a UniversalMolecule object with metadata.
        """
        blocks = self.create_blocks(raw_item)
        return UniversalMolecule(
            id=raw_item.stem, # Use the file stem as the ID (e.g., '1a2b')
            dataset_type=self.dataset_type,
            blocks=blocks,
            properties={}
        )