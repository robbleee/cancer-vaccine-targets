#!/usr/bin/env python3
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

# Enhanced criteria for cancer vaccine targets
# 1. Cellular location - already implemented
# 2. Expression level and pattern in tumor samples
# 3. Cancer-specific antigens and mutational hotspots

def add_expression_data(dataset_dir):
    """Add tumor-specific expression data to immunogenicity scores"""
    annotations_file = os.path.join(dataset_dir, "protein_annotations.csv")
    
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        return False
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}")
    df = pd.read_csv(annotations_file)
    
    # Determine expression file and metadata file
    expression_file = None
    if dataset_dir.endswith("example"):
        expression_file = os.path.join(dataset_dir, "expression.csv")
    elif dataset_dir.endswith("collarectal"):
        expression_file = os.path.join(dataset_dir, "GSE251845_htseq_raw_counts.csv")
    
    if not expression_file or not os.path.exists(expression_file):
        print(f"Expression file not found for {dataset_dir}")
        return False
    
    # Load expression data
    print(f"Loading expression data from {expression_file}")
    expr_df = pd.read_csv(expression_file)
    
    print(f"Analyzing expression data...")
    
    # Get column with gene/protein IDs from expression data
    if 'gene_id' in expr_df.columns:
        gene_id_col = 'gene_id'
    else:
        gene_id_col = expr_df.columns[0]  # First column usually contains gene IDs
    
    # Process expression data to identify potential targets
    expr_metrics = {}
    
    # For datasets with multiple samples
    if len(expr_df.columns) > 2:  # Multiple samples
        # Get sample columns (exclude gene ID column)
        sample_cols = [col for col in expr_df.columns if col != gene_id_col]
        
        # Calculate mean expression across all samples
        expr_df['mean_expr'] = expr_df[sample_cols].mean(axis=1)
        
        # Calculate coefficient of variation (CV) to assess expression stability
        # Add a small value to avoid division by zero
        expr_df['std_expr'] = expr_df[sample_cols].std(axis=1)
        expr_df['cv'] = expr_df['std_expr'] / (expr_df['mean_expr'] + 1)
        
        # Calculate expression percentile (0-100)
        expr_df['expr_percentile'] = expr_df['mean_expr'].rank(pct=True) * 100
        
        # Calculate stability score (reverse of CV) - lower CV means more stable
        expr_df['stability'] = 1 - expr_df['cv'].rank(pct=True)
        
        # Create metrics dictionary
        for _, row in expr_df.iterrows():
            gene_id = row[gene_id_col]
            # Clean gene ID if needed
            if isinstance(gene_id, str) and '"' in gene_id:
                gene_id = gene_id.strip('"')
                
            expr_metrics[gene_id] = {
                'mean_expr': row['mean_expr'],
                'expr_percentile': row['expr_percentile'],
                'stability': row['stability']
            }
    
    # Save original scores
    df['base_score'] = df['immunogenicity_score']
    
    # Define expression-based bonuses
    # The approach here is to reward:
    # 1. High expression (high percentile)
    # 2. Stable expression (high stability)
    
    print("Enhancing immunogenicity scores with expression data...")
    high_expr_count = 0
    stable_expr_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        protein_id = row['protein_id']
        gene_id = None
        
        # Extract gene ID from protein ID
        if protein_id.startswith('ENSP'):
            gene_id = f"ENSG{protein_id[4:]}"
        elif protein_id.startswith('PROT_'):
            gene_id = protein_id[5:]  # Remove PROT_ prefix
        
        # Apply expression-based bonuses
        expression_bonus = 0
        if gene_id in expr_metrics:
            metrics = expr_metrics[gene_id]
            
            # High expression bonus (up to 0.15)
            if metrics['expr_percentile'] > 90:  # Top 10%
                expression_bonus += 0.15
                high_expr_count += 1
            elif metrics['expr_percentile'] > 75:  # Top 25%
                expression_bonus += 0.10
                high_expr_count += 1
            elif metrics['expr_percentile'] > 50:  # Top 50%
                expression_bonus += 0.05
                
            # Expression stability bonus (up to 0.05)
            if metrics['stability'] > 0.9:  # Very stable
                expression_bonus += 0.05
                stable_expr_count += 1
            elif metrics['stability'] > 0.75:  # Reasonably stable
                expression_bonus += 0.03
                stable_expr_count += 1
                
            # Apply expression bonus
            if expression_bonus > 0:
                df.at[index, 'immunogenicity_score'] = min(1.0, row['immunogenicity_score'] + expression_bonus)
    
    # Add metadata about known cancer antigens and mutational hotspots
    # This considers colorectal cancer (CRC) specifically
    
    # Well-known CRC antigens and commonly mutated genes
    crc_genes = {
        # Membrane proteins highly relevant to CRC
        'EGFR': 0.2,      # Overexpressed in CRC, target for cetuximab
        'ERBB2': 0.2,     # HER2, amplified in some CRC
        'ERBB3': 0.2,     # HER3, dimerizes with HER2
        'MET': 0.2,       # c-MET, often overexpressed
        'EPCAM': 0.2,     # Epithelial cell adhesion molecule
        'CEA': 0.2,       # Carcinoembryonic antigen
        'CEACAM5': 0.2,   # CEA related
        'GUCY2C': 0.2,    # Intestine-specific receptor
        
        # Commonly mutated genes in CRC
        'APC': 0.2,       # Mutated in ~80% of CRCs
        'TP53': 0.2,      # Mutated in ~60% of CRCs
        'KRAS': 0.2,      # Mutated in ~40% of CRCs
        'BRAF': 0.2,      # Mutated in ~10% of CRCs, V600E hotspot
        'PIK3CA': 0.2,    # Mutated in ~15% of CRCs
        'SMAD4': 0.2,     # Mutated in ~10-15% of CRCs
        'PTEN': 0.15,     # Loss in ~30% of CRCs
        'FBXW7': 0.15,    # Mutated in ~10% of CRCs
        
        # Cancer-testis antigens
        'MAGE': 0.15,     # MAGE family
        'GAGE': 0.15,     # GAGE family
        'SSX': 0.15,      # SSX family
        'NY-ESO-1': 0.15, # Cancer-testis antigen 1
        'CTAG1': 0.15,    # Cancer-testis antigen 1 (alternate name)
    }
    
    # Gene families relevant to cancer
    relevant_families = {
        # Tyrosine kinase receptors
        'EGFR': 0.1,
        'ERBB': 0.1,
        'FGF': 0.1,
        'PDGF': 0.1,
        'VEGF': 0.1,
        'IGF': 0.1,
        
        # WNT pathway (critical in CRC)
        'WNT': 0.1,
        'FZD': 0.1,
        'LRP': 0.1,
        'AXIN': 0.1,
        'GSK3': 0.1,
        'CTNNB': 0.1,    # Beta-catenin
        
        # MMR genes (relevant for MSI-high CRC)
        'MLH': 0.1,
        'MSH': 0.1,
        'PMS': 0.1,
    }
    
    # Apply cancer-specific bonuses
    print("Adding cancer biology knowledge...")
    crc_gene_count = 0
    family_count = 0
    
    for index, row in df.iterrows():
        protein_name = row['protein_name'].upper()
        gene_symbol = None
        
        # Extract gene symbol from protein name
        if 'protein' in protein_name.lower():
            gene_symbol = protein_name.split(' protein')[0].upper()
        else:
            gene_symbol = protein_name
        
        # Check if protein matches a known CRC gene
        for gene, bonus in crc_genes.items():
            if gene in gene_symbol:
                df.at[index, 'immunogenicity_score'] = min(1.0, row['immunogenicity_score'] + bonus)
                crc_gene_count += 1
                break
                
        # Check if protein is part of a relevant gene family
        for family, bonus in relevant_families.items():
            if family in gene_symbol:
                df.at[index, 'immunogenicity_score'] = min(1.0, row['immunogenicity_score'] + bonus)
                family_count += 1
                break
    
    print(f"Added high expression bonus to {high_expr_count} proteins")
    print(f"Added expression stability bonus to {stable_expr_count} proteins")
    print(f"Added bonus to {crc_gene_count} proteins matching known CRC genes")
    print(f"Added bonus to {family_count} proteins from relevant gene families")
    
    # Save updated annotations
    df.to_csv(annotations_file, index=False)
    print(f"Saved enhanced annotations to {annotations_file}")
    
    # Show top proteins by score
    top_proteins = df.sort_values('immunogenicity_score', ascending=False).head(10)
    print("\nTop 10 proteins by enhanced immunogenicity score:")
    for _, row in top_proteins.iterrows():
        orig_score = row['base_score']
        final_score = row['immunogenicity_score']
        improvement = final_score - orig_score
        print(f"{row['protein_name']} ({row['cellular_location']}) - Score: {final_score:.2f} (↑{improvement:.2f})")
    
    return True

def main():
    """Enhance immunogenicity scores with additional factors"""
    print("Enhancing immunogenicity scores for cancer vaccine targets...")
    
    # Define data directory
    data_dir = "data"
    
    # Process example dataset
    print("\nProcessing example dataset...")
    add_expression_data(os.path.join(data_dir, "example"))
    
    # Process colorectal dataset
    print("\nProcessing colorectal dataset...")
    add_expression_data(os.path.join(data_dir, "collarectal"))
    
    print("\nDone!")

if __name__ == "__main__":
    main() 