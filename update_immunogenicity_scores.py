#!/usr/bin/env python3
import pandas as pd
import os
from tqdm import tqdm

def update_scores_for_dataset(dataset_dir):
    """Update immunogenicity scores for a dataset"""
    annotations_file = os.path.join(dataset_dir, "protein_annotations.csv")
    
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        return False
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}")
    df = pd.read_csv(annotations_file)
    
    # Original scores
    print("Original score distribution:")
    print(df.groupby('cellular_location')['immunogenicity_score'].mean())
    
    # Count by location
    location_counts = df['cellular_location'].value_counts()
    total = len(df)
    print("\nLocation distribution:")
    for loc, count in location_counts.items():
        print(f"- {loc}: {count} proteins ({count/total*100:.1f}%)")
    
    # Update scores
    print("\nUpdating immunogenicity scores for cancer vaccine targets...")
    
    # Make a copy of original scores for comparison
    df['original_score'] = df['immunogenicity_score']
    
    # Update each location with new cancer-appropriate score
    # The base is a deterministic score without the random variation
    for index, row in tqdm(df.iterrows(), total=len(df)):
        location = row['cellular_location']
        protein_id = row['protein_id']
        
        # Base scores appropriate for cancer vaccines
        if location == 'membrane':
            base_score = 0.85  # Unchanged - accessible to both antibodies and T-cells
        elif location == 'secreted':
            base_score = 0.70  # Reduced from 0.80 - can cause off-target effects
        elif location == 'cytoplasm':
            base_score = 0.60  # Increased from 0.30 - many cancer mutations are cytoplasmic
        elif location == 'nucleus':
            base_score = 0.50  # Increased from 0.20 - many cancer mutations are in nuclear proteins
        elif location == 'mitochondria':
            base_score = 0.30  # Increased from 0.25 - less common but still relevant
        else:
            base_score = 0.0   # Unknown locations
        
        # Add small variation based on protein ID to create diversity in rankings
        # Same formula as original to maintain consistency
        if base_score > 0:
            variation = (hash(protein_id) % 20 - 10) / 100
            score = max(0.0, min(1.0, base_score + variation))
        else:
            score = 0.0
            
        df.at[index, 'immunogenicity_score'] = score
    
    # New scores
    print("\nNew score distribution:")
    print(df.groupby('cellular_location')['immunogenicity_score'].mean())
    
    # Save updated annotations
    df.drop(columns=['original_score'], inplace=True)
    df.to_csv(annotations_file, index=False)
    print(f"Saved updated annotations to {annotations_file}")
    
    return True

def main():
    """Update immunogenicity scores for all datasets"""
    print("Updating immunogenicity scores for cancer vaccine targets...")
    
    # Define data directory
    data_dir = "data"
    
    # Process example dataset
    print("\nProcessing example dataset...")
    update_scores_for_dataset(os.path.join(data_dir, "example"))
    
    # Process colorectal dataset
    print("\nProcessing colorectal dataset...")
    update_scores_for_dataset(os.path.join(data_dir, "collarectal"))
    
    print("\nDone!")

if __name__ == "__main__":
    main() 