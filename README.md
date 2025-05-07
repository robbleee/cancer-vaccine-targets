# Early-Stage Antigen Prioritizer Tool

A streamlit application for identifying and ranking potential protein targets for prophylactic cancer vaccines based on gene expression data and predicted immunogenicity features.

## Overview

This tool analyzes gene expression data to:
1. Identify genes that are upregulated in early disease stages (precancer or early cancer) compared to normal tissue
2. Map these genes to their corresponding proteins
3. Score and rank proteins based on their potential as cancer vaccine targets using factors such as:
   - Fold change in expression
   - Statistical significance of differential expression
   - Predicted cellular location (membrane proteins are prioritized)
4. Provide AI-powered expert review of the results using OpenAI's GPT models

## Installation

1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Set up OpenAI API access (optional, for AI review feature):
   - Create a `.streamlit/secrets.toml` file with your OpenAI API key:
   ```
   [openai]
   api_key = "your-api-key-here"
   ```

## Usage

1. Start the application:
```
streamlit run app/app.py
```

2. Use the sidebar to:
   - Choose a data source:
     - Example data: A small synthetic dataset with 15 genes
     - Real-world test dataset: A larger dataset with 30 cancer-related genes and 18 patient samples
     - Upload your own data files
   - Select control group (e.g., 'normal') and case group (e.g., 'precancer' or 'early_cancer')
   - Set threshold parameters for filtering (minimum fold change, maximum p-value)

3. Click the "Prioritize Targets" button to run the analysis

4. View the results:
   - Ranked table of potential protein targets
   - AI expert review with assessment of findings, clinical relevance, and recommended next steps
   - Visualizations of expression patterns
   - Priority factor radar charts for top candidates
   - Download results as CSV

## Data Format

If uploading your own data, you need to prepare the following CSV files:

1. **expression.csv**: Gene expression data
   - First column: gene_id
   - Remaining columns: Sample IDs matching those in metadata.csv

2. **metadata.csv**: Sample information
   - sample_id: Sample identifiers
   - condition: Group labels (e.g., 'normal', 'precancer', 'early_cancer')

3. **gene_protein_map.csv**: Gene to protein mapping
   - gene_id: Gene identifiers matching those in expression.csv
   - protein_id: Corresponding protein identifiers

4. **protein_annotations.csv**: Protein annotations
   - protein_id: Protein identifiers matching those in gene_protein_map.csv
   - protein_name: Human-readable protein names
   - cellular_location: Subcellular localization (e.g., 'membrane', 'secreted', 'nucleus')

## Included Datasets

### Example Dataset
A small synthetic dataset with 15 genes across 12 samples. Useful for quick testing and demonstration.

### Real-World Test Dataset
A more comprehensive dataset containing expression data for 30 cancer-related genes from 18 patient samples. It includes many known cancer driver genes and therapeutic targets with proper annotations.

## AI Expert Review

The AI review feature uses OpenAI's GPT models to provide expert analysis of the results, including:
- Overall assessment of identified targets
- Analysis of the most promising target
- Clinical relevance to cancer vaccine development
- Limitations of the current analysis
- Recommended next steps for validation

This feature requires a valid OpenAI API key configured in the Streamlit secrets. # cancer-vaccine-targets
