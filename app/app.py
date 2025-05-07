import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import os
import json

# Add proper path handling
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Page setup
st.set_page_config(page_title="Early-Stage Antigen Prioritizer", layout="wide")

# OpenAI API configuration - with better error handling
openai_available = False
try:
    from openai import OpenAI
    # Check for secrets without directly accessing st.secrets
    try:
        _ = st.secrets["openai"]["api_key"]
        client = OpenAI(api_key=st.secrets["openai"]["api_key"])
        openai_available = True
    except (KeyError, FileNotFoundError):
        st.warning("OpenAI API key not found in secrets. AI review feature will be disabled.")
except ImportError:
    st.warning("OpenAI package not installed. AI review feature will be disabled.")

def get_json_from_prompt(prompt: str) -> dict:
    """Helper function to call OpenAI and return the JSON-parsed response."""
    if not openai_available:
        return {"error": "OpenAI API key not configured"}
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable cancer immunologist who returns valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

def get_ai_review(results_df, case_group, control_group):
    """Generate an AI review of the analysis results."""
    if not openai_available:
        return None
    
    # Create a summary of the results
    top_genes = results_df.head(5)[['rank', 'gene_id', 'protein_name', 'fold_change', 'p_value', 'cellular_location', 'immunogenicity_score']].to_dict(orient='records')
    
    # Format the prompt
    prompt = f"""
    As a cancer immunologist, review these potential cancer vaccine targets identified when comparing {case_group} vs {control_group} tissue samples.

    Here are the top 5 ranked targets:
    {json.dumps(top_genes, indent=2)}

    Please provide your expert analysis in JSON format with the following fields:
    1. "overall_assessment": A paragraph summarizing what these results indicate
    2. "top_target_review": Analysis of the most promising target and why
    3. "clinical_relevance": How these findings might inform cancer vaccine development
    4. "limitations": Key limitations to consider with this analysis
    5. "next_steps": Recommended next steps for validation
    
    Keep each field concise (2-4 sentences max).
    """
    
    return get_json_from_prompt(prompt)

# Title and introduction
st.title("Early-Stage Antigen Prioritizer Tool")
st.markdown("""
This tool helps identify and rank potential protein targets for prophylactic cancer vaccines 
based on gene expression data and predicted immunogenicity features.
""")

# Add explanation of updated methodology
with st.expander("About Our Enhanced Scoring System"):
    st.markdown("""
    ### Enhanced Cancer Vaccine Target Identification
    
    Our updated scoring system (0-1 scale) represents a significant improvement in cancer vaccine target identification:
    
    **Key Improvements:**
    
    1. **Accurate protein locations**: 
       - Replaced synthetic location data with real cellular locations from BioMart, UniProt, and Human Protein Atlas
       - Increased real data coverage from <1% to 36.3% across the colorectal dataset
    
    2. **Cancer-specific scoring adjustment**:
       - Maintained highest base scores for membrane/secreted proteins (1.0)
       - Assigned appropriate scores to intracellular proteins: cytoplasmic (0.15), nuclear (0.05)
       - Added targeted bonuses for cancer-specific genes (+0.2) and pathway members (+0.1)
    
    3. **Expression-based refinements**:
       - Added bonuses for highly expressed genes (+0.05 to +0.15)
       - Added stability bonuses for consistently expressed genes (+0.03 to +0.05)
    
    The final immunogenicity score (capped at 1.0) combines these factors to identify both traditional vaccine targets 
    (accessible membrane proteins) and cancer-relevant proteins that might be presented to T-cells.
    """)

# Sidebar for inputs
st.sidebar.header("Data Input")

# Option for using pre-loaded datasets
data_option = st.sidebar.radio(
    "Select data source:",
    ("Use example data", "Use real-world test dataset", "Upload your own data")
)

expression_data = None
metadata = None
gene_protein_map = None
protein_annotations = None

if data_option == "Use example data":
    # Load example data with correct paths
    try:
        expression_data = pd.read_csv(os.path.join(DATA_DIR, "example", "expression.csv"))
        metadata = pd.read_csv(os.path.join(DATA_DIR, "example", "metadata.csv"))
        gene_protein_map = pd.read_csv(os.path.join(DATA_DIR, "example", "gene_protein_map.csv"))
        protein_annotations = pd.read_csv(os.path.join(DATA_DIR, "example", "protein_annotations.csv"))
        
        st.sidebar.success("Example data loaded successfully!")
        st.sidebar.info("This is a small synthetic dataset with gene expression data across multiple samples.")
    except FileNotFoundError as e:
        st.error(f"Error loading example data: {str(e)}")
        st.info(f"Looking for data in: {os.path.join(DATA_DIR, 'example')}")
        st.info(f"Current working directory: {os.getcwd()}")
elif data_option == "Use real-world test dataset":
    # Load the real-world test dataset (colorectal cancer data)
    try:
        # The expression data file has a different name in the colorectal dataset
        expression_data = pd.read_csv(os.path.join(DATA_DIR, "collarectal", "GSE251845_htseq_raw_counts.csv"))
        metadata = pd.read_csv(os.path.join(DATA_DIR, "collarectal", "metadata.csv"))
        gene_protein_map = pd.read_csv(os.path.join(DATA_DIR, "collarectal", "gene_protein_map.csv"))
        protein_annotations = pd.read_csv(os.path.join(DATA_DIR, "collarectal", "protein_annotations.csv"))
        
        st.sidebar.success("Colorectal cancer dataset loaded successfully!")
        st.sidebar.info("""
        This dataset contains gene expression data from colorectal cancer patients,
        with enhanced protein location annotations and immunogenicity scoring.
        """)
    except FileNotFoundError as e:
        st.error(f"Error loading colorectal cancer data: {str(e)}")
        st.info(f"Looking for data in: {os.path.join(DATA_DIR, 'collarectal')}")
        st.info(f"Current working directory: {os.getcwd()}")
else:
    # File uploaders
    expression_file = st.sidebar.file_uploader("Upload expression data (CSV)", type=["csv"])
    metadata_file = st.sidebar.file_uploader("Upload metadata (CSV)", type=["csv"])
    gene_protein_file = st.sidebar.file_uploader("Upload gene-protein mapping (CSV)", type=["csv"])
    protein_annotations_file = st.sidebar.file_uploader("Upload protein annotations (CSV)", type=["csv"])
    
    if expression_file and metadata_file and gene_protein_file and protein_annotations_file:
        expression_data = pd.read_csv(expression_file)
        metadata = pd.read_csv(metadata_file)
        gene_protein_map = pd.read_csv(gene_protein_file)
        protein_annotations = pd.read_csv(protein_annotations_file)
        
        st.sidebar.success("All files uploaded successfully!")

# Parameter selection
st.sidebar.header("Analysis Parameters")

if metadata is not None:
    # Get unique conditions from metadata
    conditions = metadata['condition'].unique().tolist()
    
    # Dropdowns for control and case selection
    control_group = st.sidebar.selectbox("Select control group:", conditions, index=0)
    
    # Filter out the control group from the case options
    case_options = [c for c in conditions if c != control_group]
    case_group = st.sidebar.selectbox("Select case group:", case_options, index=0)
    
    # Threshold parameters
    min_fold_change = st.sidebar.slider("Minimum fold change:", 1.0, 10.0, 2.0, 0.1)
    max_pvalue = st.sidebar.slider("Maximum p-value:", 0.001, 0.1, 0.05, 0.001)

    # Analysis button
    analyze_button = st.sidebar.button("Prioritize Targets")
    
    # Main content
    if analyze_button and expression_data is not None and metadata is not None:
        with st.spinner("Analyzing data..."):
            # Get sample IDs for the selected groups
            control_samples = metadata[metadata['condition'] == control_group]['sample_id'].tolist()
            case_samples = metadata[metadata['condition'] == case_group]['sample_id'].tolist()
            
            # Check if we have samples for both groups
            if len(control_samples) == 0 or len(case_samples) == 0:
                st.error(f"No samples found for {control_group if len(control_samples) == 0 else case_group}.")
            else:
                # Perform simplified differential expression analysis
                def perform_differential_expression(expr_data, control_samples, case_samples, min_fc, max_p):
                    # Create a results dataframe
                    results = pd.DataFrame(index=expr_data['gene_id'])
                    
                    # Calculate mean expression for each group
                    results['mean_control'] = expr_data[control_samples].mean(axis=1)
                    results['mean_case'] = expr_data[case_samples].mean(axis=1)
                    
                    # Calculate fold change
                    results['fold_change'] = results['mean_case'] / results['mean_control']
                    
                    # Simplified p-value calculation using t-test
                    p_values = []
                    
                    for gene in expr_data['gene_id']:
                        gene_data = expr_data.loc[expr_data['gene_id'] == gene]
                        control_expr = gene_data[control_samples].values.flatten()
                        case_expr = gene_data[case_samples].values.flatten()
                        
                        # Perform t-test
                        t_stat, p_val = stats.ttest_ind(case_expr, control_expr)
                        p_values.append(p_val)
                    
                    results['p_value'] = p_values
                    
                    # Filter based on thresholds
                    filtered_results = results[(results['fold_change'] >= min_fc) & 
                                              (results['p_value'] <= max_p)]
                    
                    return filtered_results.reset_index()
                
                # Set gene_id as the first column (index column)
                expression_data = expression_data.set_index('gene_id', drop=False)
                
                # Perform differential expression analysis
                de_results = perform_differential_expression(
                    expression_data, 
                    control_samples, 
                    case_samples, 
                    min_fold_change, 
                    max_pvalue
                )
                
                # Map genes to proteins
                def map_genes_to_proteins(de_results, gene_protein_map, protein_annotations):
                    # Merge DE results with gene-protein mapping
                    merged = pd.merge(
                        de_results,
                        gene_protein_map,
                        on='gene_id',
                        how='left'
                    )
                    
                    # Merge with annotations
                    final = pd.merge(
                        merged,
                        protein_annotations,
                        on='protein_id',
                        how='left'
                    )
                    
                    return final
                
                # Map genes to proteins
                mapped_results = map_genes_to_proteins(
                    de_results, 
                    gene_protein_map, 
                    protein_annotations
                )
                
                # Calculate immunogenicity score
                def calculate_immunogenicity_score(mapped_data, expr_data, case_samples):
                    # Simply use the existing immunogenicity_score from the annotations file
                    # No need to calculate it - just ensure it exists
                    if 'immunogenicity_score' not in mapped_data.columns:
                        st.error("Protein annotations file missing 'immunogenicity_score' column. Using defaults.")
                        # If missing, provide a basic score as fallback
                        mapped_data['immunogenicity_score'] = 0
                        # Apply basic location-based scoring as fallback
                        mapped_data.loc[mapped_data['cellular_location'] == 'membrane', 'immunogenicity_score'] = 3
                        mapped_data.loc[mapped_data['cellular_location'] == 'secreted', 'immunogenicity_score'] = 3
                    
                    return mapped_data
                
                # Calculate immunogenicity scores
                scored_results = calculate_immunogenicity_score(mapped_results, expression_data, case_samples)
                
                # Calculate final rank
                def calculate_final_rank(scored_data):
                    # Create a combined score for ranking
                    # Adjust for the immunogenicity_score being on a 0-3 scale by normalizing it
                    scored_data['immunogenicity_norm'] = scored_data['immunogenicity_score'] / 3.0
                    
                    scored_data['combined_score'] = (
                        scored_data['fold_change'] * 0.4 +
                        (1 - scored_data['p_value']) * 0.3 +
                        scored_data['immunogenicity_norm'] * 0.3
                    )
                    
                    # Sort by combined score and add rank
                    ranked_data = scored_data.sort_values(by='combined_score', ascending=False)
                    ranked_data['rank'] = range(1, len(ranked_data) + 1)
                    
                    return ranked_data
                
                # Calculate final ranks
                final_results = calculate_final_rank(scored_results)
                
                # Display results
                st.success("Analysis complete!")
                
                # Show result counts
                st.write(f"Found {len(final_results)} potential targets for {case_group} vs {control_group}.")
                
                # Add methodology explanation section
                with st.expander("Immunogenicity Scoring Methodology"):
                    st.markdown("""
                    ### How Immunogenicity Scores Are Calculated
                    
                    The immunogenicity score represents a protein's potential as a cancer vaccine target on a scale of 0 to 1, where higher scores indicate better vaccine targets.
                    
                    #### 1. Protein Location Data Collection
                    We've significantly improved protein location annotation through:
                    - **BioMart querying:** Direct access to Gene Ontology cellular component data for over 60,000 genes
                    - **UniProt integration:** Additional location data from manually curated protein database
                    - **Human Protein Atlas:** Experimental validation of protein locations
                    - **Real data coverage:** Achieved 36.3% real data coverage, avoiding synthetic data generation
                    
                    #### 2. Cellular Location Base Scores (0-1 scale)
                    - **Plasma Membrane (1.0)**: Highest priority as these proteins are directly accessible to antibodies
                    - **Secreted/Extracellular (1.0)**: Also highly accessible for antibody targeting
                    - **Cytoplasm (0.15)**: Limited accessibility, primarily for T-cell responses
                    - **Nucleus (0.05)**: Low accessibility, but may contain cancer-specific mutations
                    - **Mitochondrion (0.05)**: Low accessibility proteins
                    - **ER/Golgi (0.10)**: Intermediate accessibility through processing pathways
                    - **Lysosome/Endosome (0.10)**: Intermediate accessibility
                    - **Other/Unknown (0.0)**: Unknown locations receive no base score
                    
                    #### 3. Expression-Based Bonuses (additive)
                    - **High expression bonus**: 
                      - Top 10% expressed genes: +0.15
                      - Top 25% expressed genes: +0.10
                      - Top 50% expressed genes: +0.05
                    - **Expression stability bonus**:
                      - Very stable expression (low variation): +0.05
                      - Moderately stable expression: +0.03
                    
                    #### 4. Cancer-Specific Gene Bonuses (additive)
                    - **Known colorectal cancer genes** (+0.20): EGFR, ERBB2, TP53, KRAS, APC, etc.
                    - **Relevant gene families** (+0.10):
                      - WNT pathway genes (critical in colorectal cancer)
                      - Tyrosine kinase receptors (EGFR, ERBB family, etc.)
                      - Mismatch repair genes (relevant for MSI-high colorectal cancer)
                    
                    #### 5. Final Score Calculation
                    Final score = Base location score + Expression bonuses + Cancer gene bonuses
                    
                    All scores are capped at 1.0, and a small random variation (±0.10) is added for ranking diversity. This approach ensures all scores remain on a consistent 0-1 scale.
                    
                    This combined approach allows us to prioritize both accessible proteins (membrane/secreted) as well as cancer-relevant intracellular proteins that may be presented via MHC for T-cell responses.
                    """)
                
                # Display the results table
                st.subheader("Ranked Protein Targets")
                
                # Select columns to display
                display_columns = [
                    'rank', 'gene_id', 'protein_id', 'protein_name', 'fold_change', 'p_value',
                    'cellular_location', 'immunogenicity_score', 'combined_score'
                ]
                
                # Format the table for display
                display_table = final_results[display_columns].copy()
                display_table['fold_change'] = display_table['fold_change'].round(2)
                display_table['p_value'] = display_table['p_value'].map('{:.6f}'.format)
                display_table['combined_score'] = display_table['combined_score'].round(2)
                
                st.dataframe(display_table)
                
                # AI review of results
                st.subheader("AI Expert Review")
                
                if openai_available and not final_results.empty:
                    with st.spinner("Generating AI expert review..."):
                        ai_review = get_ai_review(final_results, case_group, control_group)
                        
                        if ai_review and "error" not in ai_review:
                            # Create expandable sections for each part of the review
                            with st.expander("Overall Assessment", expanded=True):
                                st.write(ai_review.get("overall_assessment", "No assessment available"))
                            
                            with st.expander("Top Target Analysis"):
                                st.write(ai_review.get("top_target_review", "No top target analysis available"))
                            
                            with st.expander("Clinical Relevance"):
                                st.write(ai_review.get("clinical_relevance", "No clinical relevance assessment available"))
                            
                            with st.expander("Limitations"):
                                st.write(ai_review.get("limitations", "No limitations described"))
                            
                            with st.expander("Recommended Next Steps"):
                                st.write(ai_review.get("next_steps", "No next steps provided"))
                        else:
                            error_msg = ai_review.get("error", "Failed to generate AI review") if ai_review else "Failed to generate AI review"
                            st.error(f"AI Review Error: {error_msg}")
                else:
                    if not openai_available:
                        st.info("AI expert review disabled - OpenAI API key not configured.")
                    elif final_results.empty:
                        st.info("No results available for AI review.")
                
                # Visualization
                if not final_results.empty:
                    st.subheader("Visualizations")
                    
                    # Only show top 10 for visualizations
                    top_results = final_results.head(10)
                    
                    # Expression plot for top genes
                    st.write("Expression of Top Ranked Genes")
                    
                    # Create a melted dataframe for plotting
                    plot_data = []
                    
                    for gene in top_results['gene_id'].tolist():
                        gene_expr = expression_data.loc[expression_data['gene_id'] == gene]
                        
                        for sample in control_samples + case_samples:
                            row_dict = {
                                'Gene': gene,
                                'Sample': sample,
                                'Expression': gene_expr[sample].values[0],
                                'Group': 'Control' if sample in control_samples else 'Case'
                            }
                            plot_data.append(row_dict)
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Create a box plot
                    fig = px.box(
                        plot_df, 
                        x='Gene', 
                        y='Expression', 
                        color='Group',
                        title=f"Expression in {control_group} vs {case_group}",
                        color_discrete_map={'Control': 'blue', 'Case': 'red'}
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Feature importance
                    st.write("Target Prioritization Factors")
                    
                    # Create a radar chart
                    radar_data = top_results[['protein_name', 'fold_change', 'p_value', 'immunogenicity_score']].head(5)
                    
                    # Convert p-value to significance score (1 - p_value)
                    radar_data['significance'] = 1 - radar_data['p_value']
                    
                    # Normalize fold change (0-1 scale)
                    max_fc = radar_data['fold_change'].max()
                    radar_data['fold_change_norm'] = radar_data['fold_change'] / max_fc
                    
                    # Normalize immunogenicity score (0-1 scale)
                    max_immune = radar_data['immunogenicity_score'].max()
                    if max_immune > 0:
                        radar_data['immunogenicity_norm'] = radar_data['immunogenicity_score'] / max_immune
                    else:
                        radar_data['immunogenicity_norm'] = radar_data['immunogenicity_score']
                    
                    # Create radar chart
                    for i, row in radar_data.iterrows():
                        scores = [
                            row['fold_change_norm'],
                            row['significance'],
                            row['immunogenicity_norm']
                        ]
                        
                        categories = ['Fold Change', 'Statistical Significance', 'Immunogenicity']
                        
                        # Add protein name to each category
                        labeled_categories = [f"{cat} ({row['protein_name']})" for cat in categories]
                        
                        # Create a dataframe for radar chart
                        radar_df = pd.DataFrame(dict(
                            r=scores,
                            theta=categories
                        ))
                        
                        fig = px.line_polar(
                            radar_df, 
                            r='r', 
                            theta='theta', 
                            line_close=True,
                            title=f"Priority Factors for {row['protein_name']}"
                        )
                        
                        st.plotly_chart(fig)
                    
                    # Add immunogenicity score breakdown visualization
                    st.subheader("Immunogenicity Score Breakdown")
                    st.write("""
                    This visualization shows how different factors contribute to the final immunogenicity score (0-1 scale) for the top-ranked proteins.
                    
                    **Components:**
                    - **Location Base Score**: Initial score based on cellular location (0-1 scale)
                      - Membrane/Secreted: 1.0
                      - Cytoplasm: 0.15
                      - ER/Golgi: 0.10
                      - Nuclear/Mitochondrial: 0.05
                    - **Cancer Gene Bonus**: Additional score for cancer relevance (additive)
                      - Known cancer genes: +0.20
                      - Cancer pathway genes: +0.10
                    - **Expression Bonus**: Score based on expression data (additive)
                      - High expression: +0.05 to +0.15
                      - Expression stability: +0.03 to +0.05
                    - **Other Factors**: Small variations for ranking diversity
                    
                    The sum of all components is capped at 1.0, which is why membrane and secreted proteins have an advantage when it comes to reaching maximum scores.
                    """)
                    
                    # Extract the top 5 proteins for breakdown
                    top5_proteins = final_results.head(5)
                    
                    # Create data for the breakdown chart
                    breakdown_data = []
                    
                    for _, row in top5_proteins.iterrows():
                        protein_name = row['protein_name']
                        location = row['cellular_location']
                        
                        # Base score by location
                        location_scores = {
                            'membrane': 1.0,
                            'secreted': 1.0,
                            'cytoplasm': 0.15,
                            'nucleus': 0.05,
                            'mitochondria': 0.05,
                            'er': 0.1,
                            'golgi': 0.1,
                            'lysosome': 0.1,
                            'endosome': 0.1,
                            'unknown': 0.0
                        }
                        
                        base_score = location_scores.get(location, 0.0)
                        
                        # Estimate expression bonus based on gene ID
                        expression_bonus = 0.0
                        # Check if it matches any known CRC genes
                        crc_genes = {'EGFR', 'ERBB2', 'ERBB3', 'MET', 'EPCAM', 
                                     'CEA', 'CEACAM5', 'GUCY2C', 'APC', 'TP53', 
                                     'KRAS', 'BRAF', 'PIK3CA', 'SMAD4', 'PTEN', 'FBXW7'}
                        
                        gene_families = {'WNT', 'FZD', 'AXIN', 'CTNNB', 'EGFR', 'ERBB', 
                                        'FGF', 'PDGF', 'MLH', 'MSH', 'PMS'}
                        
                        cancer_bonus = 0.0
                        protein_upper = protein_name.upper()
                        
                        # Check for CRC-specific gene matches
                        for gene in crc_genes:
                            if gene in protein_upper:
                                cancer_bonus = 0.2
                                break
                                
                        # Check for gene family matches if no CRC gene match
                        if cancer_bonus == 0.0:
                            for family in gene_families:
                                if family in protein_upper:
                                    cancer_bonus = 0.1
                                    break
                        
                        # Estimate expression bonus (difference between total and other components)
                        remaining_bonus = max(0, row['immunogenicity_score'] - base_score - cancer_bonus)
                        expression_bonus = min(remaining_bonus, 0.15)  # Cap at max possible expression bonus
                        
                        # Create records for the chart
                        breakdown_data.append({
                            'Protein': protein_name,
                            'Component': 'Location Base Score',
                            'Value': base_score
                        })
                        
                        if cancer_bonus > 0:
                            breakdown_data.append({
                                'Protein': protein_name,
                                'Component': 'Cancer Gene Bonus',
                                'Value': cancer_bonus
                            })
                            
                        if expression_bonus > 0:
                            breakdown_data.append({
                                'Protein': protein_name,
                                'Component': 'Expression Bonus',
                                'Value': expression_bonus
                            })
                            
                        # Account for any remaining score (could be from random variation)
                        remaining = max(0, row['immunogenicity_score'] - base_score - cancer_bonus - expression_bonus)
                        if remaining > 0.01:  # Only show if significant
                            breakdown_data.append({
                                'Protein': protein_name,
                                'Component': 'Other Factors',
                                'Value': remaining
                            })
                    
                    # Create dataframe for plotting
                    if breakdown_data:
                        breakdown_df = pd.DataFrame(breakdown_data)
                        
                        # Create stacked bar chart
                        fig = px.bar(
                            breakdown_df,
                            x='Protein',
                            y='Value',
                            color='Component',
                            title='Immunogenicity Score Components',
                            labels={'Value': 'Score Component', 'Protein': 'Protein Name'},
                            color_discrete_map={
                                'Location Base Score': '#1f77b4',
                                'Cancer Gene Bonus': '#ff7f0e',
                                'Expression Bonus': '#2ca02c',
                                'Other Factors': '#d62728'
                            }
                        )
                        
                        # Add text annotations to show total score
                        for protein in top5_proteins['protein_name']:
                            score = top5_proteins.loc[top5_proteins['protein_name'] == protein, 'immunogenicity_score'].values[0]
                            fig.add_annotation(
                                x=protein,
                                y=min(score + 0.05, 1.0),
                                text=f"Total: {score:.2f}",
                                showarrow=False
                            )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title="Protein",
                            yaxis_title="Score Component",
                            yaxis_range=[0, 1.1],
                            legend_title="Component"
                        )
                        
                        st.plotly_chart(fig)
                    
                    # Download results
                    csv = final_results.to_csv(index=False)
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "antigen_prioritizer_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
else:
    st.info("Please select a data source and upload the required files to begin analysis.") 