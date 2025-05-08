import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- Custom CSS for Modern UI ---
def apply_modern_styles():
    st.markdown("""
    <style>
    /* Main container padding and styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    h1 {
        font-size: 2.4rem !important;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(109, 40, 217, 0.2);
        margin-bottom: 1.5rem;
    }
    h2 {
        font-size: 1.8rem !important;
        margin-top: 2rem;
    }
    h3 {
        font-size: 1.4rem !important;
    }
    
    /* Card-like elements */
    .stAlert, div.stForm, div[data-testid="stExpander"] > div:first-child {
        border-radius: 10px !important;
        border: 1px solid rgba(109, 40, 217, 0.2) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Info and warning boxes */
    div[data-testid="stInfo"] {
        background-color: #EDE9FE !important;
        border-left-color: #6D28D9 !important;
    }
    
    /* Better spacing for metric elements */
    div[data-testid="stMetric"] {
        background-color: #EDE9FE;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Methodology cards */
    .methodology-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        margin-bottom: 1.5rem;
    }
    
    /* Highlight boxes */
    .highlight-box {
        background-color: #EDE9FE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Equation styling */
    .equation {
        background-color: #EDE9FE;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Reference styling */
    .reference {
        font-size: 0.9rem;
        border-left: 3px solid #6D28D9;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page configuration ---
st.set_page_config(
    page_title="Methodology - Early-Stage Antigen Prioritizer",
    layout="wide"
)

# Apply modern styling
apply_modern_styles()

# --- Title and introduction ---
st.title("Calculation Methodology")
st.markdown("""
This page provides a detailed explanation of the calculation methods used to identify and rank potential 
protein targets for prophylactic cancer vaccines in the Early-Stage Antigen Prioritizer Tool.
""")

# --- Overview of the Pipeline ---
st.header("Analysis Pipeline Overview")
st.markdown("""
The target prioritization process follows a sequential pipeline of data analysis and scoring:

1. **Differential Expression Analysis**: Comparing gene expression between case and control groups
2. **Gene-to-Protein Mapping**: Linking differentially expressed genes to their corresponding proteins
3. **Immunogenicity Score Calculation**: Assessing protein features relevant to vaccine target potential
4. **Combined Score Calculation**: Integrating multiple factors to produce a final priority ranking
""")

# --- Section 1: Differential Expression Analysis ---
st.header("1. Differential Expression Analysis")
st.markdown("""
The first step identifies genes with significant expression differences between the case group (e.g., tumor) 
and control group (e.g., normal tissue).

**Inputs:**
- Gene expression matrix (genes × samples)
- Sample metadata with condition labels

**Method:**
- For each gene, we calculate:
  - Mean expression in control group
  - Mean expression in case group
  - Fold change: ratio of case mean / control mean
  - Statistical significance: Welch's t-test (unequal variance t-test)

**Filtering criteria:**
- Minimum fold change (user-adjustable, default: 2.0)
- Maximum p-value (user-adjustable, default: 0.05)

**Formula:**

$$\\text{Fold Change} = \\frac{\\text{Mean Expression in Case}}{\\text{Mean Expression in Control}}$$

$$p\\text{-value} = \\text{Welch's t-test}(\\text{Case samples}, \\text{Control samples})$$
""")

# --- Visualization for Differential Expression ---
st.markdown("### Differential Expression Visualization")

# Create example data for visualization
np.random.seed(42)
example_genes = 100
example_data = pd.DataFrame({
    'log2_fold_change': np.random.normal(0, 1.5, example_genes),
    'neg_log10_pvalue': -np.log10(np.random.beta(1, 10, example_genes)),
    'gene': [f'Gene_{i}' for i in range(example_genes)]
})

# Add significance coloring
example_data['significant'] = (example_data['log2_fold_change'].abs() > 1) & (example_data['neg_log10_pvalue'] > 1.3)
example_data['color'] = example_data['significant'].map({True: 'significant', False: 'not significant'})

# Create volcano plot
fig = px.scatter(
    example_data,
    x='log2_fold_change',
    y='neg_log10_pvalue',
    color='color',
    color_discrete_map={'significant': 'red', 'not significant': 'gray'},
    hover_name='gene',
    labels={
        'log2_fold_change': 'Log2 Fold Change',
        'neg_log10_pvalue': '-Log10 p-value'
    },
    title="Example Volcano Plot: Differential Expression Analysis"
)

# Add threshold lines
fig.add_hline(y=1.3, line_dash="dash", line_color="gray")  # -log10(0.05) ≈ 1.3
fig.add_vline(x=1, line_dash="dash", line_color="gray")
fig.add_vline(x=-1, line_dash="dash", line_color="gray")

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Explanation of Volcano Plot:**
- Each point represents a gene
- X-axis: Log2 Fold Change (measure of effect size)
- Y-axis: -Log10 p-value (measure of statistical significance)
- Red points: Genes passing both fold change and p-value thresholds
- Dashed lines: Typical significance thresholds (|Log2 FC| > 1, p-value < 0.05)

The genes in the upper-right and upper-left quadrants (high significance, large fold change) 
proceed to the next stage of analysis.
""")

# --- Section 2: Gene-to-Protein Mapping ---
st.header("2. Gene-to-Protein Mapping")
st.markdown("""
After identifying differentially expressed genes, we map them to their corresponding proteins using established biological databases.

**Inputs:**
- Differentially expressed genes (output from step 1)
- Gene-to-protein mapping data from biological databases

**Method:**
- Map Ensembl gene IDs to corresponding protein IDs using:
  - BioMart: Primary source for mapping Ensembl gene IDs to UniProt protein IDs
  - UniProt database: For additional protein annotations and mappings
  - Human Protein Atlas: For validation and additional protein data
- Merge differential expression results with the mapped protein information
- For each gene, retrieve the corresponding protein ID(s) and annotations

**Implementation notes:**
- We prioritize 1:1 gene-to-protein mappings for simplicity, although in reality, alternative splicing can result in multiple proteins from a single gene
- For the colorectal cancer dataset, we successfully mapped over 60,000 Ensembl gene IDs to their corresponding protein IDs
- These protein mappings serve as the foundation for subsequent immunogenicity scoring and cellular location determination
""")

# --- Section 3: Immunogenicity Score Calculation ---
st.header("3. Immunogenicity Score Calculation")
st.markdown("""
The immunogenicity score evaluates how suitable a protein is as a cancer vaccine target on a scale of 0 to 3, where higher scores indicate better vaccine targets.

**Source:**
- Pre-computed protein annotations file with immunogenicity scores (0-3 scale)

**Scoring factors included in annotations:**
1. **Cellular Location Base Score:**
   - Membrane proteins: 2 points
   - Secreted proteins: 2 points
   - Other locations: 0 points

2. **Additional Bonuses:**
   - High fold change: +1 point for genes with significant upregulation

**Formula in annotations:**

$$\\text{Immunogenicity Score} = \\text{Location Score} + \\text{Additional Bonuses}$$

The maximum score of 3 indicates a membrane or secreted protein with high fold change.

**Score Normalization:**
For ranking purposes, the raw immunogenicity score (0-3) is normalized to a 0-1 scale when calculating the final combined score:

$$\\text{Normalized Immunogenicity} = \\frac{\\text{Immunogenicity Score}}{3}$$
""")

# --- Visualization for Immunogenicity Scoring ---
st.markdown("### Immunogenicity Score Distribution")

# Create example data for immunogenicity scores with original 0-3 scale
locations = ['membrane', 'secreted', 'cytoplasm', 'nucleus', 'mitochondria', 'er', 'golgi', 'unknown']
weights = [0.35, 0.25, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.0125]
example_immuno = pd.DataFrame({
    'cellular_location': np.random.choice(locations, size=200, p=weights),
    'expression_percentile': np.random.uniform(0, 100, size=200),
})

# Calculate location base score (2 points for membrane/secreted)
location_scores = {
    'membrane': 2, 
    'secreted': 2, 
    'cytoplasm': 0, 
    'nucleus': 0, 
    'mitochondria': 0,
    'er': 0,
    'golgi': 0,
    'unknown': 0
}
example_immuno['location_score'] = example_immuno['cellular_location'].map(location_scores)

# Calculate fold change bonus (1 point for high fold change)
def get_expression_bonus(percentile):
    if percentile > 90:
        return 1.0  # High fold change (top 10%)
    elif percentile > 75:
        return 0.5  # Medium fold change
    elif percentile > 50:
        return 0.25  # Low fold change
    return 0.0

example_immuno['fold_change_bonus'] = example_immuno['expression_percentile'].apply(get_expression_bonus)

# Calculate total immunogenicity score (0-3 scale)
example_immuno['raw_score'] = example_immuno['location_score'] + example_immuno['fold_change_bonus']

# Create a long-format dataframe for comparing raw vs normalized scores
score_comparison = []
for _, row in example_immuno.iterrows():
    # Add raw score (0-3 scale)
    score_comparison.append({
        'Score Type': 'Raw Score (0-3)',
        'Score Value': row['raw_score'],
        'Cellular Location': row['cellular_location']
    })
    # Add normalized score (0-1 scale)
    score_comparison.append({
        'Score Type': 'Normalized Score (0-1)',
        'Score Value': row['raw_score'] / 3.0,
        'Cellular Location': row['cellular_location']
    })

score_df = pd.DataFrame(score_comparison)

# Create histogram showing both raw and normalized scores
fig = px.histogram(
    score_df, 
    x='Score Value',
    color='Score Type',
    facet_row='Score Type',
    barmode='overlay',
    title="Distribution of Immunogenicity Scores - Raw (0-3) vs Normalized (0-1)",
    labels={'Score Value': 'Immunogenicity Score', 'Cellular Location': 'Cellular Location'},
    color_discrete_map={
        'Raw Score (0-3)': '#ff7f0e',
        'Normalized Score (0-1)': '#1f77b4'
    },
    opacity=0.7
)

# Set x-axis range based on score type
fig.update_xaxes(range=[0, 3.2], row=1, col=1)  # For raw scores
fig.update_xaxes(range=[0, 1.1], row=2, col=1)  # For normalized scores

st.plotly_chart(fig, use_container_width=True)

# Add descriptive text
st.markdown("""
This histogram shows the distribution of immunogenicity scores from the protein annotations file:

- **Raw Scores (0-3 scale)**: Original scores from annotations with:
  - Base location score (0-2) - membrane/secreted proteins get 2 points
  - Fold change bonus (0-1) - high fold change gets 1 additional point
  
- **Normalized Scores (0-1 scale)**: Used in the final ranking calculation, obtained by dividing the raw score by 3.

The distribution shows that membrane and secreted proteins predominantly have higher scores, which is expected as they are the most accessible targets for vaccines.
""")

# Add Raw Immunogenicity Scores by Cellular Location
st.markdown("### Raw Immunogenicity Scores by Cellular Location")
st.markdown("""
This visualization shows how raw scores (0-3 scale) are distributed by cellular location.
""")

# Create histogram of raw scores by cellular location
fig = px.histogram(
    example_immuno, 
    x='raw_score',
    color='cellular_location',
    barmode='group',
    title="Distribution of Raw Immunogenicity Scores (0-3) by Cellular Location",
    labels={'raw_score': 'Raw Immunogenicity Score (0-3)', 'cellular_location': 'Cellular Location'},
    range_x=[0, 3.2],
)

st.plotly_chart(fig, use_container_width=True)

# Create a stacked bar chart for score components
st.markdown("### Score Component Breakdown")
st.markdown("""
This chart shows how different factors contribute to the raw immunogenicity score (0-3) for example proteins.
""")

# Select proteins with a range of scores
membrane_proteins = example_immuno[example_immuno['cellular_location'] == 'membrane'].copy()
secreted_proteins = example_immuno[example_immuno['cellular_location'] == 'secreted'].copy()
other_proteins = example_immuno[~example_immuno['cellular_location'].isin(['membrane', 'secreted'])].copy()

# Get samples from each
top_membrane = membrane_proteins.sort_values('raw_score', ascending=False).head(2)
top_secreted = secreted_proteins.sort_values('raw_score', ascending=False).head(1)
top_other = other_proteins.sort_values('fold_change_bonus', ascending=False).head(2)

# Combine 
top_examples = pd.concat([top_membrane, top_secreted, top_other]).reset_index(drop=True)
top_examples['protein_name'] = [f'Protein {i+1} ({row["cellular_location"]})' for i, (_, row) in enumerate(top_examples.iterrows())]

components_data = []
for _, row in top_examples.iterrows():
    # Base location score
    components_data.append({
        'Protein': row['protein_name'],
        'Component': 'Location Base Score',
        'Value': row['location_score']
    })
    
    # Expression bonus
    if row['fold_change_bonus'] > 0:
        components_data.append({
            'Protein': row['protein_name'],
            'Component': 'Fold Change Bonus',
            'Value': row['fold_change_bonus']
        })

components_df = pd.DataFrame(components_data)

# Create stacked bar chart
fig = px.bar(
    components_df,
    x='Protein',
    y='Value',
    color='Component',
    title='Immunogenicity Score Components (0-3 Scale)',
    labels={'Value': 'Score Component', 'Protein': 'Protein Name'},
    color_discrete_map={
        'Location Base Score': '#1f77b4',
        'Fold Change Bonus': '#2ca02c'
    }
)

# Update layout for stacked bar chart
fig.update_layout(
    xaxis_title="Protein",
    yaxis_title="Score Component",
    yaxis_range=[0, 3.2],
    legend_title="Component"
)

st.plotly_chart(fig, use_container_width=True)

# Add a table showing raw vs normalized scores
raw_vs_norm = pd.DataFrame({
    'Protein': top_examples['protein_name'],
    'Raw Score (0-3)': top_examples['raw_score'],
    'Normalized Score (0-1)': top_examples['raw_score']/3.0,
    'Cellular Location': top_examples['cellular_location']
})

st.write("**Raw vs. Normalized Scores:**")
st.dataframe(raw_vs_norm.round(2))

st.markdown("""
**Explanation of Immunogenicity Score Components:**
- **Location Base Score (0-2)**: Membrane and secreted proteins receive 2 points
- **Fold Change Bonus (0-1)**: Additional points for high fold change

The raw scores range from 0-3, but in the final ranking calculation, they are normalized to a 0-1 scale.
""")

# --- Section 4: Combined Score Calculation ---
st.header("4. Combined Score Calculation")
st.markdown("""
The final step combines multiple factors into a single score for ranking protein targets.

**Inputs:**
- Differentially expressed genes with fold change and p-values
- Immunogenicity scores (0-3 scale)

**Method:**
1. **Normalize the immunogenicity score**:
   - The raw immunogenicity score (0-3 scale) is normalized to a 0-1 scale:
   
   $$\\text{Normalized Immunogenicity} = \\frac{\\text{Raw Immunogenicity Score}}{3}$$

2. **Calculate weighted sum**:
   - The combined score uses a weighted sum of three main factors:

   $$\\text{Combined Score} = 0.4 \\cdot \\text{Fold Change} + 0.3 \\cdot \\text{Statistical Significance} + 0.3 \\cdot \\text{Normalized Immunogenicity}$$

   Where:
   - Fold change is the raw fold change value (not log-transformed)
   - Statistical significance is calculated as (1 - p-value)
   - Normalized immunogenicity is the 0-1 scale score derived from the raw 0-3 scale

The proteins are then ranked in descending order of their combined score, with the highest scoring proteins representing the most promising vaccine targets.
""")

# --- Visualization for Combined Score ---
st.markdown("### Combined Score Components")

# Create a radar chart to show the weighting of components
categories = ['Fold Change (40%)', 'Statistical\nSignificance (30%)', 'Immunogenicity\nScore (30%)']
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=[0.4, 0.3, 0.3],
    theta=categories,
    fill='toself',
    name='Component Weights'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 0.5]
        )),
    showlegend=False,
    title="Combined Score Component Weights"
)

st.plotly_chart(fig, use_container_width=True)

# Create example data for combined score with more realistic values
example_combined = pd.DataFrame({
    'gene_id': [f'GENE_{i}' for i in range(10)],
    'protein_name': [f'Protein {i}' for i in range(10)],
    'fold_change': np.random.uniform(2, 8, 10),
    'p_value': np.random.uniform(0.001, 0.05, 10),
    'raw_immunogenicity_score': np.random.choice([0, 1, 2, 3], 10)  # 0-3 scale scores
})

# Add cellular location to better demonstrate the example
locations = []
for score in example_combined['raw_immunogenicity_score']:
    if score >= 3:
        locations.append('membrane')  # Score 3 = membrane + high fold change
    elif score == 2:
        locations.append('secreted')  # Score 2 = secreted (or membrane without fold change bonus)
    elif score > 0:
        locations.append('cytoplasm')  # Scores 1 = cytoplasm with fold change bonus
    else:
        locations.append('unknown')   # Score 0 = unknown/other

example_combined['cellular_location'] = locations

# Calculate combined score components
example_combined['statistical_significance'] = 1 - example_combined['p_value']
example_combined['normalized_immunogenicity'] = example_combined['raw_immunogenicity_score'] / 3.0

# Calculate combined score
example_combined['combined_score'] = (
    example_combined['fold_change'] * 0.4 +
    example_combined['statistical_significance'] * 0.3 +
    example_combined['normalized_immunogenicity'] * 0.3
)

# Sort by combined score
example_combined = example_combined.sort_values('combined_score', ascending=False).reset_index(drop=True)
example_combined['rank'] = example_combined.index + 1

# Display example table
st.markdown("### Example Combined Score Calculation")
display_cols = ['rank', 'gene_id', 'protein_name', 'cellular_location', 'fold_change', 'p_value', 
                'raw_immunogenicity_score', 'normalized_immunogenicity', 'combined_score']
st.dataframe(example_combined[display_cols].round(3))

# --- Limitations and Considerations ---
st.header("Limitations and Considerations")
st.markdown("""
While this methodology provides a systematic approach to prioritizing potential vaccine targets, 
there are important limitations to consider:

1. **Statistical Simplification**:
   - We use Welch's t-test, which assumes normal distribution of data
   - More sophisticated methods like DESeq2 or edgeR would be more appropriate for RNA-seq data
   - No multiple testing correction is applied to p-values

2. **Protein Annotation Sources**:
   - For the colorectal cancer dataset, protein cellular locations were obtained from:
     - **BioMart:** Primary source for mapping Ensembl gene IDs to GO cellular components (36.3% coverage)
     - **UniProt:** Used for additional protein annotations and subcellular locations
     - **Human Protein Atlas:** Used for experimental validation of protein localization
   - While we achieved 36.3% real data coverage, the remaining proteins required assignment based on homology or prediction

3. **Immunogenicity Assessment**:
   - Our scoring is simplified and focuses mainly on cellular location
   - Real immunogenicity assessment would include:
     - Epitope prediction
     - MHC binding affinity
     - Sequence homology with self-proteins
     - Post-translational modifications
     - Existing immune tolerance

4. **Additional Factors Not Considered**:
   - Expression in non-target tissues (off-target effects)
   - Role in cancer progression/survival (oncogenic drivers vs. passengers)
   - Genetic variation across populations
   - Existing antibody titers in the population

For real vaccine target selection, these additional factors should be considered, along with
experimental validation of the prioritized targets.
""")

# --- Data Sources and Attribution ---
st.header("Data Sources and Attribution")
st.markdown("""
The protein location and annotation data used in this tool were sourced from multiple public databases:

1. **Gene Ontology Cellular Component Data:**
   - Accessed via BioMart through the Ensembl API
   - Used for primary cellular location assignment (36.3% coverage)
   - Source: [Gene Ontology Consortium](http://geneontology.org/)

2. **UniProt Protein Database:**
   - Used for protein annotations, functions, and subcellular localization
   - Batch queries implemented for retrieving protein information
   - Source: [UniProt](https://www.uniprot.org/)

3. **Human Protein Atlas:**
   - Used for experimental validation of protein localization
   - Provided tissue-specific expression patterns
   - Source: [Human Protein Atlas](https://www.proteinatlas.org/)

4. **Colorectal Cancer Gene Data:**
   - Known colorectal cancer genes and pathway members were identified from:
     - [Cancer Gene Census](https://cancer.sanger.ac.uk/census)
     - [KEGG Pathway Database](https://www.genome.jp/kegg/pathway.html)
     - [cBioPortal](https://www.cbioportal.org/)

We significantly improved data quality by replacing synthetic location data with real biological data, 
increasing real data coverage from <1% to 36.3% across the colorectal dataset.
""")

# --- References ---
st.header("References and Further Reading")
st.markdown("""
1. **Differential Expression Analysis**:
   - Love MI, Huber W, Anders S. (2014). "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2." *Genome Biology*, 15:550. DOI: 10.1186/s13059-014-0550-8.
   - Ritchie ME, et al. (2015). "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research*, 43(7):e47. DOI: 10.1093/nar/gkv007.

2. **Cancer Vaccine Target Identification**:
   - Finn OJ. (2017). "Human Tumor Antigens Yesterday, Today, and Tomorrow." *Cancer Immunology Research*, 5(5):347-354. DOI: 10.1158/2326-6066.CIR-17-0112.
   - Smith CC, et al. (2019). "Alternative tumour-specific antigens." *Nature Reviews Cancer*, 19:465-478. DOI: 10.1038/s41568-019-0162-4.

3. **Protein Cellular Localization and Vaccine Development**:
   - Schiller JT, Lowy DR. (2018). "Prophylactic human papillomavirus vaccines." *Journal of Clinical Investigation*, 128(12):4799-4808. DOI: 10.1172/JCI122481.
   - Cheever MA, et al. (2009). "The prioritization of cancer antigens: a National Cancer Institute pilot project for the acceleration of translational research." *Clinical Cancer Research*, 15(17):5323-5337. DOI: 10.1158/1078-0432.CCR-09-0737.
""") 

# --- Future Roadmap - AI Integration ---
st.header("Future Roadmap - AI Integration")
st.markdown("""
## Foundation Model Integration

In future iterations of this application, we plan to integrate advanced foundation AI models like T-X Gemma to enhance multiple aspects of the cancer vaccine target identification pipeline. These improvements will create a more comprehensive and powerful tool for researchers.

### Planned AI-Driven Enhancements:

1. **Enhanced AI Expert Review**
   - Replace the current OpenAI-based review system with locally deployed foundation models like T-X Gemma
   - Develop specialized prompts tailored to cancer immunology and vaccine development
   - Enable more detailed analysis of candidate targets with biological context and mechanistic insights
   - Provide target-specific literature summaries to support decision-making

2. **Advanced Protein Annotation**
   - Generate higher-quality protein descriptions and functional annotations by leveraging biomedical knowledge in foundation models
   - Predict protein domains and motifs relevant to immunogenicity
   - Assess post-translational modifications that might affect vaccine efficacy
   - Evaluate protein conservation across populations to ensure broad vaccine coverage

3. **Automated Literature Integration**
   - Implement real-time scientific literature search and summarization for top candidate proteins
   - Extract cancer type-specific information about expression patterns and clinical relevance
   - Identify previous experimental validation of targets from published studies
   - Highlight contradictory findings or potential safety concerns

4. **Sequence-Based Analysis**
   - Add epitope prediction capabilities using foundation models trained on immune receptor-antigen interactions
   - Assess potential cross-reactivity with human proteins to minimize autoimmune risks
   - Evaluate mutational burden and neoantigen potential in different cancer types
   - Predict MHC presentation likelihood for candidate epitopes

5. **Interactive Explanation Interface**
   - Create a conversational interface where users can ask specific questions about results
   - Enable natural language queries like "Why is this protein ranked highly?" or "What's known about this target in colorectal cancer?"
   - Provide customized explanations of the ranking algorithm based on user expertise level
   - Allow users to explore "what-if" scenarios by adjusting parameters through conversation

### Technical Implementation Considerations:

- Foundation models can be deployed locally using frameworks like Hugging Face Transformers
- Quantized versions of models like T-X Gemma can run efficiently on standard research hardware
- Domain-specific fine-tuning on biomedical literature would enhance performance for this application
- A modular architecture would allow swapping different models as the technology evolves

These AI-driven enhancements would transform the application from a data analysis tool into an intelligent research assistant, accelerating the identification and validation of promising cancer vaccine targets.
""") 