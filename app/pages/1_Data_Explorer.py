import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
from datetime import datetime

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
    
    /* Improved button styling */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.3) !important;
    }
    
    /* Improving form elements */
    div[data-baseweb="input"] {
        border-radius: 8px !important;
    }
    
    /* Info and warning boxes */
    div[data-testid="stInfo"] {
        background-color: #EDE9FE !important;
        border-left-color: #6D28D9 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        padding-left: 1rem;
    }
    
    /* Better spacing for metric elements */
    div[data-testid="stMetric"] {
        background-color: #EDE9FE;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Data tables styling */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Better tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #EDE9FE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6D28D9 !important;
        color: white !important;
    }

    /* Info cards */
    .info-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Record the date of access if not already set
if 'download_date' not in st.session_state:
    st.session_state.download_date = datetime.now().strftime("%B %d, %Y")

# --- Add proper path handling (assuming this is in an 'app/pages' directory) ---
try:
    # Standard execution from 'app/pages' directory
    PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = os.path.dirname(PAGES_DIR)
    ROOT_DIR = os.path.dirname(APP_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    # Check if data directory exists, otherwise try relative path for Streamlit Cloud
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "data" # Fallback for Streamlit Cloud or running from root
    # Define colorectal data directory
    COLORECTAL_DATA_DIR = os.path.join(DATA_DIR, "collarectal")
    # Define example data directory
    EXAMPLE_DATA_DIR = os.path.join(DATA_DIR, "example")
except NameError:
    # Handling cases where __file__ is not defined (e.g., notebooks, direct execution)
    APP_DIR = os.getcwd()
    # Try to find the 'data' directory relative to current working directory
    if os.path.exists("data"):
        DATA_DIR = "data"
    else:
        DATA_DIR = os.path.join(os.path.dirname(APP_DIR), "data")
    # Define colorectal data directory
    COLORECTAL_DATA_DIR = os.path.join(DATA_DIR, "collarectal")
    # Define example data directory
    EXAMPLE_DATA_DIR = os.path.join(DATA_DIR, "example")

# --- Page configuration ---
st.set_page_config(
    page_title="Data Explorer - Cancer Vaccine Development Tool",
    layout="wide"
)

# Apply modern styling
apply_modern_styles()

# --- Title and introduction ---
st.title("Data Explorer")

st.markdown("""
<div style="background-color: #EDE9FE; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
    <p>
    This page provides an in-depth explanation of the datasets used in the Cancer Vaccine Development Tool.
    Understanding these datasets is crucial for interpreting results and making informed decisions in cancer vaccine development.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar for dataset selection ---
with st.sidebar:
    st.markdown("## Dataset Selection")
    
    # Dataset options
    dataset_option = st.radio(
        "Select dataset to explore:",
        ["Example Dataset", "Real-World Dataset (Colorectal Cancer)"]
    )

def load_data(dataset_path):
    """Load all required data files from the specified path"""
    try:
        # Define file paths
        expression_file = os.path.join(dataset_path, "expression.csv")
        metadata_file = os.path.join(dataset_path, "metadata.csv")
        gene_protein_file = os.path.join(dataset_path, "gene_protein_map.csv")
        annotations_file = os.path.join(dataset_path, "protein_annotations.csv")
        
        # Special case for colorectal dataset
        if "collarectal" in dataset_path:
            expression_file = os.path.join(dataset_path, "GSE251845_htseq_raw_counts.csv")
        
        # Load data if all files exist
        if os.path.exists(expression_file) and os.path.exists(metadata_file) and os.path.exists(gene_protein_file) and os.path.exists(annotations_file):
            expression_data = pd.read_csv(expression_file)
            
            # Special handling for colorectal dataset
            if "collarectal" in dataset_path:
                # For colorectal data, we need to extract gene_ids from index and create a proper column
                gene_ids = expression_data.index.astype(str).tolist() if expression_data.index.name == 'gene_id' else expression_data.index.astype(str).tolist()
                temp_expr = pd.DataFrame({'gene_id': gene_ids})
                for col in expression_data.columns:
                    clean_col = col.replace('_htseq.out', '')
                    temp_expr[clean_col] = expression_data[col].values
                expression_data = temp_expr
            
            gene_id_column = 'gene_id'
            if gene_id_column in expression_data.columns:
                expression_data = expression_data.set_index(gene_id_column)
            
            metadata = pd.read_csv(metadata_file)
            gene_protein_map = pd.read_csv(gene_protein_file)
            protein_annotations = pd.read_csv(annotations_file)
            
            return expression_data, metadata, gene_protein_map, protein_annotations, True
        else:
            missing_files = []
            if not os.path.exists(expression_file): 
                if "collarectal" in dataset_path:
                    missing_files.append("GSE251845_htseq_raw_counts.csv")
                else:
                    missing_files.append("expression.csv")
            if not os.path.exists(metadata_file): missing_files.append("metadata.csv")
            if not os.path.exists(gene_protein_file): missing_files.append("gene_protein_map.csv")
            if not os.path.exists(annotations_file): missing_files.append("protein_annotations.csv")
            
            st.error(f"Missing data files: {', '.join(missing_files)}")
            return None, None, None, None, False
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, False

# Load data based on selection
if dataset_option == "Example Dataset":
    expression_data, metadata, gene_protein_map, protein_annotations, data_loaded = load_data(EXAMPLE_DATA_DIR)
    dataset_description = """
    This example dataset contains gene expression data from colorectal cancer patient samples, 
    along with protein annotations that are essential for cancer vaccine target identification.
    """
else: # Real-World Dataset
    expression_data, metadata, gene_protein_map, protein_annotations, data_loaded = load_data(COLORECTAL_DATA_DIR)
    dataset_description = """
    This is a real-world dataset from colorectal cancer patients, containing RNA-seq expression data from 
    matched tumor and normal samples. The dataset includes protein annotations critical for vaccine target selection.
    """

# Create tabs for different data views
if data_loaded:
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Data Overview", "🧬 Gene Expression", "🔬 Protein Annotations", "📊 Data Visualization"])
    
    with tab1:
        # Dataset Introduction and Data Flow Explanation
        st.markdown("## Understanding the Data")
        
        # Display dataset information - replace HTML with simple markdown
        st.info(dataset_description)
        
        # Explain the data flow and relationships - replace HTML with simple markdown
        st.markdown("### How the Data Works Together")
        st.markdown("The Cancer Vaccine Development Tool uses multiple interconnected datasets to identify potential vaccine targets:")
        
        st.markdown("1. **Gene Expression Data**: Contains gene expression levels across multiple samples, comparing tumor vs normal tissue")
        st.markdown("2. **Sample Metadata**: Provides information about each sample, including tumor/normal status")
        st.markdown("3. **Gene-Protein Mapping**: Links gene IDs to their corresponding protein IDs")
        st.markdown("4. **Protein Annotations**: Contains critical information about proteins, including:")
        st.markdown("   - Cellular location (membrane, secreted, etc.) - essential for target accessibility")
        st.markdown("   - Immunogenicity scores - indicating potential to trigger immune responses")
        st.markdown("   - Base scores - underlying measurements for target evaluation")
        
        st.markdown("This integrated approach allows identification of overexpressed genes in cancer tissues that also encode proteins with favorable characteristics for vaccine targeting.")
        
        # Add Data Files Structure section using simple markdown
        st.markdown("## Data Files Structure")
        st.markdown("Understanding the structure of each data file is crucial for interpreting the results.")
        
        # Expression Data
        st.markdown("### Expression Data")
        st.markdown("- **Format:** Rows are genes (identified by Ensembl gene IDs), columns are samples")
        st.markdown("- **Values:** RNA counts normalized to allow comparison across samples")
        st.markdown("- **Size:** Typically contains 20,000+ genes and multiple samples (e.g., paired tumor/normal)")
        st.markdown("- **Data Type:** Quantitative (continuous values representing expression levels)")
        
        # Sample Metadata
        st.markdown("### Sample Metadata")
        st.markdown("- **Format:** Each row corresponds to one sample in the expression data")
        st.markdown("- **Key Columns:**")
        st.markdown("  - sample_id: Unique identifier that matches column names in expression data")
        st.markdown("  - condition: Identifies the sample type (e.g., \"tumor\", \"normal\")")
        st.markdown("  - patient_id: Links samples from the same patient (for paired analyses)")
        st.markdown("- **Purpose:** Allows proper grouping of samples for differential expression analysis")
        
        # Gene-Protein Map
        st.markdown("### Gene-Protein Map")
        st.markdown("- **Format:** Two-column mapping table")
        st.markdown("- **Key Columns:**")
        st.markdown("  - gene_id: Ensembl gene identifier (ENSG...)")
        st.markdown("  - protein_id: Corresponding protein identifier (ENSP...)")
        st.markdown("- **Relationship:** Many genes have a one-to-one relationship with proteins, though some genes can encode multiple proteins due to alternative splicing")
        st.markdown("- **Source:** Derived from Ensembl BioMart database")
        
        # Protein Annotations
        st.markdown("### Protein Annotations")
        st.markdown("- **Format:** Each row represents one protein with multiple annotation fields")
        st.markdown("- **Key Columns:**")
        st.markdown("  - protein_id: Unique identifier matching the gene-protein map")
        st.markdown("  - protein_name: Human-readable name")
        st.markdown("  - cellular_location: Where the protein is found in the cell")
        st.markdown("  - base_score: Score derived from cellular location (0-1 scale)")
        st.markdown("  - immunogenicity_score: Combined score indicating vaccine target potential (0-1 scale)")
        st.markdown("- **Sources:** UniProt, Human Protein Atlas, and BioMart with extensive data validation")
        
        # Basic statistics in a dashboard-like layout
        st.markdown("## Dataset Statistics")
        
        # Add custom CSS to make metrics smaller
        st.markdown("""
        <style>
        div[data-testid="stMetric"] {
            background-color: #EDE9FE;
            padding: 0.5rem;
            border-radius: 8px;
        }
        div[data-testid="stMetric"] > div:first-child {
            font-size: 0.9rem;
        }
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.3rem;
        }
        div[data-testid="stMetric"] > div:nth-child(3) {
            font-size: 0.8rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Genes", f"{expression_data.shape[0]:,}")
        with col2:
            st.metric("Proteins", f"{protein_annotations.shape[0]:,}")
        with col3:
            if 'condition' in metadata.columns:
                condition_counts = metadata['condition'].value_counts()
                # Format the condition string to fit better
                condition_list = [f"{k}: {v}" for k, v in condition_counts.items()]
                st.metric("Samples", ", ".join(condition_list[:1]) + "...", 
                         help=", ".join(condition_list))
        
    with tab2:
        # Gene Expression Data Explanation
        st.markdown("## Gene Expression Data")
        
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">What is Gene Expression Data?</h3>
            <p>Gene expression data quantifies the activity level of genes across different samples. In this dataset:</p>
            <ul>
                <li>Each <strong>row</strong> represents a different gene (identified by Ensembl gene ID)</li>
                <li>Each <strong>column</strong> represents a different sample</li>
                <li>The <strong>values</strong> represent expression levels (RNA counts) - higher values indicate more gene activity</li>
            </ul>
            <p>For cancer vaccine development, we look for genes that are <strong>significantly overexpressed</strong> in tumor samples compared to normal tissue, as these may encode proteins that could serve as vaccine targets.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gene Expression Data Explanation - simple version
        st.info("""
        Gene expression data quantifies the activity level of genes across different samples. In this dataset:
        
        • Each **row** represents a different gene (identified by Ensembl gene ID)
        • Each **column** represents a different sample
        • The **values** represent expression levels (RNA counts) - higher values indicate more gene activity
        
        For cancer vaccine development, we look for genes that are **significantly overexpressed** in tumor samples compared to normal tissue, as these may encode proteins that could serve as vaccine targets.
        """)
        
        # Display expression data preview
        st.markdown("### Expression Data Preview")
        st.dataframe(expression_data.head(10), use_container_width=True)
        
        # Display metadata
        st.markdown("### Sample Metadata")
        st.markdown("""
        <p>The metadata provides crucial context about each sample, including whether it's from tumor or normal tissue:</p>
        """, unsafe_allow_html=True)
        st.dataframe(metadata, use_container_width=True)
        
        # Display gene-protein mapping
        st.markdown("### Gene-Protein Mapping")
        st.markdown("""
        <p>This mapping connects gene IDs to their corresponding protein IDs, enabling integration of expression data with protein annotations:</p>
        """, unsafe_allow_html=True)
        st.dataframe(gene_protein_map.head(10), use_container_width=True)
        
    with tab3:
        # Protein Annotations Explanation
        st.markdown("## Protein Annotations")
        
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">Understanding Protein Annotations</h3>
            <p>Protein annotations contain critical information for vaccine target selection:</p>
            <ul>
                <li><strong>Cellular Location</strong>: Where the protein is found in the cell - proteins on the cell membrane or that are secreted are more accessible to the immune system and therefore better vaccine targets</li>
                <li><strong>Immunogenicity Score</strong>: A measure (0-1) of how likely the protein is to trigger an immune response - higher scores indicate better vaccine candidates</li>
                <li><strong>Base Score</strong>: An underlying evaluation metric that contributes to overall target assessment</li>
            </ul>
            <p>Ideal cancer vaccine targets are proteins that are:</p>
            <ol>
                <li>Encoded by highly expressed genes in tumor tissue</li>
                <li>Located on the cell membrane or secreted (accessible)</li>
                <li>Have high immunogenicity scores (likely to trigger immune responses)</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Display protein annotations
        st.markdown("### Protein Annotations Data")
        st.dataframe(protein_annotations, use_container_width=True)
        
        # Cellular location distribution
        st.markdown("### Cellular Location Distribution")
        
        # Create a visualization for cellular locations
        if 'cellular_location' in protein_annotations.columns:
            location_counts = protein_annotations['cellular_location'].value_counts().reset_index()
            location_counts.columns = ['Location', 'Count']
            
            # Create pie chart
            fig = px.pie(
                location_counts,
                values='Count',
                names='Location',
                title='Distribution of Proteins by Cellular Location',
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig.update_layout(
                legend_title="Cellular Location",
                font=dict(family="Arial, sans-serif", size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation of why this matters
            st.markdown("""
            <div style="background-color: #EDE9FE; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <h4 style="margin-top: 0;">Why Cellular Location Matters</h4>
                <p>
                The cellular location of a protein is crucial for vaccine target selection:
                </p>
                <ul>
                    <li><strong>Membrane proteins</strong> are directly accessible to antibodies and immune cells, making them ideal vaccine targets</li>
                    <li><strong>Secreted proteins</strong> are released from cells and can be easily recognized by the immune system</li>
                    <li><strong>Cytoplasmic and nuclear proteins</strong> are less accessible, making them less suitable vaccine targets</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Immunogenicity score distribution
        st.markdown("### Immunogenicity Score Distribution")
        
        if 'immunogenicity_score' in protein_annotations.columns:
            # Create histogram for immunogenicity scores
            fig = px.histogram(
                protein_annotations,
                x='immunogenicity_score',
                nbins=20,
                title='Distribution of Immunogenicity Scores',
                template='plotly_white',
                color_discrete_sequence=['#6D28D9']
            )
            
            fig.update_layout(
                xaxis_title="Immunogenicity Score",
                yaxis_title="Count",
                font=dict(family="Arial, sans-serif", size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explain immunogenicity scores
            st.markdown("""
            <div style="background-color: #EDE9FE; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <h4 style="margin-top: 0;">Understanding Immunogenicity Scores</h4>
                <p>
                Immunogenicity scores predict how likely a protein is to trigger an immune response:
                </p>
                <ul>
                    <li><strong>Higher scores (0.8-1.0)</strong>: Proteins with excellent potential to be recognized by the immune system</li>
                    <li><strong>Medium scores (0.5-0.8)</strong>: Proteins with moderate immunogenic potential</li>
                    <li><strong>Lower scores (below 0.5)</strong>: Proteins less likely to trigger significant immune responses</li>
                </ul>
                <p>For cancer vaccine development, proteins with scores above 0.8 that are also overexpressed in cancer tissue make promising candidates.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add detailed explanation of the base and immunogenicity score calculation
            st.markdown("### Base and Immunogenicity Score Calculation")
            
            # Base Score explanation using simple Streamlit components
            st.subheader("How Base Scores Are Calculated")
            
            st.info("""
            The base score is the foundation of our vaccine target evaluation, primarily determined by the protein's cellular location:
            
            **Formula:** Base Score = Cellular Location Score
            """)
            
            st.markdown("#### Cellular Location Scores:")
            st.markdown("""
            * **Membrane:** 0.85
            * **Secreted:** 0.70
            * **Cytoplasm:** 0.60
            * **Nucleus:** 0.50
            * **Mitochondria:** 0.30
            * **Other locations:** 0.00-0.10
            """)
            
            st.markdown("**Score Normalization:** All final scores are capped at 1.0 to maintain a consistent 0-1 scale.")
            
            # Example calculation in a success message block
            st.success("""
            **Example Calculation:**
            
            For a membrane protein (cellular location: 0.85):
            
            Base Score = 0.85
            """)

            # Immunogenicity Score explanation using simple Streamlit components
            st.subheader("How Immunogenicity Scores Are Calculated")
            
            st.info("""
            The immunogenicity score builds upon the base score by incorporating additional factors 
            that contribute to a protein's potential as a vaccine target.
            
            **Formula:** Immunogenicity Score = Base Score + Expression Bonuses + Cancer-Specific Bonuses
            """)
            
            st.markdown("#### Expression-Based Bonuses:")
            st.markdown("""
            * **High expression bonus:**
                * Top 10% expressed genes: +0.15
                * Top 25% expressed genes: +0.10
                * Top 50% expressed genes: +0.05
            * **Expression stability bonus:**
                * Highly stable expression (low variation across samples): +0.05
                * Moderately stable expression: +0.03
            """)
            
            st.markdown("#### Cancer-Specific Bonuses:")
            st.markdown("""
            * **Known cancer genes:** +0.20 for genes/proteins with established roles in cancer
            * **Relevant gene families:** +0.10 for members of cancer-associated pathways
            """)
            
            st.markdown("**Score Normalization:** All final scores are capped at 1.0 to maintain a consistent 0-1 scale.")
            
            # Example calculation in a success message block
            st.success("""
            **Example Calculation:**
            
            For a membrane protein (base score: 0.85) encoded by a gene in the top 10% of expression (bonus: +0.15):
            
            Immunogenicity Score = 0.85 + 0.15 = 1.0
            
            This high score (1.0) indicates an excellent potential vaccine target that combines ideal 
            cellular location with strong expression in the target tissue.
            """)

    with tab4:
        st.markdown("## Data Integration Visualization")
        
        # Create a scatterplot showing the relationship between cellular location and immunogenicity
        if 'cellular_location' in protein_annotations.columns and 'immunogenicity_score' in protein_annotations.columns:
            # Filter out rows with missing values
            valid_data = protein_annotations.dropna(subset=['cellular_location', 'immunogenicity_score'])
            valid_data = valid_data[valid_data['cellular_location'] != '']
            
            # Create a scatterplot
            fig = px.scatter(
                valid_data,
                x='immunogenicity_score',
                y='base_score',
                color='cellular_location',
                hover_name='protein_name',
                template='plotly_white',
                title='Protein Characteristics by Cellular Location',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig.update_layout(
                height=600,
                xaxis_title="Immunogenicity Score",
                yaxis_title="Base Score",
                legend_title="Cellular Location",
                font=dict(family="Arial, sans-serif", size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            <div style="background-color: #EDE9FE; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <h4 style="margin-top: 0;">Interpreting This Visualization</h4>
                <p>
                This plot shows the relationship between immunogenicity scores and base scores, colored by cellular location:
                </p>
                <ul>
                    <li>Points in the upper-right represent proteins with both high immunogenicity and high base scores</li>
                    <li>Membrane and secreted proteins (particularly in the upper-right) make ideal vaccine targets</li>
                    <li>The distribution of points reveals patterns in how different cellular locations correlate with immunogenic potential</li>
                </ul>
                <p>For cancer vaccine development, we prioritize membrane and secreted proteins in the upper-right quadrant that also show overexpression in tumor tissues.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Expression and protein annotations integration
        st.markdown("### Gene Expression and Protein Properties")
        
        # Check if we have both expression data and gene-protein mapping
        if not expression_data.empty and not gene_protein_map.empty and not protein_annotations.empty:
            # Select sample columns for comparison
            # Look for samples with 'C' (cancer) and 'N' (normal) suffix
            cancer_samples = [col for col in expression_data.columns if col.endswith('C')]
            normal_samples = [col for col in expression_data.columns if col.endswith('N')]
            
            if cancer_samples and normal_samples:
                # Calculate mean expression for cancer and normal samples
                expression_data['cancer_mean'] = expression_data[cancer_samples].mean(axis=1)
                expression_data['normal_mean'] = expression_data[normal_samples].mean(axis=1)
                expression_data['log2_fc'] = np.log2((expression_data['cancer_mean'] + 1) / (expression_data['normal_mean'] + 1))
                
                # Reset index to get gene_id as a column
                expr_with_id = expression_data.reset_index()
                
                # Merge with gene-protein mapping
                merged_data = expr_with_id.merge(gene_protein_map, on='gene_id', how='inner')
                
                # Merge with protein annotations
                final_data = merged_data.merge(protein_annotations, on='protein_id', how='inner')
                
                # Create visualization of log2 fold change by cellular location
                if 'cellular_location' in final_data.columns and 'log2_fc' in final_data.columns:
                    # Filter out rows with missing values or empty cellular location
                    valid_data = final_data.dropna(subset=['cellular_location', 'log2_fc'])
                    valid_data = valid_data[valid_data['cellular_location'] != '']
                    
                    # Create boxplot
                    fig = px.box(
                        valid_data,
                        x='cellular_location',
                        y='log2_fc',
                        color='cellular_location',
                        template='plotly_white',
                        title='Gene Expression Changes by Protein Cellular Location',
                        points='all',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    fig.update_layout(
                        height=600,
                        xaxis_title="Cellular Location",
                        yaxis_title="Log2 Fold Change (Cancer/Normal)",
                        legend_title="Cellular Location",
                        font=dict(family="Arial, sans-serif", size=12),
                        showlegend=False
                    )
                    
                    # Add a horizontal line at y=0
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    <div style="background-color: #EDE9FE; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                        <h4 style="margin-top: 0;">What This Visualization Shows</h4>
                        <p>
                        This boxplot demonstrates the relationship between protein cellular location and gene expression changes in cancer:
                        </p>
                        <ul>
                        <li>Points above the dashed line (log2_fc > 0) represent genes overexpressed in cancer samples</li>
                        <li>The distribution for each cellular location shows which protein types tend to be up or down-regulated in cancer</li>
                        </ul>
                        <p>For cancer vaccine development, we particularly focus on membrane and secreted proteins that are significantly overexpressed (high positive log2 fold change values).</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a visualization highlighting top vaccine candidates
                    st.markdown("### Potential Vaccine Target Candidates")
                    
                    # Define criteria for good candidates
                    candidate_data = valid_data.copy()
                    # Consider membrane or secreted proteins with high immunogenicity and overexpression
                    candidate_data['is_candidate'] = (
                        ((candidate_data['cellular_location'] == 'membrane') | 
                         (candidate_data['cellular_location'] == 'secreted')) &
                        (candidate_data['immunogenicity_score'] >= 0.8) &
                        (candidate_data['log2_fc'] >= 1)
                    )
                    
                    # Create scatter plot highlighting candidates
                    fig = px.scatter(
                        candidate_data,
                        x='log2_fc',
                        y='immunogenicity_score',
                        color='is_candidate',
                        hover_name='protein_name',
                        hover_data=['cellular_location', 'protein_id', 'gene_id'],
                        template='plotly_white',
                        title='Potential Vaccine Targets',
                        color_discrete_map={True: '#6D28D9', False: '#d1d5db'}
                    )
                    
                    fig.update_layout(
                        height=600,
                        xaxis_title="Log2 Fold Change (Cancer/Normal)",
                        yaxis_title="Immunogenicity Score",
                        legend_title="Candidate Status",
                        font=dict(family="Arial, sans-serif", size=12)
                    )
                    
                    # Add vertical and horizontal guidelines
                    fig.add_vline(x=1, line_dash="dash", line_color="gray")
                    fig.add_hline(y=0.8, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top candidates in a table
                    st.markdown("### Top Vaccine Target Candidates")
                    top_candidates = candidate_data[candidate_data['is_candidate']].sort_values(
                        by=['immunogenicity_score', 'log2_fc'], 
                        ascending=False
                    )[['protein_id', 'protein_name', 'cellular_location', 'immunogenicity_score', 'log2_fc']].head(10)
                    
                    if not top_candidates.empty:
                        st.dataframe(top_candidates, use_container_width=True)
                    else:
                        st.info("No candidates meeting the specified criteria were found in this dataset.")
                    
                    # Final explanation
                    st.markdown("""
                    <div style="background-color: #EDE9FE; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                        <h4 style="margin-top: 0;">Interpreting Vaccine Target Candidates</h4>
                        <p>
                        The highlighted points represent promising vaccine target candidates that meet three key criteria:
                        </p>
                        <ol>
                            <li><strong>Accessibility:</strong> Located on the membrane or secreted (accessible to the immune system)</li>
                            <li><strong>Immunogenicity:</strong> High immunogenicity score (≥ 0.8), suggesting strong potential to trigger immune responses</li>
                            <li><strong>Cancer-specific expression:</strong> Significantly overexpressed in cancer (log2 fold change ≥ 1, meaning at least 2x higher expression)</li>
                        </ol>
                        <p>These proteins represent promising starting points for cancer vaccine development, as they combine favorable characteristics for both targeting and immune recognition.</p>
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.error("Failed to load dataset. Please check data paths and file structure.")

# Add data access information at the bottom of the page
st.markdown("""
<div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e6e6e6; opacity: 0.7; font-size: 0.8rem;">
    Data accessed on: {0}
</div>
""".format(st.session_state.download_date), unsafe_allow_html=True) 