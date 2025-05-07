import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
from datetime import datetime

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
    page_title="Data Explorer - Early-Stage Antigen Prioritizer",
    layout="wide"
)

# --- Title and introduction ---
st.title("Data Explorer")
st.markdown("""
This page allows you to explore and visualize the datasets used in the Early-Stage Antigen Prioritizer Tool.
You can examine gene expression patterns, compare conditions, and understand the data structure.
""")

# --- Sidebar for dataset selection ---
st.sidebar.header("Dataset Selection")

# Dataset options
dataset_option = st.sidebar.radio(
    "Select dataset to explore:",
    ["Example Dataset", "Real-World Dataset (Colorectal Cancer)"]
)

def load_example_data():
    """Load the example dataset with synthetic data"""
    try:
        expression_file = os.path.join(EXAMPLE_DATA_DIR, "expression.csv")
        metadata_file = os.path.join(EXAMPLE_DATA_DIR, "metadata.csv")
        
        # Load data
        if os.path.exists(expression_file) and os.path.exists(metadata_file):
            expression_data = pd.read_csv(expression_file)
            # Set gene_id as index for compatibility
            expression_data = expression_data.set_index('gene_id')
            metadata = pd.read_csv(metadata_file)
            return expression_data, metadata, True
        else:
            st.error(f"Example data files not found at: {expression_file} or {metadata_file}")
            return None, None, False
    except Exception as e:
        st.error(f"Error loading example data: {str(e)}")
        return None, None, False

def load_real_world_data():
    """Load the real-world colorectal cancer dataset"""
    try:
        expression_file = os.path.join(COLORECTAL_DATA_DIR, "GSE251845_htseq_raw_counts.csv")
        metadata_file = os.path.join(COLORECTAL_DATA_DIR, "metadata.csv")
        
        # Load data
        if os.path.exists(expression_file) and os.path.exists(metadata_file):
            # Read expression data
            expression_data = pd.read_csv(expression_file, index_col=0)
            
            # Clean up column names - remove the "_htseq.out" suffix
            expression_data.columns = [col.replace('_htseq.out', '') for col in expression_data.columns]
            
            # Load metadata
            metadata = pd.read_csv(metadata_file)
            
            return expression_data, metadata, True
        else:
            st.error(f"Colorectal data files not found at: {expression_file} or {metadata_file}")
            return None, None, False
    except Exception as e:
        st.error(f"Error loading colorectal data: {str(e)}")
        st.exception(e)  # Show full traceback for debugging
        return None, None, False

# Load data based on selection
if dataset_option == "Example Dataset":
    expression_data, metadata, data_loaded = load_example_data()
    dataset_description = """
    This is a subset of 500 genes from the colorectal cancer dataset, created for faster loading and exploration.
    It contains gene expression data across normal and tumor tissue samples from colorectal cancer patients.
    """
else: # Real-World Dataset
    expression_data, metadata, data_loaded = load_real_world_data()
    dataset_description = """
    This is the GSE251845 dataset from NCBI GEO, containing RNA-seq data from patient-matched tumor and normal 
    samples from colorectal cancer patients. The dataset includes both early-onset colorectal cancer (EOCRC, < 50 years old) 
    and later-onset colorectal cancer (LOCRC, > 50 years old) patients.
    
    - 'C' suffix = Cancer/Tumor tissue
    - 'N' suffix = Normal adjacent tissue
    
    For more details about this dataset, see the Data Attribution page.
    """

# Display dataset information
st.markdown("## Dataset Information")
st.markdown(dataset_description)

if data_loaded:
    # Display metadata overview
    st.markdown("### Metadata Overview")
    st.dataframe(metadata.head(10))
    
    # Display expression data
    st.markdown("### Expression Data Preview")
    st.dataframe(expression_data.iloc[:10, :5])
    
    # Basic statistics
    st.markdown("### Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Genes", f"{expression_data.shape[0]:,}")
    with col2:
        st.metric("Number of Samples", f"{expression_data.shape[1]:,}")
    with col3:
        if 'condition' in metadata.columns:
            condition_counts = metadata['condition'].value_counts()
            condition_str = ", ".join([f"{k}: {v}" for k, v in condition_counts.items()])
            st.metric("Sample Conditions", condition_str)
    
    # --- Expression Analysis Tools ---
    st.markdown("## Data Analysis")
    
    # Select genes for visualization
    st.markdown("### Gene Expression Visualization")
    
    # For real-world data, we need to map Ensembl IDs to gene symbols if available
    # For the example code, we'll just show the first 15 genes
    if dataset_option == "Real-World Dataset (Colorectal Cancer)":
        # Get the first 15 genes or fewer if there are fewer genes
        displayed_genes = expression_data.index[:15].tolist()
        
        # Option to search for specific gene patterns
        gene_pattern = st.text_input("Search for Ensembl gene ID (e.g., ENSG00000141510 for TP53):")
        if gene_pattern:
            matching_genes = [gene for gene in expression_data.index if gene_pattern.upper() in gene.upper()]
            if matching_genes:
                displayed_genes = matching_genes[:15]  # Limit to first 15 matches
                st.write(f"Found {len(matching_genes)} matching genes, showing first 15.")
            else:
                st.warning(f"No genes found matching '{gene_pattern}'")
    else:
        displayed_genes = expression_data.index.tolist()
    
    # Select genes for visualization
    selected_genes = st.multiselect(
        "Select genes to visualize:",
        options=displayed_genes,
        default=displayed_genes[:3] if displayed_genes else []
    )
    
    if selected_genes:
        # Prepare data for visualization
        plot_data = pd.DataFrame()
        
        for gene in selected_genes:
            gene_data = expression_data.loc[gene]
            temp_df = pd.DataFrame({
                'Gene': gene,
                'Expression': gene_data.values,
                'Sample': gene_data.index
            })
            plot_data = pd.concat([plot_data, temp_df])
        
        # Merge with metadata to get condition information
        if 'sample_id' in metadata.columns:
            plot_data = plot_data.merge(
                metadata, 
                left_on='Sample', 
                right_on='sample_id', 
                how='left'
            )
        
        # Visualization options
        viz_type = st.radio(
            "Select visualization type:",
            ["Bar Chart", "Box Plot", "Heatmap"]
        )
        
        if viz_type == "Bar Chart":
            if 'condition' in plot_data.columns:
                fig = px.bar(
                    plot_data, 
                    x='Sample', 
                    y='Expression', 
                    color='condition',
                    facet_row='Gene',
                    title="Gene Expression Levels by Sample",
                    labels={"Expression": "Expression Level", "Sample": "Sample ID"},
                    height=150 * len(selected_genes)
                )
            else:
                fig = px.bar(
                    plot_data, 
                    x='Sample', 
                    y='Expression',
                    facet_row='Gene',
                    title="Gene Expression Levels by Sample",
                    labels={"Expression": "Expression Level", "Sample": "Sample ID"},
                    height=150 * len(selected_genes)
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Box Plot":
            if 'condition' in plot_data.columns:
                fig = px.box(
                    plot_data, 
                    x='Gene', 
                    y='Expression', 
                    color='condition',
                    title="Expression Distribution by Gene and Condition",
                    labels={"Expression": "Expression Level", "Gene": "Gene ID"},
                    height=500
                )
            else:
                fig = px.box(
                    plot_data, 
                    x='Gene', 
                    y='Expression',
                    title="Expression Distribution by Gene",
                    labels={"Expression": "Expression Level", "Gene": "Gene ID"},
                    height=500
                )
                
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Heatmap":
            # Create a pivot table for the heatmap
            if len(selected_genes) > 1:
                heatmap_data = expression_data.loc[selected_genes]
                
                # Apply log transformation for better visualization
                heatmap_data = np.log1p(heatmap_data)
                
                # Create heatmap
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Sample", y="Gene", color="log(Expression + 1)"),
                    title="Gene Expression Heatmap (log-transformed)",
                    height=max(500, 30 * len(selected_genes))
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least 2 genes for the heatmap visualization.")
    
    # Differential Expression Analysis (simplified)
    if 'condition' in metadata.columns and dataset_option == "Example Dataset":
        st.markdown("### Simple Differential Expression Analysis")
        
        # Get unique conditions
        conditions = metadata['condition'].unique()
        
        # Select conditions to compare
        cond1, cond2 = st.columns(2)
        with cond1:
            condition1 = st.selectbox("Condition 1:", conditions, index=0)
        with cond2:
            condition2 = st.selectbox("Condition 2:", conditions, index=min(1, len(conditions)-1))
        
        if condition1 != condition2:
            # Get samples for each condition
            samples1 = metadata[metadata['condition'] == condition1]['sample_id'].tolist()
            samples2 = metadata[metadata['condition'] == condition2]['sample_id'].tolist()
            
            # Calculate mean expression for each gene in each condition
            expr1 = expression_data[samples1].mean(axis=1)
            expr2 = expression_data[samples2].mean(axis=1)
            
            # Calculate log2 fold change
            fc = np.log2(expr2 / expr1.replace(0, 1))  # Replace 0 with 1 to avoid division by zero
            
            # Create a DataFrame for the results
            diff_expr = pd.DataFrame({
                'Gene': expression_data.index,
                f'Mean {condition1}': expr1.values,
                f'Mean {condition2}': expr2.values,
                'Log2 Fold Change': fc.values
            }).sort_values('Log2 Fold Change', ascending=False)
            
            # Display results
            st.dataframe(diff_expr)
            
            # Volcano plot (simplified without p-values)
            fig = px.scatter(
                diff_expr,
                x='Log2 Fold Change',
                y=f'Mean {condition2}',  # Using mean expression as a proxy for significance
                hover_name='Gene',
                labels={
                    'Log2 Fold Change': f'Log2 Fold Change ({condition2} vs {condition1})',
                    f'Mean {condition2}': f'Mean Expression in {condition2}'
                },
                title=f"Expression Change: {condition2} vs {condition1}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Advanced analysis for real-world dataset
    if dataset_option == "Real-World Dataset (Colorectal Cancer)":
        st.markdown("### Tumor vs Normal Comparison")
        
        # Group samples by condition
        cancer_samples = [col for col in expression_data.columns if col.endswith('C')]
        normal_samples = [col for col in expression_data.columns if col.endswith('N')]
        
        if cancer_samples and normal_samples:
            # Calculate mean expression for each gene in each condition
            cancer_expr = expression_data[cancer_samples].mean(axis=1)
            normal_expr = expression_data[normal_samples].mean(axis=1)
            
            # Calculate log2 fold change
            # Add a small value to avoid division by zero
            epsilon = 1.0
            fc = np.log2((cancer_expr + epsilon) / (normal_expr + epsilon))
            
            # Calculate absolute expression difference
            abs_diff = cancer_expr - normal_expr
            
            # Calculate coefficient of variation within each group
            cancer_cv = expression_data[cancer_samples].std(axis=1) / (expression_data[cancer_samples].mean(axis=1) + epsilon)
            normal_cv = expression_data[normal_samples].std(axis=1) / (expression_data[normal_samples].mean(axis=1) + epsilon)
            
            # Create a DataFrame for the results
            diff_expr = pd.DataFrame({
                'Gene': expression_data.index,
                'Mean Cancer': cancer_expr.values,
                'Mean Normal': normal_expr.values,
                'Log2 Fold Change': fc.values,
                'Absolute Difference': abs_diff.values,
                'Variability Cancer': cancer_cv.values,
                'Variability Normal': normal_cv.values
            }).sort_values('Log2 Fold Change', ascending=False)
            
            # Show top differentially expressed genes
            st.markdown("#### Top 20 Genes - Tumor vs Normal")
            st.dataframe(diff_expr.head(20))
            
            # Volcano plot using fold change and mean expression as proxy for significance
            # (since we don't have p-values in this simplified analysis)
            fig = px.scatter(
                diff_expr,
                x='Log2 Fold Change',
                y='Absolute Difference',
                hover_name='Gene',
                title="Expression Change: Tumor vs Normal Tissue",
                labels={
                    'Log2 Fold Change': 'Log2 Fold Change (Tumor vs Normal)',
                    'Absolute Difference': 'Absolute Expression Difference'
                },
                color='Absolute Difference',
                size='Mean Cancer',
                color_continuous_scale='RdBu_r'
            )
            
            # Add horizontal and vertical reference lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the differential expression results
            csv = diff_expr.to_csv(index=False)
            st.download_button(
                label="Download Differential Expression Results",
                data=csv,
                file_name="differential_expression_tumor_vs_normal.csv",
                mime="text/csv"
            )
else:
    st.error("Failed to load dataset. Please check the data files and paths.")

# --- Additional Information ---
st.markdown("## About the Data")
st.info("""
For detailed information about data sources, processing methods, citations, and licensing, 
please visit the **Data Attribution** page via the sidebar navigation.
""")

# --- Data Explorer Notes ---
with st.expander("Data Analysis Notes"):
    st.markdown("""
    **Expression Data Analysis Tips:**
    
    - For the real-world dataset, gene identifiers are Ensembl IDs (e.g., ENSG00000141510)
    - The log transformation is applied to expression values for better visualization in heatmaps
    - For tumor vs. normal comparison, we use log2 fold change to identify differentially expressed genes
    - Positive fold change values indicate higher expression in tumor samples
    - Negative fold change values indicate higher expression in normal samples
    
    **Limitations:**
    
    - This simplified analysis doesn't include statistical testing (p-values, FDR correction)
    - For real-world applications, more sophisticated differential expression analysis tools like DESeq2 or edgeR are recommended
    - Gene annotation data (function, pathways, etc.) is not included in this explorer
    """) 