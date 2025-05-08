import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import os
import json
import random
import bcrypt # Need to install: pip install bcrypt
import traceback # For better error reporting during analysis
import base64
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
        border-bottom: 1px solid rgba(167, 139, 250, 0.2);
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
        border: 1px solid rgba(167, 139, 250, 0.2) !important;
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
        box-shadow: 0 4px 12px rgba(167, 139, 250, 0.3) !important;
    }
    
    /* Improving form elements */
    div[data-baseweb="input"] {
        border-radius: 8px !important;
    }
    
    /* Info and warning boxes */
    div[data-testid="stInfo"] {
        background-color: #F5F3FF !important;
        border-left-color: #A78BFA !important;
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
        background-color: #F5F3FF;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Data tables styling */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Authentication Logic Definitions ---
# IMPORTANT: Ensure you have a .streamlit/secrets.toml file
# with the structure shown in the comments below.

def verify_user(username, password):
    """
    Verifies username and password against the list of users
    and their hashed passwords stored in st.secrets.
    Assumes secrets.toml structure:
    [auth]
    users = [
        {name = "...", username = "...", hashed_password = "..."},
        {name = "...", username = "...", hashed_password = "..."}
    ]
    """
    try:
        all_users_data = st.secrets.get("auth", {}).get("users", [])
        user_data = next((user for user in all_users_data if user.get("username") == username), None)

        if user_data:
            hashed_password_from_secrets = user_data.get("hashed_password")
            if hashed_password_from_secrets:
                password_bytes = password.encode('utf-8')
                hashed_password_bytes = hashed_password_from_secrets.encode('utf-8')
                # Use bcrypt.checkpw for secure comparison
                if bcrypt.checkpw(password_bytes, hashed_password_bytes):
                    return True # Password matches
    except FileNotFoundError:
         st.error("Secrets file not found. Please ensure `.streamlit/secrets.toml` exists.")
         return False
    except Exception as e:
        st.error(f"An error occurred during user verification: {e}")
        return False
    return False # User not found or password doesn't match

def login_screen():
    """Displays login form and handles authentication."""
    # Apply custom styling
    apply_modern_styles()
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Login Required")
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            Welcome to the Antigen Prioritizer Tool. Please log in to continue.
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if verify_user(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username # Store username
                    st.rerun() # Immediately rerun the script
                else:
                    st.error("Incorrect username or password")

# --- Path Handling ---
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(APP_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "data"
    COLORECTAL_DATA_DIR = os.path.join(DATA_DIR, "collarectal")
    EXAMPLE_DATA_DIR = os.path.join(DATA_DIR, "example")
except NameError:
    APP_DIR = os.getcwd()
    if os.path.exists("data"):
        DATA_DIR = "data"
    else:
        DATA_DIR = os.path.join(os.path.dirname(APP_DIR), "data")
    COLORECTAL_DATA_DIR = os.path.join(DATA_DIR, "collarectal")
    EXAMPLE_DATA_DIR = os.path.join(DATA_DIR, "example")

# --- Page setup ---
st.set_page_config(
    page_title="Early-Stage Antigen Prioritizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Apply modern styling ---
apply_modern_styles()

# --- Initialize session state variables ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Initialize download_date if not already set
if "download_date" not in st.session_state:
    from datetime import datetime
    st.session_state["download_date"] = datetime.now().strftime("%B %d, %Y")

# --- Main App Execution Control (Authentication Check) ---
# Check authentication status right at the beginning
if not st.session_state["authenticated"]:
    login_screen()
    # Stop execution flow here if not authenticated
    # Nothing below this line will run until authenticated=True
    st.stop()

# --- User IS Authenticated: Show Logout & Run Main App ---

# Display logged-in user info and logout button (in the sidebar)
with st.sidebar:
    user_container = st.container()
    with user_container:
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 0.5rem; background-color: #F5F3FF; border-radius: 8px; margin-bottom: 1rem;">
            <div style="width: 30px; height: 30px; border-radius: 50%; background-color: #A78BFA; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                <span style="color: white; font-weight: bold;">{st.session_state.get('username', 'N/A')[0].upper()}</span>
            </div>
            <div>
                <div style="font-weight: 500;">Logged in as:</div>
                <div>{st.session_state.get('username', 'N/A')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state.pop('username', None) # Clear username
            st.rerun() # Rerun to go back to the login screen

# --- OpenAI API configuration (Runs only if authenticated) ---
openai_available = False
client = None
try:
    from openai import OpenAI
    
    api_key = st.secrets.get("openai", {}).get("api_key")
    if api_key:
        # Simple initialization without the custom HTTP client
        client = OpenAI(api_key=api_key)
        openai_available = True
except ImportError:
    pass
except Exception as e:
    pass # Silently handle OpenAI setup issues


# --- Helper function to call OpenAI ---
def get_json_from_prompt(prompt: str) -> dict:
    """Helper function to call OpenAI and return the JSON-parsed response."""
    if not openai_available or client is None:
        return {"error": "OpenAI API client not configured or available."}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1", # or "o3-mini" if you prefer
            messages=[
                {"role": "system", "content": "You are a knowledgeable cancer immunologist who returns valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

# --- Refined AI Review Function ---
def get_ai_review(results_df, case_group, control_group, min_fc, max_p):
    """Generate an AI review focusing on the top 3 vaccine candidates."""
    if not openai_available:
        st.warning("OpenAI is not available for review.")
        return None
    if results_df.empty:
        st.info("No results to review.")
        return None

    # Select top 20 rows for analysis, but AI will only return top 3
    required_cols = ['rank', 'gene_id', 'protein_id', 'protein_name', 'fold_change', 'log2_fold_change', 'p_value', 'cellular_location', 'immunogenicity_score']
    available_cols = [col for col in required_cols if col in results_df.columns]
    top_genes_df = results_df.head(20)[available_cols]
    top_genes_dict = top_genes_df.to_dict(orient='records')

    prompt = f"""
    Analyze these potential cancer vaccine targets comparing {case_group} vs {control_group}.
    
    Based on the top 20 candidates provided below, select the 3 most promising targets:
    {json.dumps(top_genes_dict, indent=2)}
    
    Return a JSON object with the following structure:
    {{
      "top_candidates": [
        {{
          "rank": 1,
          "gene_id": "...",
          "protein_id": "...",
          "protein_name": "...",
          "cellular_location": "...",
          "rationale": "Explain why this is a promising target (2-3 sentences)",
          "novelty_score": 1-10,
          "novelty_explanation": "Explain how novel/unique this target is (1-2 sentences)"
        }},
        // candidate 2 and 3 with same structure
      ],
      "scientific_summary": "A brief scientific assessment of these three targets as a group (2-3 sentences)"
    }}
    
    Focus on cellular location (membrane/secreted are preferred), expression levels, and cancer specificity.
    """
    return get_json_from_prompt(prompt)

# --- Display AI Review Section ---
def display_ai_review(ai_review):
    """Format the AI review results in a visually appealing way"""
    if not ai_review or "error" in ai_review:
        return
    
    # Check if top_candidates exists in the response
    if "top_candidates" in ai_review and isinstance(ai_review["top_candidates"], list):
        candidates = ai_review["top_candidates"]
        
        # Create three columns for the candidates
        if len(candidates) > 0:
            cols = st.columns(min(len(candidates), 3))
            
            for i, (col, candidate) in enumerate(zip(cols, candidates[:3])):
                with col:
                    # Create a card-like container for each candidate
                    st.markdown(f"""
                    <div style="background-color: #F5F3FF; padding: 1.5rem; border-radius: 10px; height: 100%;">
                        <h3 style="color: #6D28D9; margin-top: 0;">#{i+1}: {candidate.get('protein_name', 'Unknown')}</h3>
                        <p><strong>Gene ID:</strong> {candidate.get('gene_id', 'N/A')}</p>
                        <p><strong>Location:</strong> {candidate.get('cellular_location', 'Unknown')}</p>
                        <p><strong>Novelty:</strong> {candidate.get('novelty_score', 'N/A')}/10</p>
                        <hr style="border-color: rgba(109, 40, 217, 0.2);">
                        <p><em>{candidate.get('rationale', 'No rationale provided.')}</em></p>
                        <p style="font-size: 0.9rem; opacity: 0.8;">{candidate.get('novelty_explanation', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display the scientific summary without header
        if "scientific_summary" in ai_review:
            st.info(ai_review["scientific_summary"])

# --- Title and introduction (Shown only if authenticated) ---
st.title("Early-Stage Antigen Prioritizer")
st.markdown("""
<div style="background-color: #F5F3FF; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
    <h3 style="margin-top: 0; margin-bottom: 0.8rem; color: #4B3F72;">About This Tool</h3>
    <p>
    This tool identifies and ranks potential protein targets for prophylactic cancer vaccines
    based on gene expression data and predicted immunogenicity features.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar for inputs (Shown only if authenticated) ---
with st.sidebar:
    st.markdown("## Data Input")
    data_option = st.radio(
        "Select data source:",
        ("Use example data", "Use real-world colorectal cancer dataset", "Upload your own data"),
        key="data_source_radio"
    )

# --- Data Loading Section (Runs only if authenticated) ---
# Initialize variables
expression_data = None
metadata = None
gene_protein_map = None
protein_annotations = None
data_loaded_successfully = False

# Load data based on selection
if data_option == "Use real-world colorectal cancer dataset":
    try:
        colorectal_path = os.path.join(DATA_DIR, "collarectal")
        expr_file = os.path.join(colorectal_path, "GSE251845_htseq_raw_counts.csv")
        meta_file = os.path.join(colorectal_path, "metadata.csv")
        map_file = os.path.join(colorectal_path, "gene_protein_map.csv")
        annot_file = os.path.join(colorectal_path, "protein_annotations.csv")

        if all(os.path.exists(f) for f in [expr_file, meta_file, map_file, annot_file]):
            expression_data_raw = pd.read_csv(expr_file, index_col=0)
            gene_ids = expression_data_raw.index.astype(str).tolist() # Ensure gene IDs are strings
            expression_data = pd.DataFrame({'gene_id': gene_ids})
            for col in expression_data_raw.columns:
                clean_col = col.replace('_htseq.out', '')
                expression_data[clean_col] = expression_data_raw[col].values
            metadata = pd.read_csv(meta_file)
            gene_protein_map = pd.read_csv(map_file)
            protein_annotations = pd.read_csv(annot_file)
            data_loaded_successfully = True
        else:
            with st.sidebar:
                st.error("Missing required data files. Please check the data directory.")
    except Exception as e:
        with st.sidebar:
            st.error("Error loading data. Please check file formats.")

elif data_option == "Use example data":
    try:
        expression_path = os.path.join(EXAMPLE_DATA_DIR, "expression.csv")
        metadata_path = os.path.join(EXAMPLE_DATA_DIR, "metadata.csv")
        map_path = os.path.join(EXAMPLE_DATA_DIR, "gene_protein_map.csv")
        annot_path = os.path.join(EXAMPLE_DATA_DIR, "protein_annotations.csv")

        if all(os.path.exists(p) for p in [expression_path, metadata_path, map_path, annot_path]):
            expression_data = pd.read_csv(expression_path)
            metadata = pd.read_csv(metadata_path)
            gene_protein_map = pd.read_csv(map_path)
            protein_annotations = pd.read_csv(annot_path)
            data_loaded_successfully = True
        else:
            st.sidebar.error("Missing example data files.")
    except Exception as e:
        st.sidebar.error("Error loading example data.")

# Upload your own data section
elif data_option == "Upload your own data":
    expression_file = st.sidebar.file_uploader("Upload expression data (CSV)", type=["csv"], key="exp_upload")
    metadata_file = st.sidebar.file_uploader("Upload metadata (CSV)", type=["csv"], key="meta_upload")
    gene_protein_file = st.sidebar.file_uploader("Upload gene-protein mapping (CSV)", type=["csv"], key="map_upload")
    protein_annotations_file = st.sidebar.file_uploader("Upload protein annotations (CSV)", type=["csv"], key="annot_upload")

    if expression_file and metadata_file and gene_protein_file and protein_annotations_file:
        try:
            expression_data = pd.read_csv(expression_file)
            # Ensure gene_id is string if it exists as a column
            if 'gene_id' in expression_data.columns:
                expression_data['gene_id'] = expression_data['gene_id'].astype(str)
            metadata = pd.read_csv(metadata_file)
            gene_protein_map = pd.read_csv(gene_protein_file)
            protein_annotations = pd.read_csv(protein_annotations_file)
            data_loaded_successfully = True
        except Exception as e:
            with st.sidebar:
                st.error("Error reading files. Check CSV format and structure.")
            data_loaded_successfully = False


# --- Analysis Parameters (Shown only if authenticated & data loaded) ---
st.sidebar.header("Analysis Parameters")
analyze_button = False
control_group = None
case_group = None
min_fold_change = 2.0 # Default
max_pvalue = 0.05 # Default

if data_loaded_successfully and metadata is not None:
    conditions = metadata['condition'].unique().tolist()
    if len(conditions) >= 2:
        control_group = st.sidebar.selectbox("Select control group:", conditions, index=min(1, len(conditions)-1), key="control_select") # Safer index
        case_options = [c for c in conditions if c != control_group]
        if case_options:
             case_group = st.sidebar.selectbox("Select case group:", case_options, index=0, key="case_select")
        else:
             st.sidebar.error("Need at least two distinct conditions.")

        if case_group:
            min_fold_change = st.sidebar.slider("Minimum Fold Change:", 1.0, 10.0, 2.0, 0.1, key="fc_slider")
            max_pvalue = st.sidebar.slider("Maximum P-value:", 0.001, 0.1, 0.05, 0.001, format="%.3f", key="pval_slider")
            analyze_button = st.sidebar.button("Prioritize Targets", key="analyze_button", use_container_width=True)
    else:
        st.sidebar.error("Metadata needs at least two unique conditions.")
elif not data_loaded_successfully:
    with st.sidebar:
        st.markdown("""
        <div style="background-color: #F8F9FA; padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem; font-size: 0.9rem; color: #495057;">
            Configure analysis after data is loaded.
        </div>
        """, unsafe_allow_html=True)

# --- Main content Analysis Execution (Runs only if button pressed & authenticated) ---
if analyze_button and data_loaded_successfully and case_group is not None and control_group is not None:
    # --- Input Validation ---
    valid_input = True
    validation_error = None
    
    # Basic check: Ensure essential dataframes are not None
    if expression_data is None or metadata is None or gene_protein_map is None or protein_annotations is None:
         validation_error = "Missing one or more required datasets."
         valid_input = False
    else:
        # More specific validation checks in a compact form
        if 'gene_id' not in expression_data.columns and expression_data.index.name != 'gene_id':
            validation_error = "Expression data must contain gene_id column or index."
            valid_input = False
        elif not {'sample_id', 'condition'}.issubset(metadata.columns):
            validation_error = "Metadata missing required columns: sample_id or condition."
            valid_input = False
        elif not {'gene_id', 'protein_id'}.issubset(gene_protein_map.columns):
            validation_error = "Gene-protein map missing required columns."
            valid_input = False
        elif not {'protein_id', 'cellular_location', 'protein_name'}.issubset(protein_annotations.columns):
            validation_error = "Protein annotations missing required columns."
            valid_input = False

    if not valid_input:
        st.error(validation_error)
    else:
        with st.spinner("Analyzing data... This might take a moment."):
            try:
                # --- Backend Logic Definitions (ensure these are defined above or imported) ---
                # NOTE: These are placeholder definitions matching your original code structure.
                # Replace with your actual, potentially more robust, function implementations.
                def perform_differential_expression(expr_data, control_samples, case_samples, min_fc, max_p):
                    # Create a copy to avoid modifying the original dataframe if passed directly
                    df_expr = expr_data.copy()
                    if 'gene_id' not in df_expr.columns and df_expr.index.name == 'gene_id':
                         df_expr = df_expr.reset_index() # Ensure gene_id is a column

                    # Ensure gene_id is string for consistency
                    df_expr['gene_id'] = df_expr['gene_id'].astype(str)
                    results = pd.DataFrame({'gene_id': df_expr['gene_id'].unique()}) # Start with unique gene_ids

                    # Select numeric columns only for calculations
                    numeric_cols = df_expr.select_dtypes(include=np.number).columns
                    control_samples_numeric = [s for s in control_samples if s in numeric_cols]
                    case_samples_numeric = [s for s in case_samples if s in numeric_cols]

                    if not control_samples_numeric or not case_samples_numeric:
                        return pd.DataFrame() # Return empty frame if no numeric columns

                    # Calculate means 
                    means = df_expr.groupby('gene_id')[control_samples_numeric + case_samples_numeric].mean()
                    results['mean_control'] = results['gene_id'].map(means[control_samples_numeric].mean(axis=1))
                    results['mean_case'] = results['gene_id'].map(means[case_samples_numeric].mean(axis=1))
                    results.dropna(subset=['mean_control', 'mean_case'], inplace=True)

                    epsilon = 1.0 # Add small constant for stability
                    results['log2_mean_control'] = np.log2(results['mean_control'] + epsilon)
                    results['log2_mean_case'] = np.log2(results['mean_case'] + epsilon)
                    results['log2_fold_change'] = results['log2_mean_case'] - results['log2_mean_control']
                    results['fold_change'] = (results['mean_case'] + epsilon) / (results['mean_control'] + epsilon)

                    # Calculate p-values using scipy.stats.ttest_ind
                    p_values = []
                    gene_id_map = df_expr.set_index('gene_id') # For quick row lookup

                    for gene_id in results['gene_id']:
                        try:
                             gene_rows = gene_id_map.loc[[gene_id]] if isinstance(gene_id_map.index, pd.MultiIndex) else gene_id_map.loc[[gene_id]]
                             control_expr = gene_rows[control_samples_numeric].values.flatten().astype(float)
                             case_expr = gene_rows[case_samples_numeric].values.flatten().astype(float)
                             control_expr_clean = control_expr[~np.isnan(control_expr)]
                             case_expr_clean = case_expr[~np.isnan(case_expr)]

                             if len(control_expr_clean) >= 2 and len(case_expr_clean) >= 2:
                                 t_stat, p_val = stats.ttest_ind(case_expr_clean, control_expr_clean, equal_var=False, nan_policy='omit')
                                 p_values.append(p_val)
                             else:
                                 p_values.append(np.nan)
                        except Exception:
                             p_values.append(np.nan)

                    if len(p_values) == len(results):
                         results['p_value'] = p_values
                    else:
                         return pd.DataFrame() # Return empty frame if p-value calculation failed

                    results.dropna(subset=['p_value'], inplace=True) # Drop rows where p-value is NaN

                    # Filter based on fold change and p-value
                    filtered_results = results[
                        (results['fold_change'] >= min_fc) & 
                        (results['p_value'] <= max_p)
                    ].copy() 

                    filtered_results.sort_values('fold_change', ascending=False, inplace=True)
                    return filtered_results

                def map_genes_to_proteins(de_results, gp_map, prot_ann):
                    if de_results.empty: return pd.DataFrame()
                    de_results['gene_id'] = de_results['gene_id'].astype(str)
                    gp_map['gene_id'] = gp_map['gene_id'].astype(str)
                    gp_map['protein_id'] = gp_map['protein_id'].astype(str)
                    prot_ann['protein_id'] = prot_ann['protein_id'].astype(str)

                    merged = pd.merge(de_results, gp_map[['gene_id', 'protein_id']].drop_duplicates(), on='gene_id', how='left')
                    merged.dropna(subset=['protein_id'], inplace=True)
                    final = pd.merge(merged, prot_ann[['protein_id', 'cellular_location', 'protein_name', 'immunogenicity_score']].drop_duplicates(subset=['protein_id']), on='protein_id', how='left')
                    # Use fillna strategically AFTER the final merge
                    final['cellular_location'].fillna('unknown', inplace=True)
                    final['protein_name'].fillna('unknown', inplace=True)
                    # Handle missing immunogenicity score - crucial change here
                    if 'immunogenicity_score' not in final.columns:
                         st.warning("Annotations file missing 'immunogenicity_score'. Applying default logic.")
                         final['immunogenicity_score'] = 0 # Default to 0
                         final.loc[final['cellular_location'].str.lower() == 'membrane', 'immunogenicity_score'] = 3
                         final.loc[final['cellular_location'].str.lower() == 'secreted', 'immunogenicity_score'] = 3
                    else:
                         # Fill NaNs in the existing column if it exists
                         final['immunogenicity_score'].fillna(0, inplace=True) # Default score if missing for a protein

                    return final

                def calculate_final_rank(scored_data):
                    if scored_data.empty: return pd.DataFrame()
                    scored_data.dropna(subset=['p_value', 'fold_change', 'immunogenicity_score'], inplace=True)

                    # Ensure scores are numeric, coercing errors
                    scored_data['fold_change'] = pd.to_numeric(scored_data['fold_change'], errors='coerce')
                    scored_data['p_value'] = pd.to_numeric(scored_data['p_value'], errors='coerce')
                    scored_data['immunogenicity_score'] = pd.to_numeric(scored_data['immunogenicity_score'], errors='coerce')
                    scored_data.dropna(subset=['fold_change', 'p_value', 'immunogenicity_score'], inplace=True) # Drop rows where conversion failed

                    # Normalize immunogenicity score (assuming 0-3 range from your file/logic)
                    max_imm_score = 3.0 # Define the max possible score for normalization
                    scored_data['immunogenicity_norm'] = scored_data['immunogenicity_score'] / max_imm_score
                    scored_data['immunogenicity_norm'] = scored_data['immunogenicity_norm'].clip(0, 1) # Ensure it's between 0 and 1

                    # Calculate combined score (adjust weights as needed)
                    # Using log2 fold change might be more robust than raw fold change if values are huge
                    # Ensure log2 fold change is calculated if not already present
                    if 'log2_fold_change' not in scored_data.columns:
                        epsilon = 1e-9 # small epsilon for log calc if means are used directly
                        mean_control_col = 'mean_control' # ensure these cols exist
                        mean_case_col = 'mean_case'
                        if mean_control_col in scored_data.columns and mean_case_col in scored_data.columns:
                            scored_data['log2_fold_change'] = np.log2((scored_data[mean_case_col] + epsilon) / (scored_data[mean_control_col] + epsilon))
                        else: # Fallback or error if means aren't present for recalculation
                             st.warning("Log2 fold change not found, using raw fold change for ranking (less ideal).")
                             scored_data['log2_fold_change'] = np.log2(scored_data['fold_change']) # Less robust

                    # Weighting: Higher log2_fc is better, lower p_value is better (so use 1-p_value), higher immunogenicity is better
                    # Ensure log2 fold change is positive for upregulated focus, or use absolute value if bidirectional
                    # Let's assume focus on upregulation for score (log2_fold_change > 0)
                    positive_log2fc = scored_data['log2_fold_change'].clip(lower=0) # Treat negative fold changes as 0 score contribution

                    # Normalize log2fc (optional but can help balance scales) - simple scaling here
                    max_log2fc = positive_log2fc.max()
                    if max_log2fc > 0:
                         normalized_log2fc = positive_log2fc / max_log2fc
                    else:
                         normalized_log2fc = pd.Series(0.0, index=scored_data.index) # Handle case where no upregulation


                    scored_data['combined_score'] = (
                        normalized_log2fc * 0.4 +           # Weight for log2 fold change (normalized)
                        (1 - scored_data['p_value']) * 0.3 + # Weight for p-value (lower is better)
                        scored_data['immunogenicity_norm'] * 0.3  # Weight for immunogenicity (normalized)
                    )

                    ranked_data = scored_data.sort_values(by='combined_score', ascending=False).reset_index(drop=True)
                    ranked_data['rank'] = range(1, len(ranked_data) + 1)
                    return ranked_data

                # --- Run Analysis Pipeline ---
                control_samples = metadata[metadata['condition'] == control_group]['sample_id'].tolist()
                case_samples = metadata[metadata['condition'] == case_group]['sample_id'].tolist()

                if not control_samples or not case_samples:
                    st.error(f"Could not find samples for '{control_group}' or '{case_group}'. Check metadata.")
                else:
                    # Perform analysis
                    de_results = perform_differential_expression(
                        expression_data, control_samples, case_samples, min_fold_change, max_pvalue
                    )

                    if not de_results.empty:
                        mapped_results = map_genes_to_proteins(
                            de_results, gene_protein_map, protein_annotations
                        )
                        # Note: calculate_immunogenicity_score was removed as mapping now handles the score column
                        final_results = calculate_final_rank(mapped_results) # Pass mapped results directly

                        # --- Display AI Review Section ---
                        if openai_available:
                            with st.spinner(""):
                                ai_review = get_ai_review(final_results, case_group, control_group, min_fold_change, max_pvalue)
                                display_ai_review(ai_review)
                        
                        # --- Display Ranked Targets Table ---
                        display_columns = [
                            'rank', 'gene_id', 'protein_id', 'protein_name',
                            'log2_fold_change', 'fold_change', 'p_value',
                            'cellular_location', 'immunogenicity_score', 'combined_score'
                        ]
                        # Ensure columns exist, handle potential missing columns gracefully
                        actual_display_columns = [col for col in display_columns if col in final_results.columns]
                        
                        display_table = final_results[actual_display_columns].copy()

                        # Apply formatting safely
                        formatters = {
                            'log2_fold_change': lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x,
                            'fold_change': lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x,
                            'p_value': lambda x: f"{x:.3e}" if pd.notna(x) and isinstance(x, (int, float)) else x,
                            'combined_score': lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else x,
                            'immunogenicity_score': lambda x: f"{x:.1f}" if pd.notna(x) and isinstance(x, (int, float)) else x, # Format immuno score
                        }
                        for col, formatter in formatters.items():
                            if col in display_table.columns:
                                display_table[col] = display_table[col].apply(formatter)

                        st.dataframe(display_table.set_index('rank'))

                        # --- Visualization Section ---
                        if not final_results.empty:
                            top_results_vis = final_results.head(10)
                            try:
                                # Use gene_id from final_results which should match expression_data index/column
                                plot_genes = top_results_vis['gene_id'].tolist()
                                plot_samples = control_samples + case_samples

                                # Prepare expression data for plotting - ensure index matches plot_genes format (string)
                                if expression_data.index.name == 'gene_id':
                                     expr_plot_base = expression_data.loc[expression_data.index.astype(str).isin(plot_genes), plot_samples]
                                     expr_plot_base = expr_plot_base.reset_index() # gene_id becomes column
                                else: # Assume gene_id is a column
                                     expr_plot_base = expression_data[expression_data['gene_id'].astype(str).isin(plot_genes)][['gene_id'] + plot_samples]

                                if not expr_plot_base.empty:
                                    # Melt
                                    plot_df_melted = expr_plot_base.melt(id_vars='gene_id', var_name='sample_id', value_name='Expression')
                                    # Add group info
                                    group_map = metadata.set_index('sample_id')['condition']
                                    plot_df_melted['Group'] = plot_df_melted['sample_id'].map(group_map)
                                    # Filter
                                    plot_df_filtered = plot_df_melted[plot_df_melted['Group'].isin([control_group, case_group])]

                                    if not plot_df_filtered.empty:
                                        # Map gene_id to protein_name for labels
                                        gene_name_map = final_results.set_index('gene_id')['protein_name']
                                        plot_df_filtered['Gene Label'] = plot_df_filtered['gene_id'].map(gene_name_map).fillna(plot_df_filtered['gene_id']) # Fallback to gene_id

                                        fig_box = px.box(
                                            plot_df_filtered, x='Gene Label', y='Expression', color='Group',
                                            title="",
                                            points="all", color_discrete_map={control_group: 'blue', case_group: 'red'},
                                            labels={'Gene Label': 'Gene/Protein', 'Expression': 'Expression Level'}
                                        )
                                        fig_box.update_layout(xaxis_tickangle=-45)
                                        st.plotly_chart(fig_box, use_container_width=True)
                            except Exception:
                                pass

                        # --- Download Button ---
                        try:
                            csv = final_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results", data=csv,
                                file_name=f"antigen_targets_{case_group}_vs_{control_group}.csv",
                                mime='text/csv', key='download-csv'
                            )
                        except Exception:
                            pass

                    else: # If de_results empty
                        pass

            except Exception:
                 pass # Silently handle any analysis errors

# Add minimal footer without any text
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)