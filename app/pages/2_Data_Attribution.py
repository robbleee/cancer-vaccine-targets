import streamlit as st

# --- Page setup ---
st.set_page_config(
    page_title="Data Attribution - Early-Stage Antigen Prioritizer", 
    layout="wide"
)

# --- Title and introduction ---
st.title("Data Attribution")
st.markdown("""
This page provides information about the datasets used in this application, their sources, and relevant citations.
""")

# --- Example Dataset Attribution ---
st.header("Example Dataset")
st.markdown("""
The example dataset included in this application is a **subset of the colorectal cancer dataset** designed for faster loading and exploration. 
It contains:
- 500 genes sampled from the full colorectal cancer dataset
- The same sample structure as the full dataset (tumor and normal conditions)
- Simulated protein annotations and immunogenicity scores based on the gene IDs

This subset dataset provides a quicker way to test the application's functionality while maintaining the real-world nature of the gene expression patterns.
""")

# --- Real-World Dataset Attribution ---
st.header("Real-World Test Dataset")
st.markdown("""
### Early-Onset Colorectal Cancer Gene Expression Dataset

**Dataset Source**: [GSE251845](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251845) from NCBI Gene Expression Omnibus (GEO)

**Title**: Identification of unique gene expression and splicing events in early-onset colorectal cancer

**Authors**: Marx OM, Yochum GS, Koltun WA, Mankarious MM

**Publication**: Marx OM, Mankarious MM, Tutino J, He Z, Burns-Huang KE, Saha PP, Abou-Gharbia M, Vining K, McHugh K, Esterline B, Nault R, Han L, Miller DT, Meyers CL, Meudt JJ, Walter KR, Lu M, Zhang L, Betts C, Koltun WA, Yochum GS. (2024) Identification of unique gene expression and splicing events in early-onset colorectal cancer. *Frontiers in Oncology*. doi: 10.3389/fonc.2024.1365762. PMID: 38680862

**Description**: 
This dataset contains RNA-sequencing data from patient-matched tumors and adjacent, uninvolved (normal) colonic segments from early-onset colorectal cancer (EOCRC, < 50 years old) and later-onset colorectal cancer (LOCRC, > 50 years old) patients. The dataset includes 21 EOCRC patients and 22 LOCRC patients. The researchers identified an eight-gene signature in EOCRC and uncovered differences in gene expression profiles between early and late-onset colorectal cancer.

**Data Collection**:
- Tissue samples: Surgically resected tumors and patient-matched adjacent normal colonic segments
- RNA extraction: Samples were homogenized, and RNA was extracted using chloroform separation and RNAEasy Mini Kit
- Sequencing: cDNA libraries were prepared using standard Illumina protocols and sequenced on Illumina NovaSeq 6000
- Data processing: Alignment with STAR v2.7.3, gene counting with HTSeq, using hg38 reference genome

**Funding**: NIH grant R03 CA279861

**License**: Public data made available through GEO. Users are expected to respect standard scientific practice regarding citation and acknowledgment.

**Date of Access**: {st.session_state.get('download_date', 'May 2024')}
""")

# --- Data Usage Policy ---
st.header("Data Usage Policy")
st.markdown("""
The datasets used in this application are provided for research and educational purposes only. 
When using results derived from these datasets in publications or presentations, please cite the original sources as indicated above.

For the real-world dataset, please acknowledge:
> Marx et al. (2024) "Identification of unique gene expression and splicing events in early-onset colorectal cancer." Frontiers in Oncology. doi: 10.3389/fonc.2024.1365762.
""")

# --- Technical Information ---
st.header("Technical Information")
st.markdown("""
### Data Integration Process

The real-world test dataset was processed as follows:
1. Raw count data was downloaded from GEO accession GSE251845
2. Sample metadata was created to identify tumor and normal samples
3. Data was formatted for compatibility with our application's analysis pipeline
4. Gene annotations were mapped to standard identifiers

### Known Limitations
- The dataset predominantly consists of colorectal cancer samples and may not be representative of other cancer types
- Gene expression data alone doesn't capture all relevant biological processes for vaccine target identification
- Sample size limitations should be considered when interpreting results

For questions regarding data usage in this application, please contact the application administrators.
""")

# --- Data Structure Information ---
st.header("Dataset Structure")
st.markdown("""
Both the example and real-world datasets follow the same structure:

1. **Expression Data**: Gene expression values across samples
   - `gene_id`: Gene identifier
   - Additional columns: Sample identifiers with expression values

2. **Metadata**: Sample information
   - `sample_id`: Sample identifier
   - `condition`: Sample condition (normal, precancer, early_cancer)

3. **Gene-Protein Mapping**: Links genes to proteins
   - `gene_id`: Gene identifier
   - `protein_id`: Corresponding protein identifier

4. **Protein Annotations**: Protein information
   - `protein_id`: Protein identifier
   - `protein_name`: Human-readable protein name
   - `cellular_location`: Subcellular localization of the protein
""")

# --- Custom Data Guidelines ---
st.header("Using Your Own Data")
st.markdown("""
When uploading your own data to the application, please ensure:

1. Your data follows the structure outlined above
2. You have appropriate rights to use the data
3. Data has been properly anonymized if derived from patient samples
4. You maintain appropriate citations when publishing results

For guidance on preparing your data in the correct format, please refer to the application documentation.
""")

# --- Disclaimer ---
st.markdown("""
---
**Disclaimer**: This application is designed for educational and research purposes. 
Results should be validated with appropriate experimental methods before drawing scientific conclusions 
or making clinical decisions.
""") 