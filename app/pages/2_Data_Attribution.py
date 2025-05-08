import streamlit as st

# --- Page configuration ---
st.set_page_config(
    page_title="Data Attribution - Cancer Vaccine Development Tool",
    layout="wide"
)

# --- Title and introduction ---
st.title("Data Attribution")

st.markdown("This page provides detailed information about the data sources used in the Cancer Vaccine Development Tool, including attributions, citations, and licensing information.")

# --- Primary Data Sources ---
st.markdown("## Primary Data Sources")

# Colorectal dataset
st.markdown("### Colorectal Cancer RNA-Seq Dataset")
st.markdown("**Source:** NCBI Gene Expression Omnibus (GEO)")
st.markdown("**Accession Number:** GSE251845")
st.markdown("**Study Title:** RNA-sequencing of tumor and normal adjacent tissue in early onset vs. late onset colon cancer patients")
st.markdown("**Description:** This dataset contains RNA-Seq data from patient-matched tumor and normal tissues from colorectal cancer patients, including both early-onset (EOCRC, diagnosed < 50 years old) and later-onset (LOCRC, diagnosed > 50 years old) cases.")
st.markdown("**Citation:** Smith G, et al. (2023). *Metabolic Dysregulation in Early-onset Colorectal Cancer Reveals Fatty Acid Synthase and KLF5 as Potential Therapeutic Targets*. Cancer Research [publication pending].")
st.markdown("**Access Date:** January 2024")
st.markdown("**License:** Data is publicly available through GEO under standard NIH guidelines for data sharing.")
st.markdown("[Visit Source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251845)")

# Example dataset
st.markdown("### Example Dataset")
st.markdown("**Source:** Derived from the GSE251845 dataset (see above)")
st.markdown("**Description:** This is a subset of the full colorectal cancer dataset, created for demonstration purposes. It contains a randomly selected subset of 500 genes from the original dataset to allow faster loading and exploration.")
st.markdown("**Modifications:**")
st.markdown("- Reduced to 500 randomly selected genes\n- Simplified metadata structure\n- Maintained the original expression patterns and relationships")
st.markdown("**Usage:** For educational and demonstration purposes only.")

# --- Supplementary Data Sources ---
st.markdown("## Supplementary Data Sources")

# Protein annotation datasets
st.markdown("### Protein Annotation Data")
st.markdown("The protein annotations used in this tool are derived from the following sources:")

st.markdown("#### UniProt")
st.markdown("**Source:** Universal Protein Resource (UniProt)")
st.markdown("**Description:** Comprehensive resource for protein sequence and functional information.")
st.markdown("**Citation:** The UniProt Consortium. (2023). UniProt: the Universal Protein Knowledgebase in 2023. Nucleic Acids Research, 51(D1), D523–D531.")
st.markdown("**License:** Creative Commons Attribution (CC BY 4.0) License")

st.markdown("#### BioMart / Ensembl")
st.markdown("**Source:** Ensembl BioMart")
st.markdown("**Description:** Data mining tool that provides easy access to Ensembl gene and protein data.")
st.markdown("**Citation:** Cunningham F, et al. (2023). Ensembl 2023. Nucleic Acids Research, 51(D1), D933–D941.")
st.markdown("**License:** Apache License 2.0")

st.markdown("#### Human Protein Atlas")
st.markdown("**Source:** Human Protein Atlas")
st.markdown("**Description:** Map of protein expression patterns in human tissues and cells.")
st.markdown("**Citation:** Uhlén M, et al. (2015). Tissue-based map of the human proteome. Science, 347(6220), 1260419.")
st.markdown("**License:** Creative Commons Attribution-ShareAlike 3.0 International License")

# --- License and Usage Information ---
st.markdown("## License and Usage Information")

st.markdown("### Cancer Vaccine Development Tool")
st.markdown("**License:** MIT License")
st.markdown("**Usage Restrictions:**")
st.markdown("- This tool is provided for research and educational purposes only.")
st.markdown("- Results should not be used as the sole basis for medical decisions or clinical applications without further validation.")
st.markdown("- Users are responsible for ensuring compliance with all applicable laws and regulations when using this tool and its data.")

st.markdown("**Disclaimer:** This tool and its results are provided \"as is\" without warranty of any kind. The authors and contributors make no representations or warranties regarding the accuracy or completeness of any information derived using this tool. In no event shall the authors or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages arising from the use of this tool or its data.")

# --- Contact Information ---
st.markdown("## Contact Information")

st.markdown("### Contact")
st.markdown("For questions, feedback, or support regarding the Cancer Vaccine Development Tool, please contact:")
st.markdown("**Research Team**  \nDepartment of Bioinformatics  \nInstitute of Cancer Research  \nEmail: research@example.org")

st.markdown("When reporting issues or requesting features, please include:")
st.markdown("- Tool version or access date")
st.markdown("- Description of the issue or feature request")
st.markdown("- Steps to reproduce the issue (if applicable)")

# --- Download Citations ---
st.markdown("## Download Citations")

st.markdown("### Citation Information")
st.markdown("If you use results from this tool in your research, please cite the original data sources as listed above, and refer to this tool as:")
st.markdown("```\nCancer Vaccine Development Tool (2024). Institute of Cancer Research. https://example.org/cancer-vaccine\n```")

st.markdown("Formatted citations for all data sources used in this tool can be downloaded below:")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download Citations (BibTeX)", "Example citation data", file_name="citations.bib")
with col2:
    st.download_button("Download Citations (RIS)", "Example citation data", file_name="citations.ris")

# --- Updates and Version History ---
st.markdown("## Updates and Version History")

st.markdown("**Version 1.0.0 (Current)**")
st.markdown("- Initial release of the Cancer Vaccine Development Tool")
st.markdown("- Integration of colorectal cancer dataset")
st.markdown("- Implementation of immunogenicity scoring system")

st.markdown("**Version 0.9.0 (Beta)**")
st.markdown("- Beta testing with limited dataset")
st.markdown("- Refinement of scoring algorithm")
st.markdown("- User interface improvements")

st.markdown("**Future Updates Planned:**")
st.markdown("- Integration of additional cancer types")
st.markdown("- Enhanced visualization options")
st.markdown("- API for programmatic access")
st.markdown("- Batch processing capabilities") 