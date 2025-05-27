import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(page_title="scRNA-seq Analysis App", layout="wide", page_icon="🧬")

# Define the main function
def main():
    st.title("🧬 Single-Cell RNA-seq Analysis Platform")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Data Upload", "Preprocessing & Visualization", "Team Info"] # Added "Team Info"
    )

    # Initialize session state for storing data between steps
    if 'adata' not in st.session_state:
        st.session_state.adata = None

    # Render the selected page
    if page == "Home":
        home_page()
    elif page == "Data Upload":
        data_upload()
    elif page == "Preprocessing & Visualization":
        preprocessing_visualization()
    elif page == "Team Info": # Added handler for "Team Info"
        team_info_page()

def home_page():
    st.header("Welcome to the scRNA-seq Analysis Platform")

    st.markdown("""
    This application allows you to:

    1. **Upload** your single-cell RNA sequencing data (H5AD format)
    2. **Preprocess** the data and visualize with UMAP

    Get started by selecting an option from the sidebar.
    """)

    st.info("This is a simplified version focused on data upload and preprocessing.")

def data_upload():
    st.header("Upload Your H5AD Data")

    # File uploader for h5ad
    uploaded_file = st.file_uploader("Upload an h5ad file", type=["h5ad"])

    if uploaded_file is not None:
        try:
            # Load the data
            with st.spinner('Loading data...'):
                adata = sc.read_h5ad(uploaded_file)
                st.session_state.adata = adata

            st.success(f"Data loaded successfully! Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

            # Display overview
            display_data_overview(adata)

        except Exception as e:
            st.error(f"Error loading file: {e}")

def display_data_overview(adata):
    """Display overview of the dataset"""
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset Information**")
        st.write(f"Number of cells: {adata.n_obs}")
        st.write(f"Number of genes: {adata.n_vars}")

        # Display available annotations
        st.write("**Available Annotations:**")
        for col in adata.obs.columns:
            n_values = adata.obs[col].nunique()
            st.write(f"- {col}: {n_values} unique values")

    with col2:
        # Show batch distribution if available
        if 'batch' in adata.obs.columns:
            st.write("**Batch Distribution**")
            fig, ax = plt.subplots(figsize=(8, 4))
            adata.obs['batch'].value_counts().plot(kind='bar', ax=ax)
            plt.title('Cells per Batch')
            plt.ylabel('Number of Cells')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory

        # Show cell type distribution if available
        if 'celltype' in adata.obs.columns:
            st.write("**Cell Type Distribution**")
            fig, ax = plt.subplots(figsize=(8, 4))
            adata.obs['celltype'].value_counts().plot(kind='bar', ax=ax)
            plt.title('Cells per Type')
            plt.ylabel('Number of Cells')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory


def preprocessing_visualization():
    """Preprocessing and visualization function"""
    if st.session_state.adata is None:
        st.warning("Please upload data first.")
        return

    st.header("Preprocessing & UMAP Visualization")

    # Parameters
    with st.sidebar:
        st.subheader("Preprocessing Parameters")
        min_genes = st.slider("Minimum genes per cell", 0, 5000, 600)
        min_cells = st.slider("Minimum cells per gene", 0, 100, 3)
        n_neighbors = st.slider("UMAP neighbors", 5, 50, 15)
        n_pcs = st.slider("Number of PCs", 10, 100, 50)

    # Show current data info
    st.subheader("Current Data Information")
    st.write(f"Number of cells: {st.session_state.adata.n_obs}")
    st.write(f"Number of genes: {st.session_state.adata.n_vars}")

    # Preprocessing button
    if st.button("Run Preprocessing & UMAP"):
        # Get a fresh copy of the data
        adata_processed = st.session_state.adata.copy() # Use a different variable name for processed data

        with st.spinner('Processing and generating UMAP...'):
            try:
                # Filter cells
                sc.pp.filter_cells(adata_processed, min_genes=min_genes)
                st.write(f"After filtering cells: {adata_processed.n_obs} cells remaining")

                # Filter genes
                sc.pp.filter_genes(adata_processed, min_cells=min_cells)
                st.write(f"After filtering genes: {adata_processed.n_vars} genes remaining")

                # Remove mitochondrial genes
                mito_genes = [gene for gene in adata_processed.var_names if str(gene).startswith(('ERCC', 'MT-', 'mt-'))]
                if len(mito_genes) > 0:
                    adata_processed = adata_processed[:, ~adata_processed.var_names.isin(mito_genes)]
                    st.write(f"Removed {len(mito_genes)} mitochondrial genes")

                # Normalize
                sc.pp.normalize_total(adata_processed, target_sum=1e4)
                sc.pp.log1p(adata_processed)
                st.write("Normalization completed")

                # Find highly variable genes
                sc.pp.highly_variable_genes(adata_processed, min_mean=0.0125, max_mean=3, min_disp=0.5)
                n_hvg = sum(adata_processed.var.highly_variable)
                st.write(f"Found {n_hvg} highly variable genes")

                # Store raw data in .raw before filtering by highly variable genes
                adata_processed.raw = adata_processed

                # Keep only highly variable genes
                if n_hvg > 0:
                    adata_processed = adata_processed[:, adata_processed.var.highly_variable]
                else:
                    st.warning("No highly variable genes found. Skipping filtering by HVG.")


                # Scale data
                sc.pp.scale(adata_processed, max_value=10)
                st.write("Data scaling completed")

                # Run PCA
                # Ensure n_pcs is not greater than the number of features or samples minus 1
                actual_n_pcs = min(n_pcs, min(adata_processed.n_obs, adata_processed.n_vars) -1)
                if actual_n_pcs < 1: # Check if actual_n_pcs is too low
                    st.error(f"Cannot run PCA with n_comps={actual_n_pcs}. Check data dimensions after filtering.")
                    return
                sc.tl.pca(adata_processed, n_comps=actual_n_pcs)
                st.write("PCA completed")

                # Run UMAP
                # Ensure n_neighbors is not greater than the number of samples minus 1
                actual_n_neighbors = min(n_neighbors, adata_processed.n_obs -1)
                if actual_n_neighbors < 1:
                    st.error(f"Cannot run Neighbors with n_neighbors={actual_n_neighbors}. Check data dimensions after filtering.")
                    return

                sc.pp.neighbors(adata_processed, n_neighbors=actual_n_neighbors, n_pcs=actual_n_pcs)
                sc.tl.umap(adata_processed)
                st.write("UMAP completed")

                # Save the processed data
                st.session_state.adata_processed = adata_processed

                # Show results
                st.success(f"Processing complete! Final dataset: {adata_processed.n_obs} cells × {adata_processed.n_vars} genes")

            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
                st.write("Please check your data format and try again.")
                # Optionally, log the full traceback for debugging
                # import traceback
                # st.error(traceback.format_exc())
                return

    # Show UMAP visualization if processed data exists
    if 'adata_processed' in st.session_state and st.session_state.adata_processed is not None:
        st.subheader("UMAP Visualization")

        adata_to_plot = st.session_state.adata_processed # Use the processed data for plotting

        # Select color variables
        color_options = ['batch', 'celltype', 'disease', 'donor', 'protocol'] # Add more common options
        # Check if options exist in adata.obs
        valid_colors = [c for c in color_options if c in adata_to_plot.obs.columns]
        # Add any other categorical columns from obs that might be relevant
        for col in adata_to_plot.obs.select_dtypes(include=['category', 'object']).columns:
            if col not in valid_colors:
                valid_colors.append(col)

        if valid_colors:
            selected_color = st.selectbox("Color by", valid_colors)

            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.umap(adata_to_plot, color=selected_color, ax=ax, show=False, legend_fontsize=8, legend_loc='on data')
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error creating UMAP plot for '{selected_color}': {e}")
        else: # Fallback if no valid color options are found
            st.write("No suitable columns found for coloring the UMAP plot (e.g., 'batch', 'celltype'). Displaying default UMAP.")
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.umap(adata_to_plot, ax=ax, show=False)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error creating default UMAP plot: {e}")

        # Option to save the processed data
        st.subheader("Save Processed Data")
        if st.button("Download Processed Data (H5AD)"):
            try:
                # Save to a BytesIO object
                output = BytesIO()
                adata_to_plot.write_h5ad(output)
                output.seek(0)

                # Create download link
                b64 = base64.b64encode(output.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="processed_data.h5ad">Click here to download processed data</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Download link generated!")
            except Exception as e:
                st.error(f"Error saving processed data: {e}")

# New function for Team Info page
def team_info_page():
    st.header("Team Information")

    st.subheader("Team Members:")

    st.markdown("""
    - **Καλαματας Χρυσοστομος** - Inf2022062
    - **Τσιγαρας Βασίλειος** - inf2022215
    """)

    st.subheader("Contributions:")

    st.markdown("""
    **Καλαματας Χρυσοστομος (Inf2022062):**
    * Επικεντρώθηκε στην ανάπτυξη του πυρήνα της επεξεργασίας δεδομένων single-cell RNA-seq.
    * Υλοποίησε τα βήματα προεπεξεργασίας δεδομένων χρησιμοποιώντας τη βιβλιοθήκη `scanpy`, συμπεριλαμβανομένου του φιλτραρίσματος κυττάρων και γονιδίων, της κανονικοποίησης, της ανίχνευσης υψηλά μεταβλητών γονιδίων και της μείωσης διαστατικότητας (PCA).
    * Ανέλαβε τη λογική φόρτωσης των δεδομένων σε μορφή H5AD και την αποθήκευση των επεξεργασμένων δεδομένων.
    * Συνέβαλε στην αρχική δομή της εφαρμογής και στη διαχείριση της κατάστασης (session state) για τη μεταφορά δεδομένων μεταξύ των σελίδων.

    **Τσιγαρας Βασίλειος (inf2022215):**
    * Ανέλαβε τον σχεδιασμό και την υλοποίηση του περιβάλλοντος χρήστη (UI) με χρήση του Streamlit.
    * Δημιούργησε τις διάφορες σελίδες της εφαρμογής (Home, Data Upload, Preprocessing & Visualization) και την πλοήγηση μέσω της πλαϊνής μπάρας.
    * Ενσωμάτωσε τα στοιχεία ελέγχου για τις παραμέτρους προεπεξεργασίας (sliders) και τις επιλογές οπτικοποίησης.
    * Υλοποίησε την οπτικοποίηση των UMAP plots και των γραφημάτων επισκόπησης δεδομένων (π.χ., κατανομή batch, τύποι κυττάρων) χρησιμοποιώντας `matplotlib` και `seaborn`, ενσωματώνοντάς τα στο Streamlit.
    * Διαμόρφωσε την εμφάνιση των αποτελεσμάτων και των μηνυμάτων προς τον χρήστη (π.χ., μηνύματα επιτυχίας, σφάλματος).
    """)

# Run the app
if __name__ == "__main__":
    main()
