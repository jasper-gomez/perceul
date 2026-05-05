import pandas as pd
from sklearn.decomposition import PCA 

# ============================================================
# A function to retreive the loadings of the PCA components
# ============================================================

def get_pca_loadings(pca: object, feature_names: list) -> pd.DataFrame:
    """
    Get loadings of PCA components.

    Args: 
        pca: PCA object from sklearn
        feature_names: List of feature names corresponding to the original data
    
    Returns:
        loadings: (pd.DataFrame) DataFrame with PCA loadings for each component and feature
    """
    loadings = pd.DataFrame(
        pca.components_.T,                                          # transposes the get features as rows and components as columns
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],     # names the columns as PC1, PC2, ...
        index=feature_names                                         # names the rows with the original feature names
    )

    return loadings