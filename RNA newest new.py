# RNA analysis pipeline
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

try:
    import mygene
    MGENE_AVAILABLE = True
except ImportError:
    MGENE_AVAILABLE = False
    print("Note: Install 'mygene' for gene descriptions: pip install mygene")

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("Note: Install 'gseapy' for pathway enrichment: pip install gseapy")

# ============ READ DATA ============
print("\n" + "=" * 60)
print("STEP 1: READING DATA")
print("=" * 60)

adata = sc.read_h5ad("C06018D5.bin50_1.0.h5ad")
print("Original data:")
print(adata)

original_gene_names = adata.var_names.copy()
print(f"First few gene names: {original_gene_names[:5]}")

if 'real_gene_name' in adata.var.columns:
    print("\nGene symbols available in var['real_gene_name']")
    print(adata.var[['real_gene_name']].head())
    missing_symbols = adata.var['real_gene_name'].isna().sum()
    print(f"Missing gene symbols: {missing_symbols}")

# ============ BASIC FILTERING ============
print("\n" + "=" * 60)
print("STEP 2: BASIC FILTERING")
print("=" * 60)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

adata = adata[adata.obs.pct_counts_mt < 10, :].copy()
print(f"\nAfter MT filtering: {adata.shape}")

# ============ NORMALIZATION ============
print("\n" + "=" * 60)
print("STEP 3: NORMALIZATION")
print("=" * 60)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ============ HIGHLY VARIABLE GENES ============
print("\n" + "=" * 60)
print("STEP 4: HIGHLY VARIABLE GENES")
print("=" * 60)

sc.pp.highly_variable_genes(
    adata, flavor='seurat', n_top_genes=8000,
    min_mean=0.0125, max_mean=3, min_disp=0.5, inplace=True
)

print("Highly variable genes column created:", 'highly_variable' in adata.var.columns)
print(f"Gene names before HVG subset: {adata.var_names[:5]}")

adata = adata[:, adata.var.highly_variable].copy()
print(f"After HVG selection: {adata.shape}")
print(f"Gene names after HVG subset: {adata.var_names[:5]}")

sc.pl.highly_variable_genes(adata)

# ============ DIMENSIONALITY REDUCTION ============
print("\n" + "=" * 60)
print("STEP 5: DIMENSIONALITY REDUCTION")
print("=" * 60)

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
sc.pl.pca_variance_ratio(adata, log=True)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata)

# ============ CLUSTERING ============
print("\n" + "=" * 60)
print("STEP 6: CLUSTERING")
print("=" * 60)

sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color="leiden")

# ============ SPATIAL VISUALIZATION ============
print("\n" + "=" * 60)
print("STEP 7: SPATIAL ANALYSIS")
print("=" * 60)

if 'spatial' in adata.obsm:
    print("Spatial coordinates found in adata.obsm['spatial']")
    spatial_coords = adata.obsm['spatial']
    print(f"Spatial coordinates shape: {spatial_coords.shape}")

    print("\n1. Creating custom spatial cluster map...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                               c=adata.obs['leiden'].astype('category').cat.codes,
                               cmap='tab10', s=5, alpha=0.7)
    axes[0].set_title('Spatial Clusters')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    axes[0].set_aspect('equal')
    legend1 = axes[0].legend(*scatter1.legend_elements(), title="Cluster",
                             bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].add_artist(legend1)

    if 'orig.ident' in adata.obs.columns:
        unique_organoids = adata.obs['orig.ident'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_organoids)))
        for i, organoid in enumerate(unique_organoids):
            mask = adata.obs['orig.ident'] == organoid
            axes[1].scatter(spatial_coords[mask, 0], spatial_coords[mask, 1],
                            c=[colors[i]], s=5, alpha=0.7, label=organoid)
        axes[1].set_title('Organoid Identity')
        axes[1].set_xlabel('X coordinate')
        axes[1].set_ylabel('Y coordinate')
        axes[1].set_aspect('equal')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('spatial_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()

    if 'orig.ident' in adata.obs.columns:
        print("\n2. Cluster composition by organoid:")
        cluster_by_organoid = pd.crosstab(adata.obs["leiden"], adata.obs["orig.ident"])
        print(cluster_by_organoid)
        cluster_by_organoid.to_csv("cluster_by_organoid.csv")

        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_by_organoid.T.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_title('Cluster Composition by Organoid')
        ax.set_xlabel('Organoid')
        ax.set_ylabel('Number of Spots')
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('cluster_composition.png', dpi=150, bbox_inches='tight')
        plt.show()

    print("\n3. Calculating spatial metrics for each cluster...")
    spatial_metrics = []
    for cluster in adata.obs['leiden'].unique():
        mask = adata.obs['leiden'] == cluster
        coords = spatial_coords[mask]
        metrics = {'cluster': cluster, 'n_spots': mask.sum()}
        if len(coords) > 3:
            try:
                hull = ConvexHull(coords)
                metrics['area_pixels'] = hull.volume
                metrics['centroid_x'] = coords[:, 0].mean()
                metrics['centroid_y'] = coords[:, 1].mean()
                metrics['spread_x'] = coords[:, 0].std()
                metrics['spread_y'] = coords[:, 1].std()
            except:
                metrics['area_pixels'] = np.nan
                metrics['centroid_x'] = coords[:, 0].mean()
                metrics['centroid_y'] = coords[:, 1].mean()
                metrics['spread_x'] = coords[:, 0].std()
                metrics['spread_y'] = coords[:, 1].std()
        else:
            metrics['area_pixels'] = np.nan
            metrics['centroid_x'] = coords[:, 0].mean() if len(coords) > 0 else np.nan
            metrics['centroid_y'] = coords[:, 1].mean() if len(coords) > 0 else np.nan
            metrics['spread_x'] = coords[:, 0].std() if len(coords) > 1 else 0
            metrics['spread_y'] = coords[:, 1].std() if len(coords) > 1 else 0
        spatial_metrics.append(metrics)
        print(f"\nCluster {cluster}:")
        print(f"  Number of spots: {metrics['n_spots']}")
        print(f"  Centroid: ({metrics['centroid_x']:.1f}, {metrics['centroid_y']:.1f})")
        if not np.isnan(metrics['area_pixels']):
            print(f"  Area: {metrics['area_pixels']:.1f} pixels²")

    spatial_metrics_df = pd.DataFrame(spatial_metrics)
    spatial_metrics_df.to_csv("spatial_metrics.csv", index=False)

    print("\n4. Plotting cluster distribution...")
    unique_clusters = adata.obs['leiden'].unique()
    n_clusters = len(unique_clusters)
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, cluster in enumerate(unique_clusters):
        mask = adata.obs['leiden'] == cluster
        coords = spatial_coords[mask]
        axes[i].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c='lightgray', s=1, alpha=0.3)
        axes[i].scatter(coords[:, 0], coords[:, 1], c='red', s=2, alpha=0.5)
        axes[i].set_title(f'Cluster {cluster} Distribution')
        axes[i].set_aspect('equal')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n5. Analyzing radial organization...")
    center = spatial_coords.mean(axis=0)
    distances = cdist([center], spatial_coords)[0]
    adata.obs['distance_from_center'] = distances
    adata.obs['spatial_zone'] = pd.cut(distances, bins=5,
                                       labels=['core', 'inner', 'mid', 'outer', 'periphery'])

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in adata.obs['leiden'].unique():
        mask = adata.obs['leiden'] == cluster
        ax.hist(adata.obs.loc[mask, 'distance_from_center'], bins=50, alpha=0.5, label=f'Cluster {cluster}')
    ax.set_xlabel('Distance from Tissue Center')
    ax.set_ylabel('Number of Spots')
    ax.set_title('Spatial Distribution by Distance from Center')
    ax.legend()
    plt.savefig('radial_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n6. Finding genes enriched in spatial zones...")
    sc.tl.rank_genes_groups(adata, 'spatial_zone', method='wilcoxon', use_raw=False)
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, title='Spatial Zone Markers')

else:
    print("No spatial coordinates found in adata.obsm['spatial']")
    print(f"Available obsm keys: {list(adata.obsm.keys())}")

# ============ MARKER GENE IDENTIFICATION ============
print("\n" + "=" * 60)
print("STEP 8: MARKER GENE ANALYSIS")
print("=" * 60)

print("\nRunning marker gene identification...")
if 'log1p' in adata.layers:
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", layer='log1p', use_raw=False)
else:
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=False)

print("\nTop marker genes per cluster:")
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names

all_markers = {}
for group in groups:
    genes = result['names'][group][:50]
    logfcs = result['logfoldchanges'][group][:50]
    scores = result['scores'][group][:50]
    all_markers[group] = {'genes': genes, 'logfcs': logfcs, 'scores': scores}
    print(f"\nCluster {group}:")
    for gene, score, logfc in zip(genes[:10], scores[:10], logfcs[:10]):
        print(f"  {gene}: score={score:.3f}, log2FC={logfc:.3f}")

marker_genes = pd.DataFrame({
    group + '_' + key: result[key][group]
    for group in groups for key in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
})
marker_genes.to_csv("marker_genes.csv")
print("\nMarker genes saved to marker_genes.csv")

sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)

print("\nCreating dotplot of top markers...")
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, title="Top Marker Genes per Cluster")

# ============ CONVERT GENE SYMBOLS ============
print("\n" + "=" * 60)
print("STEP 9: CONVERTING GENE SYMBOLS")
print("=" * 60)

if 'real_gene_name' in adata.var.columns:
    print("\nConverting ENSG IDs to gene symbols...")
    adata.var['ensg_id'] = adata.var_names.copy()
    adata.var_names = adata.var["real_gene_name"].values
    print(f"Gene names converted to symbols: {adata.var_names[:10]}")

# Re-run after symbol conversion so enrichment uses gene symbols
print("\nRe-running marker gene identification with gene symbols...")
if 'log1p' in adata.layers:
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", layer='log1p', use_raw=False)
else:
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=False)

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names

# ============ CELL TYPE MARKER ANALYSIS ============
print("\n" + "=" * 60)
print("STEP 10: CELL TYPE MARKER ANALYSIS")
print("=" * 60)

cell_type_markers = {
    'Neurons': ['SNAP25', 'SYT1', 'STMN2', 'RBFOX3', 'MAP2', 'TUBB3', 'DCX'],
    'Astrocytes': ['GFAP', 'S100B', 'AQP4', 'ALDH1L1', 'SOX9'],
    'Oligodendrocytes': ['MBP', 'PLP1', 'MOG', 'OLIG2', 'SOX10'],
    'Microglia': ['CSF1R', 'C1QB', 'PTPRC', 'CX3CR1', 'TREM2', 'AIF1'],
    'Fibroblasts': ['COL1A1', 'COL3A1', 'DCN', 'FAP', 'PDGFRA'],
    'Endothelial': ['PECAM1', 'CLDN5', 'FLT1', 'VWF', 'CDH5'],
    'Neural Progenitors': ['SOX2', 'PAX6', 'NES', 'MKI67', 'HES1'],
    'Radial Glia': ['FABP7', 'VIM', 'GLI3']
}

print("\nChecking cell type marker expression:")
cell_type_scores = {}

for cell_type, markers in cell_type_markers.items():
    available = [m for m in markers if m in adata.var_names]
    if len(available) >= 2:
        print(f"\n{cell_type}: {len(available)}/{len(markers)} markers found: {available}")
        try:
            sc.pl.matrixplot(adata, available, groupby='leiden', title=f'{cell_type} Markers',
                             cmap='Blues', standard_scale='var',
                             save=f'_{cell_type}_markers.png', show=False, use_raw=False)
        except:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                sc.pl.dotplot(adata, available, groupby='leiden',
                              title=f'{cell_type} Markers', ax=ax, show=False, use_raw=False)
                plt.savefig(f'dotplot_{cell_type}_markers.png', dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Could not create plot for {cell_type}: {e}")

        for cluster in groups:
            mask = adata.obs['leiden'] == cluster
            cluster_exp = []
            for marker in available:
                if marker in adata.var_names:
                    marker_idx = list(adata.var_names).index(marker)
                    exp_values = adata[mask, marker_idx].X
                    if hasattr(exp_values, 'toarray'):
                        exp_values = exp_values.toarray().flatten()
                    cluster_exp.append(exp_values.mean())
            if cluster_exp:
                mean_exp = np.mean(cluster_exp)
                if cell_type not in cell_type_scores:
                    cell_type_scores[cell_type] = {}
                cell_type_scores[cell_type][cluster] = mean_exp

if cell_type_scores:
    cell_type_df = pd.DataFrame(cell_type_scores).round(3)
    cell_type_df.to_csv('cell_type_scores.csv')
    print("\nCell type scores saved to cell_type_scores.csv")
    print("\nCell type scores by cluster:")
    print(cell_type_df)
else:
    print("\nNo cell type scores could be calculated.")

# ============ PATHWAY ENRICHMENT ============
print("\n" + "=" * 60)
print("STEP 11: PATHWAY ENRICHMENT ANALYSIS")
print("=" * 60)

if GSEAPY_AVAILABLE:
    for cluster in groups:
        print(f"\nRunning enrichment for Cluster {cluster}...")
        genes = result['names'][cluster][:100]
        try:
            enr = gp.enrichr(gene_list=genes.tolist(),
                             gene_sets=['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022'],
                             organism='human', outdir=f'enrichment_cluster_{cluster}')
            print(f"\nCluster {cluster} top pathways:")
            print(enr.results.head(10)[['Term', 'Adjusted P-value']])
            enr.results.to_csv(f'enrichment_cluster_{cluster}.csv', index=False)
        except Exception as e:
            print(f"Enrichment failed for cluster {cluster}: {e}")
else:
    print("Skipping pathway enrichment - install gseapy: pip install gseapy")

# ============ ADDING GENE DESCRIPTIONS ============
print("\n" + "=" * 60)
print("STEP 12: ADDING GENE DESCRIPTIONS")
print("=" * 60)

if MGENE_AVAILABLE:
    print("\nFetching gene descriptions for top markers...")
    all_top_genes = set()
    for cluster in groups:
        cluster_genes = result['names'][cluster][:20]
        all_top_genes.update(cluster_genes)

    all_top_genes = [g for g in all_top_genes if not pd.isna(g) and g != '']
    print(f"Querying {len(all_top_genes)} gene symbols...")

    try:
        mg = mygene.MyGeneInfo()
        gene_info = mg.querymany(all_top_genes, scopes='symbol',
                                 fields='name,summary,entrezgene', species='human')
        gene_info = [g for g in gene_info if 'notfound' not in g]

        if gene_info:
            gene_info_df = pd.DataFrame(gene_info)
            available_cols = ['query']
            for col in ['name', 'summary', 'entrezgene']:
                if col in gene_info_df.columns:
                    available_cols.append(col)
            gene_info_df = gene_info_df[available_cols]
            gene_info_df.columns = ['gene'] + [c for c in available_cols[1:]]

            gene_cluster = []
            for gene in gene_info_df['gene']:
                found = False
                for cluster in groups:
                    cluster_genes = result['names'][cluster][:20]
                    if gene in cluster_genes:
                        gene_cluster.append(cluster)
                        found = True
                        break
                if not found:
                    gene_cluster.append('unknown')

            gene_info_df['top_marker_in_cluster'] = gene_cluster
            gene_info_df.to_csv('gene_descriptions.csv', index=False)
            print(f"Gene descriptions saved to gene_descriptions.csv")
            print(gene_info_df.head())
        else:
            print("No gene descriptions found")
    except Exception as e:
        print(f"Error fetching gene descriptions: {e}")
else:
    print("Skipping gene descriptions - install mygene: pip install mygene")

# ============ CREATE PUBLICATION FIGURES ============
print("\n" + "=" * 60)
print("STEP 13: CREATING PUBLICATION FIGURES")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))

ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                      c=adata.obs['leiden'].astype('category').cat.codes,
                      cmap='tab10', s=3, alpha=0.7)
ax1.set_title('A: Spatial Clusters', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
sc.pl.umap(adata, color='leiden', ax=ax2, show=False, title='B: UMAP Clusters')
ax2.set_title('B: UMAP Clusters', fontsize=14, fontweight='bold')

ax3 = plt.subplot(2, 3, 3)
# rank_genes_groups_dotplot does not support ax= embedding — draw manually
_n_top = 3
_bar_genes, _bar_scores, _bar_labels = [], [], []
_colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(groups)))
for _ci, _group in enumerate(groups):
    for _rank in range(_n_top):
        _bar_genes.append(result['names'][_group][_rank])
        _bar_scores.append(result['scores'][_group][_rank])
        _bar_labels.append(f"C{_group}")
_y_pos = np.arange(len(_bar_genes))
_bar_colors = [_colors_cycle[list(groups).index(_lbl[1:])] for _lbl in _bar_labels]
ax3.barh(_y_pos, _bar_scores, color=_bar_colors, alpha=0.8)
ax3.set_yticks(_y_pos)
ax3.set_yticklabels(_bar_genes, fontsize=8)
ax3.set_xlabel('Wilcoxon score')
ax3.set_title('C: Top Marker Genes', fontsize=14, fontweight='bold')
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color=_colors_cycle[i], label=f'Cluster {g}') for i, g in enumerate(groups)],
           fontsize=7, loc='lower right')

ax4 = plt.subplot(2, 3, 4)
representative_markers = [result['names'][cluster][0] for cluster in groups[:3]]
if len(representative_markers) >= 3:
    colors_rgb = ['red', 'green', 'blue']
    rgb_image = np.zeros((spatial_coords.shape[0], 3))
    for i, (marker, color) in enumerate(zip(representative_markers[:3], colors_rgb)):
        if marker in adata.var_names:
            marker_idx = list(adata.var_names).index(marker)
            exp = adata.X[:, marker_idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            exp = (exp - exp.min()) / (exp.max() - exp.min() + 1e-8)
            if color == 'red':
                rgb_image[:, 0] = exp
            elif color == 'green':
                rgb_image[:, 1] = exp
            elif color == 'blue':
                rgb_image[:, 2] = exp
    ax4.scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=rgb_image, s=3, alpha=0.7)
    ax4.set_title('D: Marker Overlay', fontsize=14, fontweight='bold')
    ax4.set_aspect('equal')
    ax4.axis('off')

ax5 = plt.subplot(2, 3, 5)
if cell_type_scores:
    cell_type_df.T.plot(kind='bar', ax=ax5, legend=False)
    ax5.set_title('E: Cell Type Scores', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Cell Type')
    ax5.set_ylabel('Mean Expression')

ax6 = plt.subplot(2, 3, 6)
if 'cluster_by_organoid' in locals():
    cluster_by_organoid.T.plot(kind='bar', stacked=True, ax=ax6, colormap='tab10')
    ax6.set_title('F: Cluster Composition', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Organoid')
    ax6.set_ylabel('Number of Spots')
    ax6.legend(title='Cluster', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Publication figure saved to publication_figure.png")

# ============ SPATIAL EXPRESSION OF KEY MARKERS ============
print("\n" + "=" * 60)
print("STEP 14: SPATIAL EXPRESSION OF KEY MARKERS")
print("=" * 60)

neural_markers = ["SOX2", "PAX6", "MAP2", "TUBB3", "MKI67", "GFAP",
                  "NES", "DCX", "S100B", "OLIG2", "EMX1", "DLX2",
                  "CSF1R", "C1QB", "COL1A1", "COL3A1", "DCN", "SNAP25", "SYT1"]

available_markers = [m for m in neural_markers if m in adata.var_names]
print(f"\nAvailable neural markers: {available_markers}")

if available_markers and 'spatial' in adata.obsm:
    print("\nPlotting spatial expression of neural markers...")
    spatial_coords = adata.obsm['spatial']
    n_markers = len(available_markers)

    if n_markers == 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        axes = [ax]
    else:
        n_cols = 3
        n_rows = (n_markers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

    for i, marker in enumerate(available_markers):
        marker_idx = list(adata.var_names).index(marker)
        marker_exp = adata.X[:, marker_idx]
        if hasattr(marker_exp, 'toarray'):
            marker_exp = marker_exp.toarray().flatten()
        marker_exp_norm = (marker_exp - marker_exp.min()) / (marker_exp.max() - marker_exp.min() + 1e-8)
        scatter = axes[i].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                                  c=marker_exp_norm, cmap='viridis', s=5, alpha=0.7, vmin=0, vmax=1)
        axes[i].set_title(f'{marker} Expression')
        axes[i].set_aspect('equal')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(scatter, ax=axes[i])

    if n_markers > 1:
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('spatial_marker_expression.png', dpi=150, bbox_inches='tight')
    plt.show()

    if 'COL1A1' in available_markers:
        print("\nCreating detailed COL1A1 expression plot...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        marker_idx = list(adata.var_names).index('COL1A1')
        marker_exp = adata.X[:, marker_idx]
        if hasattr(marker_exp, 'toarray'):
            marker_exp = marker_exp.toarray().flatten()

        scatter = ax1.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                              c=marker_exp, cmap='Reds', s=5, alpha=0.7)
        ax1.set_title('COL1A1 Expression (Fibroblast Marker)')
        ax1.set_aspect('equal')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(scatter, ax=ax1)

        scatter2 = ax2.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                               c=adata.obs['leiden'].astype('category').cat.codes,
                               cmap='tab10', s=3, alpha=0.5)
        ax2.set_title('Clusters with COL1A1+ regions')
        ax2.set_aspect('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])

        high_exp_threshold = np.percentile(marker_exp, 90)
        high_exp_mask = marker_exp > high_exp_threshold
        ax2.scatter(spatial_coords[high_exp_mask, 0], spatial_coords[high_exp_mask, 1],
                    c='red', s=10, alpha=0.8, label='COL1A1 high (>90th %ile)')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('COL1A1_detailed.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\nCOL1A1 Expression Statistics:")
        print(f"  Mean expression: {marker_exp.mean():.4f}")
        print(f"  Max expression: {marker_exp.max():.4f}")
        print(f"  % spots expressing: {(marker_exp > 0).mean() * 100:.1f}%")

        print("\n  COL1A1 expression by cluster:")
        for cluster in groups:
            mask = adata.obs['leiden'] == cluster
            cluster_exp = marker_exp[mask]
            print(f"    Cluster {cluster}: mean={cluster_exp.mean():.4f}, "
                  f"% expressing={(cluster_exp > 0).mean() * 100:.1f}%")

# ============ ADDITIONAL VISUALIZATIONS ============
print("\n" + "=" * 60)
print("STEP 15: ADDITIONAL VISUALIZATIONS")
print("=" * 60)

print("\nCreating heatmap of top markers...")
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, use_raw=False, show=False)

print("\nCreating violin plots for top markers...")
top_genes_per_cluster = [result['names'][group][0] for group in groups[:3]]
sc.pl.violin(adata, keys=top_genes_per_cluster, groupby='leiden', rotation=45, use_raw=False)

# ============ SAVE PROCESSED DATA ============
print("\n" + "=" * 60)
print("STEP 16: SAVING RESULTS")
print("=" * 60)

print("\nConverting string columns for saving...")
for col in adata.obs.columns:
    if hasattr(adata.obs[col], 'dtype') and pd.api.types.is_string_dtype(adata.obs[col]):
        adata.obs[col] = adata.obs[col].astype('object')
        print(f"  Converted obs.{col}")

for col in adata.var.columns:
    if hasattr(adata.var[col], 'dtype') and pd.api.types.is_string_dtype(adata.var[col]):
        adata.var[col] = adata.var[col].astype('object')
        print(f"  Converted var.{col}")

if hasattr(adata.obs.index, 'dtype') and pd.api.types.is_string_dtype(adata.obs.index):
    adata.obs.index = adata.obs.index.astype('object')
    print("  Converted obs.index")

if hasattr(adata.var.index, 'dtype') and pd.api.types.is_string_dtype(adata.var.index):
    adata.var.index = adata.var.index.astype('object')
    print("  Converted var.index")

for key in list(adata.uns.keys()):
    if isinstance(adata.uns[key], pd.DataFrame):
        print(f"  Checking uns['{key}'] DataFrame")
        for col in adata.uns[key].columns:
            if hasattr(adata.uns[key][col], 'dtype') and pd.api.types.is_string_dtype(adata.uns[key][col]):
                adata.uns[key][col] = adata.uns[key][col].astype('object')
                print(f"    Converted uns['{key}'].{col}")

print("\nSaving data...")
adata.write("processed_spatial_data.h5ad")
print("Analysis complete and data saved to 'processed_spatial_data.h5ad'!")

print("\nVerifying saved data...")
test_adata = sc.read_h5ad("processed_spatial_data.h5ad", backed='r')
print(f"Gene names in saved file: {test_adata.var_names[:10]}")
test_adata.file.close()

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Final dataset: {adata.shape[0]} spots × {adata.shape[1]} genes")
print(f"Number of clusters: {len(groups)}")
print(f"Spatial coordinates: {'✓' if 'spatial' in adata.obsm else '✗'}")
print(f"Organoid identities: {'✓' if 'orig.ident' in adata.obs.columns else '✗'}")
print(f"Gene symbols converted: {'✓' if 'real_gene_name' in adata.var.columns else '✗'}")

print("\n" + "=" * 60)
print("FILES GENERATED")
print("=" * 60)
print("  - processed_spatial_data.h5ad")
print("  - marker_genes.csv")
if MGENE_AVAILABLE:
    print("  - gene_descriptions.csv")
print("  - cell_type_scores.csv")
if GSEAPY_AVAILABLE:
    print("  - enrichment_cluster_*.csv")

# ============ STEP 17: SPATIAL DOMAIN DETECTION WITH SpaGCN ============
print("\n" + "=" * 60)
print("STEP 17: SPATIAL DOMAIN DETECTION WITH SpaGCN")
print("=" * 60)

try:
    import SpaGCN as spg

    print("\nPreparing data for SpaGCN...")
    x_array = adata.obsm['spatial'][:, 0]
    y_array = adata.obsm['spatial'][:, 1]

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    print("\nCalculating spatial graph...")
    adj = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)

    # FIX: search_l takes probability p and adj matrix only
    print("Searching for optimal l parameter...")
    l = spg.search_l(p=0.5, adj=adj, start=0.01, end=1000, tol=0.01)
    print(f"Optimal l = {l}")

    # FIX: removed invalid 'seed' keyword argument
    print("Searching for optimal resolution...")
    res = spg.search_res(adata, adj, l, target_num=5, start=0.1, step=0.1,
                         tol=5e-3, lr=0.05, max_epochs=20)
    print(f"Optimal resolution = {res}")

    print(f"Running SpaGCN with l={l}, res={res}...")
    clf = spg.SpaGCN()
    clf.set_library_size(library_size=adata.obs['total_counts'].values)
    clf.train(X, adj, init_spa=True, init=None, res=res, l=l)

    y_pred = clf.predict()
    adata.obs['spatial_domain'] = y_pred.astype(str)
    print(f"Detected {len(np.unique(y_pred))} spatial domains")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(x_array, y_array,
                    c=adata.obs['spatial_domain'].astype('category').cat.codes,
                    cmap='tab10', s=5, alpha=0.7)
    axes[0].set_title('SpaGCN Spatial Domains')
    axes[0].set_aspect('equal')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(x_array, y_array,
                    c=adata.obs['leiden'].astype('category').cat.codes,
                    cmap='tab10', s=5, alpha=0.7)
    axes[1].set_title('Leiden Clusters (Transcriptomic)')
    axes[1].set_aspect('equal')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig('spatial_domains_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    cross_tab = pd.crosstab(adata.obs['leiden'], adata.obs['spatial_domain'])
    print(cross_tab)
    cross_tab.to_csv('leiden_vs_spatial_domains.csv')

    sc.tl.rank_genes_groups(adata, 'spatial_domain', method='wilcoxon', use_raw=False)
    sc.pl.rank_genes_groups_dotplot(adata, groupby='spatial_domain',
                                    n_genes=5, title='Spatial Domain Markers')

    result_spatial = adata.uns['rank_genes_groups']
    spatial_markers = pd.DataFrame({
        group + '_' + key: result_spatial[key][group]
        for group in result_spatial['names'].dtype.names
        for key in ['names', 'scores', 'pvals_adj', 'logfoldchanges']
    })
    spatial_markers.to_csv('spatial_domain_markers.csv')

except ImportError:
    print("SpaGCN not installed. Install with: pip install SpaGCN")
except Exception as e:
    print(f"Error in SpaGCN: {e}")
    import traceback
    traceback.print_exc()

# ============ STEP 18: SPATIALLY VARIABLE GENE DETECTION ============
print("\n" + "=" * 60)
print("STEP 18: SPATIALLY VARIABLE GENE DETECTION")
print("=" * 60)

try:
    import squidpy as sq

    print("\nCalculating spatial autocorrelation (Moran's I)...")
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)
    sq.gr.spatial_autocorr(adata, mode='moran', genes=adata.var_names[:1000], n_perms=100)

    moran_results = adata.uns['moranI']
    print(f"\nAvailable columns: {moran_results.columns.tolist()}")

    # FIX: always assign gene_col in every branch
    if 'gene' in moran_results.columns:
        gene_col = 'gene'
    elif 'genes' in moran_results.columns:
        gene_col = 'genes'
    else:
        print("Using index for gene names")
        moran_results = moran_results.copy()
        moran_results['gene'] = moran_results.index
        gene_col = 'gene'

    top_spatial_genes = moran_results.nsmallest(20, 'pval_norm')[gene_col].tolist()
    print("\nTop 20 spatially variable genes:")
    print(moran_results.nsmallest(20, 'pval_norm')[['I', 'pval_norm', gene_col]])
    moran_results.to_csv('spatially_variable_genes.csv')

    print("\nPlotting top spatially variable genes...")
    spatial_coords = adata.obsm['spatial']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, gene in enumerate(top_spatial_genes[:6]):
        if gene in adata.var_names:
            gene_idx = list(adata.var_names).index(gene)
            exp = adata.X[:, gene_idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            moran_val = moran_results[moran_results[gene_col] == gene]['I'].values[0]
            scatter = axes[i].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                                      c=exp, cmap='viridis', s=3, alpha=0.7)
            axes[i].set_title(f"{gene}\nMoran's I={moran_val:.3f}")
            axes[i].set_aspect('equal')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            plt.colorbar(scatter, ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, f'{gene}\nnot in dataset',
                         ha='center', va='center', transform=axes[i].transAxes)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('spatially_variable_genes.png', dpi=150, bbox_inches='tight')
    plt.show()

    top10 = moran_results.nsmallest(10, 'pval_norm')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top10)), top10['I'].values)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10[gene_col].values)
    ax.set_xlabel("Moran's I (spatial autocorrelation)")
    ax.set_title('Top 10 Spatially Variable Genes')
    for i, (bar, pval) in enumerate(zip(bars, top10['pval_norm'].values)):
        bar.set_color('darkred' if pval < 0.001 else 'red' if pval < 0.01 else 'salmon' if pval < 0.05 else 'gray')
    plt.tight_layout()
    plt.savefig('top_spatial_genes_barplot.png', dpi=150, bbox_inches='tight')
    plt.show()

except ImportError:
    print("Squidpy not installed. Install with: pip install squidpy")
except Exception as e:
    print(f"Error in spatial analysis: {e}")
    import traceback
    traceback.print_exc()

# ============ STEP 19: HOTSPOT GENE MODULES ============
print("\n" + "=" * 60)
print("STEP 19: HOTSPOT GENE MODULE DETECTION")
print("=" * 60)

try:
    import hotspot

    print("\nRunning Hotspot analysis...")
    spatial_coords = adata.obsm['spatial']
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    hs = hotspot.Hotspot(adata, layer='X', model='danb', latent_obsm='X_pca',
                         umi_counts=adata.obs['total_counts'])
    hs.create_knn_graph(latent_obsm='spatial', n_neighbors=30)
    hs.compute_autocorrelations(jobs=4)

    significant_genes = hs.results[hs.results.FDR < 0.05].index.tolist()
    print(f"\nFound {len(significant_genes)} spatially autocorrelated genes")

    hs.create_modules(min_gene_threshold=5, core_only=True, fdr_threshold=0.05)
    module_df = hs.modules
    print("\nGene modules detected:")
    print(module_df['Module'].value_counts())
    module_df.to_csv('hotspot_modules.csv')

    module_scores = hs.module_scores()
    n_modules = module_df['Module'].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(min(n_modules, 6)):
        module_name = f'module_{i}'
        if module_name in module_scores.columns:
            scatter = axes[i].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                                      c=module_scores[module_name], cmap='RdBu_r', s=3, alpha=0.7)
            axes[i].set_title(f'Spatial Module {i}')
            axes[i].set_aspect('equal')
            plt.colorbar(scatter, ax=axes[i])
    plt.tight_layout()
    plt.savefig('hotspot_modules.png', dpi=150, bbox_inches='tight')
    plt.show()

except ImportError:
    print("Hotspot not installed. Install with: pip install hotspot")

# ============ STEP 20: SPATIAL DOMAIN REPRODUCIBILITY ============
print("\n" + "=" * 60)
print("STEP 20: SPATIAL DOMAIN REPRODUCIBILITY")
print("=" * 60)

if 'spatial_domain' in adata.obs.columns and 'orig.ident' in adata.obs.columns:
    print("\nAnalyzing spatial domain reproducibility across organoids...")
    domain_by_organoid = pd.crosstab(adata.obs['spatial_domain'], adata.obs['orig.ident'])
    print(domain_by_organoid)
    domain_by_organoid.to_csv('spatial_domains_by_organoid.csv')

    domain_proportions = domain_by_organoid.div(domain_by_organoid.sum(axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    domain_by_organoid.T.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10')
    axes[0].set_title('Spatial Domain Composition (Absolute)')
    axes[0].set_xlabel('Organoid')
    axes[0].set_ylabel('Number of Spots')
    axes[0].legend(title='Domain')
    domain_proportions.T.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10')
    axes[1].set_title('Spatial Domain Composition (Proportions)')
    axes[1].set_xlabel('Organoid')
    axes[1].set_ylabel('Proportion')
    axes[1].legend(title='Domain')
    plt.tight_layout()
    plt.savefig('spatial_domain_reproducibility.png', dpi=150, bbox_inches='tight')
    plt.show()

    from scipy.stats import chi2_contingency
    chi2_val, p_val, dof, expected = chi2_contingency(domain_by_organoid)
    print(f"\nChi-square test: p-value = {p_val:.4f}")

# ============ STEP 21: SPATIAL DOMAIN MARKERS ============
print("\n" + "=" * 60)
print("STEP 21: SPATIAL DOMAIN MARKER GENES")
print("=" * 60)

if 'spatial_domain' in adata.obs.columns:
    sc.tl.rank_genes_groups(adata, 'spatial_domain', method='wilcoxon', use_raw=False)
    result_spatial = adata.uns['rank_genes_groups']
    spatial_groups = result_spatial['names'].dtype.names
    for group in spatial_groups:
        print(f"\nSpatial Domain {group}:")
        for gene, logfc in zip(result_spatial['names'][group][:10], result_spatial['logfoldchanges'][group][:10]):
            print(f"  {gene}: log2FC={logfc:.3f}")
    spatial_markers = pd.DataFrame({
        group + '_' + key: result_spatial[key][group]
        for group in spatial_groups
        for key in ['names', 'scores', 'pvals_adj', 'logfoldchanges']
    })
    spatial_markers.to_csv('spatial_domain_markers.csv')
    sc.pl.rank_genes_groups_dotplot(adata, groupby='spatial_domain',
                                    n_genes=5, title='Spatial Domain Markers')

# ============ STEP 22: NEURONAL GENE ANALYSIS ============
print("\n" + "=" * 60)
print("STEP 22: NEURONAL GENE ANALYSIS FOR CORTICOSPINAL MOTOR ORGANOIDS")
print("=" * 60)

print("\nChecking current gene names:")
print(f"First 10 gene names: {adata.var_names[:10].tolist()}")

corticospinal_markers = {
    'Cortical Layer Markers': {
        'Upper Layer (II-IV)': ['CUX1', 'CUX2', 'SATB2', 'BCL11B'],
        'Deep Layer (V-VI)': ['BCL11B', 'TBR1', 'SOX5', 'FEZF2', 'CTIP2'],
        'Layer V (Corticospinal)': ['BCL11B', 'FEZF2', 'CRYM', 'ETV1'],
        'Layer VI': ['TBR1', 'FOXP2', 'CTGF']
    },
    'Motor Neuron Markers': {
        'General Motor Neurons': ['ISL1', 'MNX1', 'CHAT', 'SLC5A7', 'SLC18A3'],
        'Spinal Motor Neurons': ['HOXC4', 'HOXC5', 'HOXA5', 'HOXB5'],
        'Motor Progenitors': ['OLIG2', 'NKX6-1', 'PAX6', 'SOX2']
    },
    'Neuronal Subtype Markers': {
        'Glutamatergic': ['SLC17A7', 'SLC17A6', 'GRIN1', 'GRIA2'],
        'GABAergic': ['GAD1', 'GAD2', 'SLC32A1', 'DLX1', 'DLX2'],
        'Cholinergic': ['CHAT', 'ACHE', 'SLC5A7']
    },
    'Synaptic Markers': {
        'Presynaptic': ['SYP', 'SNAP25', 'STX1A', 'VAMP2'],
        'Postsynaptic': ['DLG4', 'GRIN1', 'GRIA2', 'HOMER1']
    }
}

print("\n1. CHECKING NEURONAL MARKER AVAILABILITY:")
print("-" * 40)

all_found_markers = []
for category, subcategories in corticospinal_markers.items():
    print(f"\n{category}:")
    for subcat, markers in subcategories.items():
        found = [m for m in markers if m in adata.var_names]
        for m in found:
            if m not in all_found_markers:
                all_found_markers.append(m)
        print(f"  {subcat}: {len(found)}/{len(markers)} found - {found if found else 'None'}")

print(f"\nTotal unique neuronal markers found: {len(all_found_markers)}")
print(f"Markers: {all_found_markers}")

print("\n2. TESTING HIGHER RESOLUTION CLUSTERING:")
print("-" * 40)

resolutions = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res{res}')
    n_clusters = len(adata.obs[f'leiden_res{res}'].unique())
    print(f"Resolution {res}: {n_clusters} clusters")

optimal_res = 1.2
print(f"\nUsing resolution {optimal_res} for detailed neuronal analysis")

sc.tl.leiden(adata, resolution=optimal_res, key_added='leiden_neuronal')
sc.tl.rank_genes_groups(adata, 'leiden_neuronal', method='wilcoxon', use_raw=False)

result_neuronal = adata.uns['rank_genes_groups']
neuronal_groups = result_neuronal['names'].dtype.names
print(f"\nIdentified {len(neuronal_groups)} clusters at resolution {optimal_res}")

# ============ DYNAMIC CLUSTER IDENTIFICATION ============
print("\n" + "=" * 60)
print("DYNAMIC CLUSTER IDENTIFICATION")
print("=" * 60)

motor_markers_list = ['SLC5A7', 'CHAT', 'ISL1', 'MNX1']
available_motor = [m for m in motor_markers_list if m in adata.var_names]

motor_scores = {}
if available_motor:
    for marker in available_motor:
        marker_idx = list(adata.var_names).index(marker)
        exp = adata.X[:, marker_idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        for group in neuronal_groups:
            mask = adata.obs['leiden_neuronal'] == group
            if mask.sum() > 0:
                if group not in motor_scores:
                    motor_scores[group] = []
                motor_scores[group].append(exp[mask].mean())

    motor_df = pd.DataFrame([{'cluster': k, 'motor_score': np.mean(v)}
                              for k, v in motor_scores.items() if v])
    motor_df = motor_df.sort_values('motor_score', ascending=False)
    top_motor_cluster = str(motor_df.iloc[0]['cluster'])
    print(f"\n✓ Motor neuron cluster identified: {top_motor_cluster}")
    print(f"  Motor score: {motor_df.iloc[0]['motor_score']:.3f}")
    print("\nTop 10 motor neuron clusters:")
    print(motor_df.head(10))
    motor_df.to_csv('motor_neuron_clusters.csv', index=False)
else:
    top_motor_cluster = '4'
    print(f"No motor markers found, defaulting to cluster {top_motor_cluster}")

fibroblast_markers_list = ['COL1A1', 'COL3A1', 'DCN', 'COL1A2', 'FN1', 'MGP']
available_fibro = [m for m in fibroblast_markers_list if m in adata.var_names]

fibro_scores = {}
if available_fibro:
    for marker in available_fibro:
        marker_idx = list(adata.var_names).index(marker)
        exp = adata.X[:, marker_idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        for group in neuronal_groups:
            mask = adata.obs['leiden_neuronal'] == group
            if mask.sum() > 0:
                if group not in fibro_scores:
                    fibro_scores[group] = []
                fibro_scores[group].append(exp[mask].mean())

    fibro_df = pd.DataFrame([{'cluster': k, 'fibroblast_score': np.mean(v)}
                              for k, v in fibro_scores.items() if v])
    fibro_df = fibro_df.sort_values('fibroblast_score', ascending=False)
    top_fibro_cluster = str(fibro_df.iloc[0]['cluster'])
    print(f"\n✓ Fibroblast cluster identified: {top_fibro_cluster}")
    print(f"  Fibroblast score: {fibro_df.iloc[0]['fibroblast_score']:.3f}")
    print("\nTop 10 fibroblast clusters:")
    print(fibro_df.head(10))
    fibro_df.to_csv('fibroblast_clusters.csv', index=False)
else:
    top_fibro_cluster = '8'
    print(f"No fibroblast markers found, defaulting to cluster {top_fibro_cluster}")

print("\n3. TOP MARKERS FOR EACH CLUSTER:")
print("-" * 40)
for group in neuronal_groups[:10]:
    genes = result_neuronal['names'][group][:10]
    logfcs = result_neuronal['logfoldchanges'][group][:10]
    print(f"\nCluster {group} top markers:")
    if group == top_motor_cluster:
        print("  ★ MOTOR NEURON CLUSTER ★")
    elif group == top_fibro_cluster:
        print("  ◆ FIBROBLAST CLUSTER ◆")
    for i, gene in enumerate(genes[:5]):
        logfc = logfcs[i] if i < len(logfcs) else np.nan
        tag = " (FIBROBLAST MARKER)" if gene in fibroblast_markers_list else ""
        print(f"  {gene}: log2FC={logfc:.3f}{tag}")

with open('cluster_identities.txt', 'w') as f:
    f.write(f"Motor neuron cluster: {top_motor_cluster}\n")
    f.write(f"Fibroblast cluster: {top_fibro_cluster}\n")
    f.write(f"Date: {pd.Timestamp.now()}\n")

print(f"\n✅ Dynamic cluster identification complete!")
print(f"  Motor cluster: {top_motor_cluster}")
print(f"  Fibroblast cluster: {top_fibro_cluster}")

# ============ QUANTITATIVE SPATIAL ANALYSIS ============
print("\n" + "=" * 60)
print("QUANTITATIVE SPATIAL ANALYSIS OF MOTOR NEURONS")
print("=" * 60)

spatial_coords = adata.obsm['spatial']
motor_mask = adata.obs['leiden_neuronal'] == top_motor_cluster
fibro_mask = adata.obs['leiden_neuronal'] == top_fibro_cluster
motor_coords = spatial_coords[motor_mask]
fibro_coords = spatial_coords[fibro_mask]

print(f"\nMotor neuron cluster {top_motor_cluster} statistics:")
print(f"  Number of spots: {motor_mask.sum()}")
print(f"  Percentage of total: {motor_mask.sum() / len(motor_mask) * 100:.1f}%")

if len(motor_coords) > 0:
    center = motor_coords.mean(axis=0)
    print(f"  Center position: ({center[0]:.0f}, {center[1]:.0f})")
    if len(motor_coords) > 3:
        hull = ConvexHull(motor_coords)
        print(f"  Spatial area: {hull.volume:.0f} pixels²")
    if len(motor_coords) > 1:
        from scipy.spatial.distance import pdist
        pairwise_distances = pdist(motor_coords)
        print(f"  Mean distance between spots: {np.mean(pairwise_distances):.0f} pixels")

print(f"\nFibroblast cluster {top_fibro_cluster} statistics:")
print(f"  Number of spots: {fibro_mask.sum()}")

if len(motor_coords) > 0 and len(fibro_coords) > 0:
    motor_center = motor_coords.mean(axis=0)
    fibro_center = fibro_coords.mean(axis=0)
    distance = np.linalg.norm(motor_center - fibro_center)
    print(f"\nDistance between cluster centers: {distance:.0f} pixels")
    motor_min, motor_max = motor_coords.min(axis=0), motor_coords.max(axis=0)
    fibro_min, fibro_max = fibro_coords.min(axis=0), fibro_coords.max(axis=0)
    overlap_x = not (motor_max[0] < fibro_min[0] or motor_min[0] > fibro_max[0])
    overlap_y = not (motor_max[1] < fibro_min[1] or motor_min[1] > fibro_max[1])
    if overlap_x and overlap_y:
        print("  ✓ Motor neurons and fibroblasts OVERLAP spatially")
    else:
        print("  ✗ Motor neurons and fibroblasts are SEPARATE")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c='lightgray', s=3, alpha=0.3)
    axes[0].scatter(motor_coords[:, 0], motor_coords[:, 1], c='red', s=10, alpha=0.8)
    axes[0].set_title(f'Motor Neuron Cluster {top_motor_cluster}\n({motor_mask.sum()} spots)')
    axes[0].set_aspect('equal')
    axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c='lightgray', s=3, alpha=0.3)
    axes[1].scatter(fibro_coords[:, 0], fibro_coords[:, 1], c='blue', s=10, alpha=0.8)
    axes[1].set_title(f'Fibroblast Cluster {top_fibro_cluster}\n({fibro_mask.sum()} spots)')
    axes[1].set_aspect('equal')
    axes[2].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c='lightgray', s=3, alpha=0.3)
    axes[2].scatter(motor_coords[:, 0], motor_coords[:, 1], c='red', s=10, alpha=0.8, label='Motor neurons')
    axes[2].scatter(fibro_coords[:, 0], fibro_coords[:, 1], c='blue', s=10, alpha=0.8, label='Fibroblasts')
    axes[2].set_title('Motor Neurons vs Fibroblasts')
    axes[2].set_aspect('equal')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('motor_vs_fibroblast_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

if 'SLC5A7' in adata.var_names:
    idx = list(adata.var_names).index('SLC5A7')
    exp = adata.X[:, idx]
    if hasattr(exp, 'toarray'):
        exp = exp.toarray().flatten()
    motor_exp = exp[motor_mask]
    print(f"\nSLC5A7 expression in motor cluster {top_motor_cluster}:")
    print(f"  Mean: {motor_exp.mean():.4f}")
    print(f"  % expressing: {(motor_exp > 0).mean() * 100:.1f}%")

print(f"\nTop marker genes for motor neuron cluster {top_motor_cluster}:")
motor_genes = []
for i, gene in enumerate(result_neuronal['names'][top_motor_cluster][:20]):
    score = result_neuronal['scores'][top_motor_cluster][i]
    logfc = result_neuronal['logfoldchanges'][top_motor_cluster][i]
    pval = result_neuronal['pvals'][top_motor_cluster][i]
    motor_genes.append({'gene': gene, 'score': score, 'log2FC': logfc, 'p_value': pval})
    print(f"  {gene}: score={score:.3f}")

motor_df_cluster = pd.DataFrame(motor_genes)
motor_df_cluster.to_csv(f'motor_neuron_cluster_{top_motor_cluster}_markers.csv', index=False)

print("\n✅ Analysis complete!")
print("=" * 60)

print("\nLayer marker expression in motor neuron cluster:")
layer_markers = {
    'Upper Layer (CUX1, CUX2)': ['CUX1', 'CUX2'],
    'Layer V/VI (BCL11B, TBR1)': ['BCL11B', 'TBR1', 'FEZF2'],
    'Motor Neuron (SLC5A7, MNX1)': ['SLC5A7', 'MNX1', 'CHAT']
}

motor_data = adata[adata.obs['leiden_neuronal'] == top_motor_cluster, :]
for layer, markers in layer_markers.items():
    print(f"\n{layer}:")
    for marker in markers:
        if marker in adata.var_names:
            idx = list(adata.var_names).index(marker)
            exp = motor_data.X[:, idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            print(f"  {marker}: {(exp > 0).mean() * 100:.1f}% expressing, mean={exp.mean():.3f}")
        else:
            print(f"  {marker}: not detected")

print(f"\nTop marker genes for motor neuron cluster {top_motor_cluster}:")
motor_genes_full = []
for i, gene in enumerate(result_neuronal['names'][top_motor_cluster][:50]):
    score = result_neuronal['scores'][top_motor_cluster][i]
    logfc = result_neuronal['logfoldchanges'][top_motor_cluster][i]
    pval = result_neuronal['pvals'][top_motor_cluster][i]
    motor_genes_full.append({'gene': gene, 'score': score, 'log2FC': logfc, 'p_value': pval})

motor_df_full = pd.DataFrame(motor_genes_full)
motor_df_full.to_csv(f'motor_neuron_cluster_{top_motor_cluster}_markers.csv', index=False)
print(f"\nSaved {len(motor_df_full)} marker genes")
print(motor_df_full.head(10))

upper_mn = ['BCL11B', 'FEZF2', 'CRYM', 'ETV1', 'SOX5']
lower_mn = ['ISL1', 'MNX1', 'HOXC4', 'HOXC5', 'HOXA5']

print("\nMotor neuron subtype analysis:")
print("Upper motor neuron markers:")
for marker in upper_mn:
    if marker in adata.var_names:
        idx = list(adata.var_names).index(marker)
        exp = adata[adata.obs['leiden_neuronal'] == top_motor_cluster, idx].X
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        print(f"  {marker}: {exp.mean():.3f}")

print("\nLower motor neuron markers:")
for marker in lower_mn:
    if marker in adata.var_names:
        idx = list(adata.var_names).index(marker)
        exp = adata[adata.obs['leiden_neuronal'] == top_motor_cluster, idx].X
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        print(f"  {marker}: {exp.mean():.3f}")

# Publication figure
fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 3, 1)
mask = adata.obs['leiden_neuronal'] == top_motor_cluster
ax1.scatter(adata.obsm['spatial'][~mask, 0], adata.obsm['spatial'][~mask, 1],
            c='lightgray', s=3, alpha=0.3, label='Other cells')
ax1.scatter(adata.obsm['spatial'][mask, 0], adata.obsm['spatial'][mask, 1],
            c='red', s=10, alpha=0.8, label='Motor neurons')
ax1.set_title('A: Spatial Distribution of Motor Neurons', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')
ax1.legend()

ax2 = plt.subplot(2, 3, 2)
if 'SLC5A7' in adata.var_names:
    idx = list(adata.var_names).index('SLC5A7')
    exp = adata.X[:, idx]
    if hasattr(exp, 'toarray'):
        exp = exp.toarray().flatten()
    scatter = ax2.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                          c=exp, cmap='Reds', s=5, alpha=0.7)
    ax2.set_title('B: SLC5A7 Expression (Motor Neuron Marker)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2)

ax3 = plt.subplot(2, 3, 3)
top10_markers = motor_df_full.head(10)
bars = ax3.barh(range(10), top10_markers['score'].values)
ax3.set_yticks(range(10))
ax3.set_yticklabels(top10_markers['gene'].values)
ax3.set_xlabel('Wilcoxon Score')
ax3.set_title('C: Top Motor Neuron Markers', fontsize=14, fontweight='bold')
for i, (bar, p) in enumerate(zip(bars, top10_markers['p_value'].values)):
    bar.set_color('darkred' if p < 0.001 else 'red' if p < 0.01 else 'salmon' if p < 0.05 else 'gray')

ax4 = plt.subplot(2, 3, 4)
motor_mask = adata.obs['leiden_neuronal'] == top_motor_cluster
fib_mask = adata.obs['leiden_neuronal'] == top_fibro_cluster
ax4.scatter(adata.obsm['spatial'][motor_mask, 0], adata.obsm['spatial'][motor_mask, 1],
            c='red', s=10, alpha=0.8, label='Motor neurons')
ax4.scatter(adata.obsm['spatial'][fib_mask, 0], adata.obsm['spatial'][fib_mask, 1],
            c='blue', s=10, alpha=0.8, label='Fibroblasts')
ax4.set_title('D: Motor Neuron-Fibroblast Interaction', fontsize=14, fontweight='bold')
ax4.set_aspect('equal')
ax4.legend()

ax5 = plt.subplot(2, 3, 5)
if 'GRIA2' in adata.var_names:
    idx = list(adata.var_names).index('GRIA2')
    exp = adata.X[:, idx]
    if hasattr(exp, 'toarray'):
        exp = exp.toarray().flatten()
    scatter = ax5.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                          c=exp, cmap='Blues', s=5, alpha=0.7)
    ax5.set_title('E: GRIA2 Expression (Glutamatergic)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax5)

ax6 = plt.subplot(2, 3, 6)
cluster_sizes = adata.obs['leiden_neuronal'].value_counts()
motor_pct = (cluster_sizes[top_motor_cluster] / len(adata)) * 100
fib_pct = (cluster_sizes[top_fibro_cluster] / len(adata)) * 100
other_pct = 100 - motor_pct - fib_pct
ax6.pie([motor_pct, fib_pct, other_pct], labels=['Motor Neurons', 'Fibroblasts', 'Other'],
        colors=['red', 'blue', 'lightgray'], autopct='%1.1f%%')
ax6.set_title('F: Tissue Composition', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('motor_neuron_publication_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Publication figure saved to motor_neuron_publication_figure.png")

# ============ ADDITIONAL NEURONAL SUBTYPE ANALYSIS ============
print("\n" + "=" * 60)
print("ADDITIONAL NEURONAL SUBTYPE ANALYSIS")
print("=" * 60)

subtype_markers = {
    'Sensory Spinal Neurons': {
        'General Sensory': ['NTRK1', 'NTRK2', 'NTRK3', 'RET', 'RUNX1', 'RUNX3'],
        'Mechanoreceptors': ['MECOM', 'MAFA', 'MAFB', 'NEFH', 'PVALB'],
        'Proprioceptors': ['ETV1', 'ETV4', 'RUNX3', 'PARV', 'NEFH'],
        'Nociceptors': ['TRPV1', 'TRPA1', 'SCN9A', 'SCN10A', 'TAC1', 'CALCA'],
        'Pruriceptors': ['SST', 'NPPB', 'NPY2R', 'MRGPRD']
    },
    'Cortical Neurons': {
        'General Cortical': ['SATB2', 'BCL11B', 'TBR1', 'CUX1', 'CUX2', 'RORB'],
        'Upper Layer (II-IV)': ['CUX1', 'CUX2', 'SATB2', 'RORB'],
        'Deep Layer (V-VI)': ['BCL11B', 'TBR1', 'SOX5', 'FEZF2', 'CTIP2'],
        'Layer V Corticospinal': ['BCL11B', 'FEZF2', 'CRYM', 'ETV1'],
        'Layer VI': ['TBR1', 'FOXP2', 'CTGF']
    },
    'GABAergic Neurons': {
        'General GABAergic': ['GAD1', 'GAD2', 'SLC32A1', 'GABRA1', 'GABRG2'],
        'Parvalbumin+ (PV)': ['PVALB', 'SST', 'VIP', 'CCK', 'NPY'],
        'Somatostatin+ (SST)': ['SST', 'NPY', 'CALB2'],
        'VIP+': ['VIP', 'CCK', 'CALB2'],
        'Calretinin+': ['CALB2', 'VIP'],
        'Calbindin+': ['CALB1']
    },
    'Glutamatergic Neurons': {
        'General Glutamatergic': ['SLC17A7', 'SLC17A6', 'GRIN1', 'GRIA2', 'GRIK1'],
        'AMPA Receptors': ['GRIA1', 'GRIA2', 'GRIA3', 'GRIA4'],
        'NMDA Receptors': ['GRIN1', 'GRIN2A', 'GRIN2B', 'GRIN2D']
    }
}

print("\n1. CHECKING NEURONAL SUBTYPE MARKER AVAILABILITY:")
print("-" * 60)

all_subtype_markers = {}
for category, subcategories in subtype_markers.items():
    print(f"\n{category}:")
    category_markers = []
    for subcat, markers in subcategories.items():
        found = [m for m in markers if m in adata.var_names]
        category_markers.extend(found)
        print(f"  {subcat}: {len(found)}/{len(markers)} found - {found if found else 'None'}")
    all_subtype_markers[category] = list(set(category_markers))

print("\n2. CALCULATING SUBTYPE SCORES PER CLUSTER:")
print("-" * 60)

subtype_scores = {}
for category, markers in all_subtype_markers.items():
    if len(markers) >= 2:
        print(f"\n{category} ({len(markers)} markers):")
        category_scores = []
        for group in neuronal_groups[:15]:
            mask = adata.obs['leiden_neuronal'] == group
            if mask.sum() > 0:
                cluster_exp = []
                for marker in markers:
                    if marker in adata.var_names:
                        marker_idx = list(adata.var_names).index(marker)
                        exp = adata[mask, marker_idx].X
                        if hasattr(exp, 'toarray'):
                            exp = exp.toarray().flatten()
                        cluster_exp.append(exp.mean())
                if cluster_exp:
                    category_scores.append({'cluster': group, 'score': np.mean(cluster_exp)})
        if category_scores:
            score_df = pd.DataFrame(category_scores).sort_values('score', ascending=False)
            subtype_scores[category] = score_df
            print(f"  Top 3 clusters:")
            for _, row in score_df.head(3).iterrows():
                star = "★" if row['cluster'] == top_motor_cluster else ""
                print(f"    Cluster {row['cluster']}{star}: {row['score']:.4f}")

print("\n3. CREATING SUBTYPE SCORE HEATMAP...")
heatmap_data, heatmap_rows = [], []
heatmap_cols = list(neuronal_groups[:15])

for category, score_df in subtype_scores.items():
    if not score_df.empty:
        heatmap_rows.append(category)
        row_data = []
        for cluster in heatmap_cols:
            score_val = score_df[score_df['cluster'] == cluster]['score'].values
            row_data.append(score_val[0] if len(score_val) > 0 else 0)
        heatmap_data.append(row_data)

if heatmap_data:
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_xticklabels(heatmap_cols, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_rows)))
    ax.set_yticklabels(heatmap_rows)
    if top_motor_cluster in heatmap_cols:
        motor_idx = heatmap_cols.index(top_motor_cluster)
        ax.axvline(x=motor_idx, color='red', linewidth=2, linestyle='--', alpha=0.5)
        ax.text(motor_idx, -0.5, 'Motor\nNeuron', ha='center', va='top', color='red', fontsize=9)
    plt.colorbar(im, ax=ax, label='Mean Expression')
    ax.set_title('Neuronal Subtype Scores by Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('neuronal_subtype_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Heatmap saved to neuronal_subtype_heatmap.png")

print("\n4. MOTOR NEURON CLUSTER SUBTYPE ANALYSIS:")
print("-" * 60)
print(f"Analyzing motor neuron cluster {top_motor_cluster}")

motor_data = adata[adata.obs['leiden_neuronal'] == top_motor_cluster, :]
for category, markers in all_subtype_markers.items():
    if markers:
        print(f"\n{category}:")
        for marker in markers[:10]:
            if marker in adata.var_names:
                idx = list(adata.var_names).index(marker)
                exp = motor_data.X[:, idx]
                if hasattr(exp, 'toarray'):
                    exp = exp.toarray().flatten()
                if exp.mean() > 0.1:
                    print(f"  {marker}: {(exp > 0).mean() * 100:.1f}% expressing, mean={exp.mean():.3f}")

print("\n5. CREATING SUBTYPE SCORE SUMMARY...")
all_scores_list = []
for category, score_df in subtype_scores.items():
    for _, row in score_df.iterrows():
        all_scores_list.append({'cluster': row['cluster'], 'subtype': category, 'score': row['score']})

all_scores_df = pd.DataFrame(all_scores_list)
pivot_scores = all_scores_df.pivot(index='cluster', columns='subtype', values='score').fillna(0)
pivot_scores.to_csv('neuronal_subtype_scores_all.csv')
print("Subtype scores saved to neuronal_subtype_scores_all.csv")
print("\nTop clusters for each subtype:")
print(pivot_scores.head(10))

print("\n6. DOMINANT SUBTYPE IN MOTOR NEURON CLUSTER:")
print("-" * 60)
motor_subtypes = pivot_scores.loc[top_motor_cluster] if top_motor_cluster in pivot_scores.index else None
if motor_subtypes is not None:
    motor_subtypes = motor_subtypes.sort_values(ascending=False)
    print(f"\nMotor neuron cluster {top_motor_cluster} subtype scores:")
    for subtype, score in motor_subtypes.head(5).items():
        print(f"  {subtype}: {score:.4f}")
    dominant = motor_subtypes.index[0]
    print(f"\n→ Dominant subtype: {dominant}")

print("\n7. VISUALIZING TOP SUBTYPES...")
top_subtypes = all_scores_df.groupby('subtype')['score'].mean().nlargest(5).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, subtype in enumerate(top_subtypes[:5]):
    subtype_data = all_scores_df[all_scores_df['subtype'] == subtype]
    axes[i].bar(range(len(subtype_data)), subtype_data['score'].values)
    axes[i].set_xticks(range(len(subtype_data)))
    axes[i].set_xticklabels(subtype_data['cluster'].values, rotation=45, ha='right')
    axes[i].set_title(f'{subtype} Scores')
    axes[i].set_xlabel('Cluster')
    axes[i].set_ylabel('Score')
    if top_motor_cluster in subtype_data['cluster'].values:
        motor_idx = list(subtype_data['cluster']).index(top_motor_cluster)
        axes[i].patches[motor_idx].set_color('red')
if len(top_subtypes) < 6:
    axes[5].set_visible(False)
plt.tight_layout()
plt.savefig('subtype_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Additional neuronal subtype analysis complete!")

# ============ SKELETAL MUSCLE ANALYSIS ============
print("\n" + "=" * 60)
print("SKELETAL MUSCLE ANALYSIS")
print("=" * 60)

skeletal_muscle_markers = {
    'Myogenic regulators': ['MYOD1', 'MYF5', 'MYOG', 'PAX3', 'PAX7'],
    'Structural proteins': ['MYH1', 'MYH2', 'MYH3', 'MYH7', 'MYH8', 'ACTA1', 'DES', 'TNNT1', 'TNNT3', 'TNNC1', 'TNNC2'],
    'Sarcomeric': ['TTN', 'NEB', 'ACTN2', 'MYBPC1', 'MYBPC2', 'MYL1', 'MYL2', 'MYL3'],
    'Muscle-specific': ['CKMT2', 'CKM', 'CHRNG', 'CHRND', 'CHRNE']
}

print("\n1. CHECKING SKELETAL MUSCLE MARKER AVAILABILITY:")
print("-" * 40)
all_muscle_markers = []
for category, markers in skeletal_muscle_markers.items():
    found = [m for m in markers if m in adata.var_names]
    all_muscle_markers.extend(found)
    print(f"{category}: {len(found)}/{len(markers)} found - {found if found else 'None'}")

print(f"\nTotal skeletal muscle markers found: {len(all_muscle_markers)}")
print(f"Markers: {all_muscle_markers}")

if len(all_muscle_markers) >= 2:
    print("\n2. CALCULATING SKELETAL MUSCLE SCORES PER CLUSTER:")
    print("-" * 40)

    muscle_scores_dict = {}
    for group in neuronal_groups:
        mask = adata.obs['leiden_neuronal'] == group
        if mask.sum() > 0:
            cluster_exp = []
            for marker in all_muscle_markers:
                if marker in adata.var_names:
                    marker_idx = list(adata.var_names).index(marker)
                    exp = adata[mask, marker_idx].X
                    if hasattr(exp, 'toarray'):
                        exp = exp.toarray().flatten()
                    cluster_exp.append(exp.mean())
            if cluster_exp:
                muscle_scores_dict[group] = np.mean(cluster_exp)

    if muscle_scores_dict:
        muscle_score_df = pd.DataFrame([{'cluster': k, 'muscle_score': v}
                                        for k, v in muscle_scores_dict.items()]).sort_values('muscle_score', ascending=False)
        print("\nTop clusters with skeletal muscle signature:")
        print(muscle_score_df.head(10))
        muscle_score_df.to_csv('skeletal_muscle_clusters.csv', index=False)

        top_muscle_cluster = muscle_score_df.iloc[0]['cluster']
        print(f"\n✓ Top skeletal muscle cluster: {top_muscle_cluster}")
        print(f"  Score: {muscle_score_df.iloc[0]['muscle_score']:.4f}")
        if top_muscle_cluster == top_motor_cluster:
            print("  → This is the SAME as your motor neuron cluster!")
        else:
            print(f"  → Different from motor neuron cluster {top_motor_cluster}")

        fig, ax = plt.subplots(figsize=(8, 8))
        mask = adata.obs['leiden_neuronal'] == top_muscle_cluster
        ax.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], c='lightgray', s=3, alpha=0.3)
        ax.scatter(adata.obsm['spatial'][mask, 0], adata.obsm['spatial'][mask, 1],
                   c='green', s=10, alpha=0.8, label=f'Skeletal Muscle Cluster {top_muscle_cluster}')
        ax.set_title(f'Skeletal Muscle Cluster {top_muscle_cluster}')
        ax.set_aspect('equal')
        ax.legend()
        plt.savefig('skeletal_muscle_spatial.png', dpi=150, bbox_inches='tight')
        plt.show()

# ============ TARDBP ENRICHMENT ANALYSIS ============
print("\n" + "=" * 60)
print("TARDBP ENRICHMENT ANALYSIS")
print("=" * 60)

c9_gene = 'TARDBP'
if c9_gene in adata.var_names:
    print(f"\n✓ {c9_gene} found in dataset")
    c9_idx = list(adata.var_names).index(c9_gene)
    c9_exp = adata.X[:, c9_idx]
    if hasattr(c9_exp, 'toarray'):
        c9_exp = c9_exp.toarray().flatten()
    adata.obs['TARDBP_expression'] = c9_exp

    if 'orig.ident' in adata.obs.columns:
        organoids_tardbp = adata.obs['orig.ident'].unique()
        c9_by_organoid = []
        for organoid in organoids_tardbp:
            mask = adata.obs['orig.ident'] == organoid
            organoid_exp = c9_exp[mask]
            stats = {'organoid': organoid, 'mean_expression': organoid_exp.mean(),
                     'median_expression': np.median(organoid_exp), 'max_expression': organoid_exp.max(),
                     'percent_expressing': (organoid_exp > 0).mean() * 100, 'n_cells': mask.sum()}
            c9_by_organoid.append(stats)
            print(f"\n{organoid}: mean={stats['mean_expression']:.4f}, "
                  f"% expressing={stats['percent_expressing']:.1f}%")
        pd.DataFrame(c9_by_organoid).to_csv('TARDBP_by_organoid.csv', index=False)
else:
    print(f"\n✗ {c9_gene} NOT found in dataset")
    similar = [g for g in adata.var_names if 'C9orf' in g or 'C9' in g]
    print(f"Available genes similar to TARDBP:")
    print(similar[:10] if similar else "None found")

# ============ ALS/NEURODEGENERATION GENE ANALYSIS ============
print("\n" + "=" * 60)
print("ALS/NEURODEGENERATION GENE ANALYSIS")
print("=" * 60)

als_gene_groups = {
    'ALS_Core': ['TARDBP', 'FUS', 'OPTN', 'SOD1', 'NEK1', 'TBK1', 'CHMP2B', 'UNC13A'],
    'FTD_Core': ['MAPT', 'GRN', 'TMEM106B'],
    'HOX_Spinal': ['HOXA7', 'HOXA10', 'HOXA11'],
    'Signaling': ['STK10', 'MAP4K3', 'EFR3A', 'EPHA4'],
    'Other': ['SHOX2', 'CAMTA1', 'CDH22', 'MTX2']
}

print("\n1. CHECKING ALS GENE AVAILABILITY:")
print("-" * 60)

all_als_markers = []
for group_name, genes in als_gene_groups.items():
    available = [g for g in genes if g in adata.var_names]
    all_als_markers.extend(available)
    print(f"\n{group_name}: {len(available)}/{len(genes)} found")
    if available:
        print(f"  → {', '.join(available)}")
    else:
        print(f"  → None found")

print(f"\nTotal ALS-related genes found: {len(all_als_markers)}")
if all_als_markers:
    print(f"Genes: {', '.join(all_als_markers)}")

if all_als_markers:
    print(f"\n2. ANALYZING ALS GENE EXPRESSION:")
    print("-" * 60)

    als_stats = []
    for gene in all_als_markers:
        idx = list(adata.var_names).index(gene)
        exp = adata.X[:, idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        stats = {'gene': gene, 'mean': exp.mean(), 'median': np.median(exp),
                 'std': exp.std(), 'sem': exp.std() / np.sqrt(len(exp)),
                 'min': exp.min(), 'max': exp.max(),
                 'pct_expressing': (exp > 0).mean() * 100, 'n_cells': len(exp)}
        als_stats.append(stats)
        print(f"\n📊 {gene}:")
        print(f"    Mean ± SEM: {stats['mean']:.4f} ± {stats['sem']:.4f}")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    % expressing: {stats['pct_expressing']:.1f}%")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    als_stats_df = pd.DataFrame(als_stats)
    als_stats_df.to_csv('ALS_genes_statistics.csv', index=False)
    print("\n✅ ALS gene statistics saved to 'ALS_genes_statistics.csv'")

    print("\n3. CREATING VISUALIZATIONS:")
    print("-" * 60)

    genes_list = als_stats_df['gene'].values
    means = als_stats_df['mean'].values
    errors = als_stats_df['sem'].values
    x_pos = np.arange(len(genes_list))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genes_list, rotation=45, ha='right')
    ax.set_ylabel('Mean Expression ± SEM')
    ax.set_title('ALS Gene Expression Levels', fontweight='bold')
    plt.tight_layout()
    plt.savefig('ALS_genes_barplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Bar plot saved to 'ALS_genes_barplot.png'")

    fig, ax = plt.subplots(figsize=(12, 6))
    pct = als_stats_df['pct_expressing'].values
    bars = ax.bar(x_pos, pct, color='coral', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genes_list, rotation=45, ha='right')
    ax.set_ylabel('% Expressing Cells')
    ax.set_ylim(0, 100)
    ax.set_title('Percentage of Cells Expressing ALS Genes', fontweight='bold')
    plt.tight_layout()
    plt.savefig('ALS_genes_percentage.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Percentage plot saved to 'ALS_genes_percentage.png'")

    if len(all_als_markers) >= 2:
        als_exp_matrix = []
        for gene in all_als_markers:
            idx = list(adata.var_names).index(gene)
            exp = adata.X[:, idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            als_exp_matrix.append(exp)
        als_exp_matrix = np.array(als_exp_matrix).T
        corr_matrix = np.corrcoef(als_exp_matrix.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(all_als_markers)))
        ax.set_xticklabels(all_als_markers, rotation=45, ha='right')
        ax.set_yticks(range(len(all_als_markers)))
        ax.set_yticklabels(all_als_markers)
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('ALS Gene Expression Correlation Matrix', fontsize=14, fontweight='bold')
        for i in range(len(all_als_markers)):
            for j in range(len(all_als_markers)):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
        plt.tight_layout()
        plt.savefig('ALS_genes_correlation.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✅ Correlation matrix saved to 'ALS_genes_correlation.png'")

    print("\n4. CLUSTER-SPECIFIC ENRICHMENT:")
    print("-" * 60)
    for gene in all_als_markers:
        print(f"\n{gene} expression by cluster:")
        idx = list(adata.var_names).index(gene)
        exp = adata.X[:, idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        threshold = np.percentile(exp, 75)
        cluster_means = []
        for cluster in neuronal_groups[:10]:
            mask = adata.obs['leiden_neuronal'] == cluster
            if mask.sum() > 0:
                cluster_exp = exp[mask]
                mean_exp = cluster_exp.mean()
                pct_high = (cluster_exp > threshold).mean() * 100
                if mean_exp > 0.01:
                    cluster_means.append((cluster, mean_exp, pct_high))
        cluster_means.sort(key=lambda x: x[1], reverse=True)
        for cluster, mean_exp, pct_high in cluster_means[:5]:
            star = "★" if cluster == top_motor_cluster else ""
            print(f"  Cluster {cluster}{star}: mean={mean_exp:.4f}, {pct_high:.1f}% high expressors")

    print("\n5. SPATIAL VISUALIZATION OF ALS GENES:")
    print("-" * 60)
    spatial_coords_als = adata.obsm['spatial']
    n_genes = len(all_als_markers)
    n_cols = min(3, n_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_genes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, gene in enumerate(all_als_markers):
        idx = list(adata.var_names).index(gene)
        exp = adata.X[:, idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        vmax = np.percentile(exp[exp > 0], 95) if (exp > 0).any() else 1
        scatter = axes[i].scatter(spatial_coords_als[:, 0], spatial_coords_als[:, 1],
                                  c=exp, cmap='Reds', s=5, alpha=0.7, vmin=0, vmax=vmax)
        axes[i].set_title(f'{gene} Expression', fontweight='bold')
        axes[i].set_aspect('equal')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(scatter, ax=axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Spatial Expression of ALS-Related Genes', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ALS_genes_spatial.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Spatial plots saved to 'ALS_genes_spatial.png'")

    print("\n" + "=" * 60)
    print("ALS GENE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nTotal ALS-related genes analyzed: {len(all_als_markers)}")
    print(f"\nGenes detected: {', '.join(all_als_markers)}")
    print("\nExpression Summary:")
    print(als_stats_df[['gene', 'mean', 'pct_expressing']].round(4).to_string(index=False))

else:
    print("\n❌ No ALS-related genes found in dataset")

print("\n✅ ALS gene analysis complete!")

# ============ SEPARATING INDIVIDUAL ORGANOIDS ============
print("\n" + "=" * 60)
print("SEPARATING INDIVIDUAL ORGANOIDS")
print("=" * 60)

from sklearn.cluster import KMeans

spatial_coords = adata.obsm['spatial']
print("\n1. Using K-means to separate organoids...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
spatial_clusters = kmeans.fit_predict(spatial_coords)
adata.obs['organoid'] = [f'Organoid_{i + 1}' for i in spatial_clusters]

print("Organoid distribution:")
print(adata.obs['organoid'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                c=adata.obs['leiden'].astype('category').cat.codes, cmap='tab10', s=5, alpha=0.7)
axes[0].set_title('Original Clusters')
axes[0].set_aspect('equal')
scatter = axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                          c=spatial_clusters, cmap='Set1', s=5, alpha=0.7)
axes[1].set_title('K-means Separated Organoids')
axes[1].set_aspect('equal')
plt.colorbar(scatter, ax=axes[1])
plt.tight_layout()
plt.savefig('organoid_separation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ STATISTICAL COMPARISON BETWEEN ORGANOIDS ============
print("\n" + "=" * 60)
print("STATISTICAL COMPARISON BETWEEN ORGANOIDS")
print("=" * 60)

print(f"Using motor neuron cluster: {top_motor_cluster}")

organoids = adata.obs['organoid'].unique()
print(f"\nOrganoids identified: {organoids.tolist()}")

print("\n1. MOTOR NEURON ABUNDANCE BY ORGANOID:")
print("-" * 40)

motor_abundance = []
for organoid in organoids:
    mask_organoid = adata.obs['organoid'] == organoid
    mask_motor = adata.obs['leiden_neuronal'] == top_motor_cluster
    mask = mask_organoid & mask_motor
    total_cells = mask_organoid.sum()
    motor_cells = mask.sum()
    stats = {'organoid': organoid, 'total_cells': total_cells,
             'motor_neuron_cells': motor_cells,
             'percentage_motor': (motor_cells / total_cells * 100) if total_cells > 0 else 0}
    motor_abundance.append(stats)
    print(f"\n{organoid}:")
    print(f"  Total cells: {total_cells}")
    print(f"  Motor neurons: {motor_cells}")
    print(f"  % Motor neurons: {stats['percentage_motor']:.2f}%")

motor_df_organoid = pd.DataFrame(motor_abundance)
motor_df_organoid.to_csv('motor_neuron_abundance_by_organoid.csv', index=False)

from scipy.stats import chi2_contingency

print("\n2. STATISTICAL TEST FOR MOTOR NEURON ABUNDANCE:")
print("-" * 40)

contingency_data = []
for organoid in organoids:
    mask_organoid = adata.obs['organoid'] == organoid
    mask_motor = adata.obs['leiden_neuronal'] == top_motor_cluster
    motor_cells = (mask_organoid & mask_motor).sum()
    other_cells = (mask_organoid & ~mask_motor).sum()
    contingency_data.append([motor_cells, other_cells])

contingency = pd.DataFrame(contingency_data, index=organoids,
                           columns=['Motor Neurons', 'Other Cells'])
print("\nContingency table:")
print(contingency)

chi2_val, p_val, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test for independence:")
print(f"  χ² = {chi2_val:.3f}")
print(f"  p-value = {p_val:.6f}")
if p_val < 0.05:
    print("  → SIGNIFICANT difference in motor neuron abundance between organoids")
else:
    print("  → No significant difference")

print("\n3. MOTOR NEURON MARKER EXPRESSION BY ORGANOID:")
print("-" * 40)

if 'SLC5A7' in adata.var_names:
    marker_idx = list(adata.var_names).index('SLC5A7')
    slc5a7_exp = adata.X[:, marker_idx]
    if hasattr(slc5a7_exp, 'toarray'):
        slc5a7_exp = slc5a7_exp.toarray().flatten()

    marker_stats = []
    expression_groups = []

    for organoid in organoids:
        mask = adata.obs['organoid'] == organoid
        organoid_exp = slc5a7_exp[mask]
        expression_groups.append(organoid_exp)
        stats = {'organoid': organoid, 'mean_expression': organoid_exp.mean(),
                 'median_expression': np.median(organoid_exp),
                 'std_expression': organoid_exp.std(),
                 'sem_expression': organoid_exp.std() / np.sqrt(len(organoid_exp)),
                 'percent_expressing': (organoid_exp > 0).mean() * 100}
        marker_stats.append(stats)
        print(f"\n{organoid}:")
        print(f"  Mean SLC5A7: {stats['mean_expression']:.4f} ± {stats['sem_expression']:.4f}")
        print(f"  % expressing: {stats['percent_expressing']:.1f}%")

    marker_df = pd.DataFrame(marker_stats)
    marker_df.to_csv('SLC5A7_expression_by_organoid.csv', index=False)

    from scipy.stats import f_oneway
    f_stat, p_val_anova = f_oneway(*expression_groups)
    print(f"\nANOVA test for SLC5A7 expression:")
    print(f"  F-statistic = {f_stat:.3f}")
    print(f"  p-value = {p_val_anova:.6f}")
    if p_val_anova < 0.05:
        print("  → SIGNIFICANT difference in SLC5A7 expression between organoids")
    else:
        print("  → No significant difference")

    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    print("\nPairwise comparisons (t-test with Bonferroni correction):")
    p_values = []
    comparisons = []
    for i in range(len(organoids)):
        for j in range(i + 1, len(organoids)):
            t_stat, p = ttest_ind(expression_groups[i], expression_groups[j])
            p_values.append(p)
            comparisons.append(f"{organoids[i]} vs {organoids[j]}")

    rejected, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
    for comp, p_raw, p_corr in zip(comparisons, p_values, p_corrected):
        sig_star = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        print(f"  {comp}: p={p_raw:.4f} (adj={p_corr:.4f}) {sig_star}")

print("\n4. VISUALIZING MOTOR NEURONS BY ORGANOID:")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()
colors_org = ['red', 'blue', 'green', 'purple']

for i, organoid in enumerate(organoids):
    if i < 4:
        organoid_mask = adata.obs['organoid'] == organoid
        axes[i].scatter(spatial_coords[organoid_mask, 0], spatial_coords[organoid_mask, 1],
                        c='lightgray', s=3, alpha=0.3)
        motor_mask_org = organoid_mask & (adata.obs['leiden_neuronal'] == top_motor_cluster)
        axes[i].scatter(spatial_coords[motor_mask_org, 0], spatial_coords[motor_mask_org, 1],
                        c=colors_org[i % len(colors_org)], s=10, alpha=0.8, label='Motor Neurons')
        axes[i].set_title(f'{organoid}')
        axes[i].set_aspect('equal')
        axes[i].legend()

plt.tight_layout()
plt.savefig('motor_neurons_by_organoid.png', dpi=150, bbox_inches='tight')
plt.show()

if 'SLC5A7' in adata.var_names:
    print("\n5. CREATING BOXPLOT OF SLC5A7 EXPRESSION BY ORGANOID:")
    print("-" * 40)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = [slc5a7_exp[adata.obs['organoid'] == org] for org in organoids]
    bp = ax.boxplot(plot_data, patch_artist=True, labels=organoids)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors_org[i % len(colors_org)])
    ax.set_ylabel('SLC5A7 Expression')
    ax.set_title('Motor Neuron Marker Expression by Organoid')
    if 'p_val_anova' in locals() and p_val_anova < 0.05:
        ax.text(0.5, 0.95, f'ANOVA: p={p_val_anova:.4f}', transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig('slc5a7_by_organoid_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "=" * 60)
print("SUMMARY: ORGANOID COMPARISON")
print("=" * 60)

summary_df = motor_df_organoid
if 'marker_df' in locals():
    summary_df = summary_df.merge(marker_df, on='organoid', how='outer')

print("\nOrganoid Comparison Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv('organoid_comparison_summary.csv', index=False)

print("\n✅ Organoid separation and comparison complete!")
print("Files saved:")
print("  - organoid_separation.png")
print("  - motor_neuron_abundance_by_organoid.csv")
if 'marker_df' in locals():
    print("  - SLC5A7_expression_by_organoid.csv")
    print("  - slc5a7_by_organoid_boxplot.png")
print("  - motor_neurons_by_organoid.png")
print("  - organoid_comparison_summary.csv")

# ============ UMAPS FOR EACH ORGANOID ============
# FIX: all sc.pl.umap calls with gene color now include use_raw=False
print("\n" + "=" * 60)
print("CREATING INDIVIDUAL UMAPS FOR EACH ORGANOID")
print("=" * 60)

organoids = adata.obs['organoid'].unique()
print(f"Organoids: {organoids.tolist()}")

key_genes = ['SLC5A7', 'BCL11B', 'SNAP25', 'GRIA2', 'SLC32A1',
             'COL1A1', 'COL3A1', 'DCN', 'RBFOX3', 'CSF1R', 'TREM2']

available_genes = [g for g in key_genes if g in adata.var_names]
print(f"\nGenes to visualize: {available_genes}")

print("\n1. CREATING INDIVIDUAL UMAPS WITH GENE OVERLAYS...")

for organoid in organoids:
    organoid_mask = adata.obs['organoid'] == organoid
    adata_organoid = adata[organoid_mask, :].copy()

    if adata_organoid.shape[0] < 10:
        print(f"  Skipping {organoid} - only {adata_organoid.shape[0]} cells")
        continue

    print(f"\n  Processing {organoid}: {adata_organoid.shape[0]} cells")

    # Check which genes are expressed in this subset
    genes_in_organoid = []
    for gene in available_genes:
        if gene in adata_organoid.var_names:
            gene_idx = list(adata_organoid.var_names).index(gene)
            exp = adata_organoid.X[:, gene_idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            if exp.max() > 0:
                genes_in_organoid.append(gene)

    print(f"    Genes expressed in this organoid: {genes_in_organoid}")

    if not genes_in_organoid:
        print(f"    No marker genes expressed in {organoid}, skipping...")
        continue

    n_genes = len(genes_in_organoid) + 1
    n_cols = min(3, n_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_genes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Panel 1: clusters (obs column, no use_raw needed)
    try:
        sc.pl.umap(adata_organoid, color='leiden_neuronal', ax=axes[0],
                   title=f'{organoid} - Clusters', show=False, legend_loc='on data')
    except Exception as e:
        print(f"    Warning: Could not plot clusters - {e}")
        axes[0].text(0.5, 0.5, 'Clusters\nnot available',
                     ha='center', va='center', transform=axes[0].transAxes)

    # Remaining panels: gene expression — FIX: use_raw=False
    for i, gene in enumerate(genes_in_organoid, start=1):
        if i < len(axes):
            try:
                sc.pl.umap(adata_organoid, color=gene, ax=axes[i],
                           title=f'{organoid} - {gene}', show=False,
                           cmap='Reds', vmax='p95', use_raw=False)   # ← FIX
            except Exception as e:
                print(f"    Warning: Could not plot {gene} - {e}")
                axes[i].text(0.5, 0.5, f'{gene}\nnot available',
                             ha='center', va='center', transform=axes[i].transAxes)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{organoid} UMAP Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'umap_{organoid}_genes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: umap_{organoid}_genes.png")

print("\n2. CREATING SIDE-BY-SIDE UMAP COMPARISONS FOR EACH GENE...")

for gene in available_genes:
    print(f"\n  Creating UMAPs for {gene} across organoids...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, organoid in enumerate(organoids):
        if i < 4:
            organoid_mask = adata.obs['organoid'] == organoid
            adata_organoid = adata[organoid_mask, :].copy()

            if adata_organoid.shape[0] >= 5:
                try:
                    if gene in adata_organoid.var_names:
                        # FIX: use_raw=False
                        sc.pl.umap(adata_organoid, color=gene, ax=axes[i],
                                   title=f'{organoid} - {gene}', show=False,
                                   cmap='Reds', vmax='p95', use_raw=False)   # ← FIX
                    else:
                        axes[i].text(0.5, 0.5, f'{gene}\nnot in dataset',
                                     ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f'{organoid}')
                except Exception as e:
                    print(f"    Warning: Could not plot {organoid} - {e}")
                    axes[i].text(0.5, 0.5, f'Error\n{str(e)[:30]}',
                                 ha='center', va='center', transform=axes[i].transAxes)
            else:
                axes[i].text(0.5, 0.5, f'{organoid}\n(insufficient cells)',
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{organoid}')

    plt.suptitle(f'{gene} Expression Across Organoids', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'umap_comparison_{gene}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: umap_comparison_{gene}.png")

print("\n3. CREATING SUMMARY PLOTS FOR TOP MOTOR NEURON MARKERS...")

top_motor_genes = []
if 'rank_genes_groups' in adata.uns:
    result_neuronal_check = adata.uns['rank_genes_groups']
    if top_motor_cluster in result_neuronal_check['names'].dtype.names:
        for gene in result_neuronal_check['names'][top_motor_cluster][:5]:
            if gene in adata.var_names:
                top_motor_genes.append(gene)

if not top_motor_genes:
    top_motor_genes = [g for g in ['SLC5A7', 'BCL11B', 'SNAP25'] if g in adata.var_names]

if top_motor_genes:
    print(f"Top motor neuron markers: {top_motor_genes}")

    for organoid in organoids:
        organoid_mask = adata.obs['organoid'] == organoid
        adata_organoid = adata[organoid_mask, :].copy()

        if adata_organoid.shape[0] < 10:
            continue

        motor_in_organoid = [g for g in top_motor_genes if g in adata_organoid.var_names]
        if not motor_in_organoid:
            print(f"    No motor markers expressed in {organoid}")
            continue

        n_plots = len(motor_in_organoid) + 1
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        try:
            sc.pl.umap(adata_organoid, color='leiden_neuronal', ax=axes[0],
                       title=f'{organoid} - Clusters', show=False, legend_loc='on data')
        except:
            axes[0].text(0.5, 0.5, 'Clusters', ha='center', va='center')

        for i, gene in enumerate(motor_in_organoid):
            try:
                # FIX: use_raw=False
                sc.pl.umap(adata_organoid, color=gene, ax=axes[i + 1],
                           title=f'{organoid} - {gene}', show=False,
                           cmap='Reds', use_raw=False)   # ← FIX
            except:
                axes[i + 1].text(0.5, 0.5, f'{gene}', ha='center', va='center')

        plt.suptitle(f'{organoid} - Motor Neuron Markers', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'umap_{organoid}_motor_markers.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  ✅ Saved: umap_{organoid}_motor_markers.png")
else:
    print("  No motor neuron markers available")

print("\n✅ UMAP generation complete!")

# ============ CUSTOM COLORED UMAPS FOR EACH ORGANOID WITH CELL TYPE LABELS ============
print("\n" + "=" * 60)
print("CREATING CUSTOM COLORED UMAPS FOR EACH ORGANOID")
print("=" * 60)

# Define cell type colors based on your image
cell_type_colors = {
    'Motor Neurons': '#FF0000',  # Red
    'Cortical Neurons': '#FFA500',  # Orange
    'Glutamatergic': '#FFFF00',  # Yellow
    'GABAergic': '#00FF00',  # Green
    'Neural Progenitors': '#ADD8E6',  # Light Blue
    'Fibroblasts': '#FFC0CB',  # Pink
    'Glial': '#808080',  # Gray
    'Other': '#FFFFFF'  # White
}

# First, assign cell type labels to each cluster based on your marker analysis
print("\n1. ASSIGNING CELL TYPE LABELS TO CLUSTERS...")

# Create a mapping from cluster to cell type
cluster_to_celltype = {}

# From your analysis results:
# - Motor neuron cluster is {top_motor_cluster}
# - Fibroblast cluster is {top_fibro_cluster}
# - Other clusters need to be assigned based on marker expression

# Calculate scores for each cluster to determine cell type
cell_type_scores = {}

# Define marker genes for each cell type
cell_type_markers_custom = {
    'Motor Neurons': ['SLC5A7', 'BCL11B', 'CHAT'],
    'Cortical Neurons': ['SNAP25', 'RBFOX3', 'MAP2'],
    'Glutamatergic': ['GRIA2', 'SLC17A7', 'GRIN1'],
    'GABAergic': ['SLC32A1', 'GAD1', 'GAD2'],
    'Neural Progenitors': ['SOX2', 'PAX6', 'NES'],
    'Fibroblasts': ['COL1A1', 'COL3A1', 'DCN'],
    'Glial': ['GFAP', 'S100B', 'OLIG2']
}

# For each cluster, calculate score for each cell type
for cluster in neuronal_groups:
    mask = adata.obs['leiden_neuronal'] == cluster
    if mask.sum() == 0:
        continue

    scores = {}
    for cell_type, markers in cell_type_markers_custom.items():
        available_markers = [m for m in markers if m in adata.var_names]
        if available_markers:
            cluster_exp = []
            for marker in available_markers:
                marker_idx = list(adata.var_names).index(marker)
                exp = adata[mask, marker_idx].X
                if hasattr(exp, 'toarray'):
                    exp = exp.toarray().flatten()
                cluster_exp.append(exp.mean())
            scores[cell_type] = np.mean(cluster_exp)
        else:
            scores[cell_type] = 0

    # Assign cell type based on highest score
    if scores:
        best_cell_type = max(scores, key=scores.get)
        # Only assign if score is meaningful
        if scores[best_cell_type] > 0.01:
            cluster_to_celltype[cluster] = best_cell_type
        else:
            cluster_to_celltype[cluster] = 'Other'

# Force known clusters
cluster_to_celltype[top_motor_cluster] = 'Motor Neurons'
cluster_to_celltype[top_fibro_cluster] = 'Fibroblasts'

print("\nCluster to cell type mapping:")
for cluster in sorted(cluster_to_celltype.keys()):
    print(f"  Cluster {cluster}: {cluster_to_celltype[cluster]}")

# Add cell type labels to adata
adata.obs['cell_type'] = adata.obs['leiden_neuronal'].map(cluster_to_celltype).fillna('Other')

# Create a color map for all unique cell types
unique_cell_types = adata.obs['cell_type'].unique()
cell_type_color_map = {}
for ct in unique_cell_types:
    if ct in cell_type_colors:
        cell_type_color_map[ct] = cell_type_colors[ct]
    else:
        # Assign random color for any missing types
        cell_type_color_map[ct] = '#{:06x}'.format(np.random.randint(0, 0xFFFFFF))

print(f"\nCell types found: {list(unique_cell_types)}")

# ============ CREATE UMAP FOR EACH ORGANOID WITH CUSTOM COLORS ============
print("\n2. CREATING CUSTOM UMAP FOR EACH ORGANOID...")

for organoid in organoids:
    # Subset data for this organoid
    organoid_mask = adata.obs['organoid'] == organoid
    adata_organoid = adata[organoid_mask, :].copy()

    if adata_organoid.shape[0] < 10:
        print(f"  Skipping {organoid} - only {adata_organoid.shape[0]} cells")
        continue

    print(f"\n  Processing {organoid}: {adata_organoid.shape[0]} cells")

    # Get cell types present in this organoid
    organoid_cell_types = adata_organoid.obs['cell_type'].unique()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot each cell type separately with its own color
    for cell_type in organoid_cell_types:
        mask = adata_organoid.obs['cell_type'] == cell_type
        color = cell_type_color_map[cell_type]

        ax.scatter(adata_organoid.obsm['X_umap'][mask, 0],
                   adata_organoid.obsm['X_umap'][mask, 1],
                   c=color, s=10, alpha=0.8, label=cell_type, edgecolors='none')

    # Customize the plot
    ax.set_title(f'{organoid} - UMAP', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP_1', fontsize=12)
    ax.set_ylabel('UMAP_2', fontsize=12)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                       frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    plt.savefig(f'umap_{organoid}_colored.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: umap_{organoid}_colored.png")

# ============ CREATE SIDE-BY-SIDE UMAPS FOR ALL ORGANOIDS ============
print("\n3. CREATING SIDE-BY-SIDE UMAP COMPARISON...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i, organoid in enumerate(organoids):
    if i >= 4:
        break

    organoid_mask = adata.obs['organoid'] == organoid
    adata_organoid = adata[organoid_mask, :].copy()

    if adata_organoid.shape[0] < 5:
        axes[i].text(0.5, 0.5, f'{organoid}\n(insufficient cells)',
                     ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(f'{organoid}')
        continue

    # Plot each cell type
    organoid_cell_types = adata_organoid.obs['cell_type'].unique()

    for cell_type in organoid_cell_types:
        mask = adata_organoid.obs['cell_type'] == cell_type
        color = cell_type_color_map[cell_type]

        axes[i].scatter(adata_organoid.obsm['X_umap'][mask, 0],
                        adata_organoid.obsm['X_umap'][mask, 1],
                        c=color, s=8, alpha=0.8, label=cell_type if i == 0 else "",
                        edgecolors='none')

    axes[i].set_title(f'{organoid}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('UMAP_1', fontsize=10)
    axes[i].set_ylabel('UMAP_2', fontsize=10)
    axes[i].tick_params(axis='both', which='both', length=0)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

# Add a single legend for all plots
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cell_type_color_map[ct],
                      markersize=10, label=ct) for ct in unique_cell_types]
fig.legend(handles=handles, bbox_to_anchor=(0.5, 0.02), loc='lower center',
           ncol=min(4, len(unique_cell_types)), frameon=True, fancybox=True, shadow=True)

plt.suptitle('Cell Type Distribution Across Organoids', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('umap_all_organoids_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("  ✅ Saved: umap_all_organoids_comparison.png")

# ============ CREATE UMAP WITH TOP MARKER GENES FOR EACH ORGANOID ============
print("\n4. CREATING UMAPS WITH TOP MARKER GENES FOR EACH ORGANOID...")

# Get top 3 marker genes for each cell type
top_markers_by_type = {}

for cell_type, markers in cell_type_markers_custom.items():
    available = [m for m in markers if m in adata.var_names]
    if available:
        top_markers_by_type[cell_type] = available[:3]

print("\nTop markers by cell type:")
for ct, markers in top_markers_by_type.items():
    print(f"  {ct}: {markers}")

for organoid in organoids:
    organoid_mask = adata.obs['organoid'] == organoid
    adata_organoid = adata[organoid_mask, :].copy()

    if adata_organoid.shape[0] < 10:
        continue

    # Collect all unique markers to plot
    all_markers_to_plot = []
    for markers in top_markers_by_type.values():
        all_markers_to_plot.extend([m for m in markers if m in adata_organoid.var_names])
    all_markers_to_plot = list(set(all_markers_to_plot))

    if not all_markers_to_plot:
        continue

    n_plots = len(all_markers_to_plot) + 1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Panel 1: Colored by cell type
    for cell_type in adata_organoid.obs['cell_type'].unique():
        mask = adata_organoid.obs['cell_type'] == cell_type
        color = cell_type_color_map[cell_type]
        axes[0].scatter(adata_organoid.obsm['X_umap'][mask, 0],
                        adata_organoid.obsm['X_umap'][mask, 1],
                        c=color, s=5, alpha=0.8, label=cell_type, edgecolors='none')

    axes[0].set_title(f'{organoid} - Cell Types', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Remaining panels: Marker genes
    for i, gene in enumerate(all_markers_to_plot, start=1):
        if i < len(axes):
            try:
                sc.pl.umap(adata_organoid, color=gene, ax=axes[i],
                           title=f'{organoid} - {gene}', show=False,
                           cmap='Reds', vmax='p95', use_raw=False)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            except Exception as e:
                axes[i].text(0.5, 0.5, f'{gene}\nerror', ha='center', va='center')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{organoid} - Cell Types and Markers', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'umap_{organoid}_with_markers.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: umap_{organoid}_with_markers.png")

# ============ CREATE LEGEND ONLY FIGURE ============
print("\n5. CREATING LEGEND FIGURE...")

fig, ax = plt.subplots(figsize=(8, len(unique_cell_types) * 0.5))
ax.axis('off')

# Create legend elements
legend_elements = []
for cell_type, color in cell_type_color_map.items():
    if cell_type in cell_type_colors:  # Only show defined types
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=15,
                                          label=cell_type))

ax.legend(handles=legend_elements, loc='center', frameon=True,
          fancybox=True, shadow=True, fontsize=12, ncol=2)

plt.title('Cell Type Legend', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('cell_type_legend.png', dpi=300, bbox_inches='tight')
plt.show()
print("  ✅ Saved: cell_type_legend.png")

print("\n" + "=" * 60)
print("CUSTOM UMAP GENERATION SUMMARY")
print("=" * 60)
print(f"\nOrganoids processed: {len(organoids)}")
print(f"Cell types identified: {len(unique_cell_types)}")
print("\nFiles created:")
for organoid in organoids:
    print(f"  - umap_{organoid}_colored.png")
    print(f"  - umap_{organoid}_with_markers.png")
print("  - umap_all_organoids_comparison.png")
print("  - cell_type_legend.png")
print("\n✅ All custom UMAPs generated successfully!")

# ============ COMPREHENSIVE ORGANOID SIMILARITY ANALYSIS ============
print("\n" + "=" * 60)
print("COMPREHENSIVE ORGANOID SIMILARITY ANALYSIS")
print("=" * 60)

# ============ 1. CELL TYPE COMPOSITION ============
print("\n1. CELL TYPE COMPOSITION BY ORGANOID")
print("-" * 40)

# Get cell type distribution for each organoid
cell_type_distributions = []

for organoid in organoids:
    mask = adata.obs['organoid'] == organoid
    cell_type_counts = adata.obs[mask]['cell_type'].value_counts()
    cell_type_pct = (cell_type_counts / cell_type_counts.sum() * 100).round(2)

    print(f"\n{organoid} Cell Type Composition:")
    dist_dict = {'organoid': organoid}
    for ct in cell_type_pct.index:
        dist_dict[ct] = cell_type_pct[ct]
        print(f"  {ct}: {cell_type_pct[ct]}%")
    cell_type_distributions.append(dist_dict)

# Create composition dataframe
comp_df = pd.DataFrame(cell_type_distributions).fillna(0)
print("\nCell Type Composition Matrix (%):")
print(comp_df.to_string(index=False))
comp_df.to_csv('organoid_cell_type_composition.csv', index=False)

# ============ 2. ALL MARKER GENE EXPRESSION ============
print("\n2. MARKER GENE EXPRESSION BY ORGANOID")
print("-" * 40)

# Define all marker gene categories
all_marker_categories = {
    'Motor Neuron': ['SLC5A7', 'BCL11B', 'CHAT', 'ISL1', 'MNX1'],
    'Cortical': ['SNAP25', 'RBFOX3', 'MAP2', 'TUBB3', 'DCX'],
    'Glutamatergic': ['GRIA2', 'SLC17A7', 'GRIN1', 'GRIK1'],
    'GABAergic': ['SLC32A1', 'GAD1', 'GAD2', 'PVALB', 'SST'],
    'Fibroblast': ['COL1A1', 'COL3A1', 'DCN', 'FAP', 'PDGFRA'],
    'Microglia': ['CSF1R', 'TREM2', 'AIF1', 'CX3CR1'],
    'Astrocyte': ['GFAP', 'S100B', 'AQP4', 'ALDH1L1'],
    'Oligodendrocyte': ['MBP', 'PLP1', 'MOG', 'OLIG2'],
    'Neural Progenitor': ['SOX2', 'PAX6', 'NES', 'MKI67'],
    'Synaptic': ['SYP', 'STX1A', 'VAMP2', 'DLG4']
}

# Collect expression data for all markers
marker_expression = []

for organoid in organoids:
    mask = adata.obs['organoid'] == organoid
    organoid_data = adata[mask, :]

    exp_dict = {'organoid': organoid}

    for category, markers in all_marker_categories.items():
        for marker in markers:
            if marker in organoid_data.var_names:
                marker_idx = list(organoid_data.var_names).index(marker)
                exp = organoid_data.X[:, marker_idx]
                if hasattr(exp, 'toarray'):
                    exp = exp.toarray().flatten()
                exp_dict[f'{category}_{marker}'] = exp.mean()
                exp_dict[f'{category}_{marker}_pct'] = (exp > 0).mean() * 100

    marker_expression.append(exp_dict)

marker_df = pd.DataFrame(marker_expression).fillna(0)
print(f"\nCollected expression data for {len(marker_df.columns) - 1} markers")
marker_df.to_csv('organoid_marker_expression.csv', index=False)

# ============ 3. GLOBAL GENE EXPRESSION CORRELATION ============
print("\n3. GLOBAL GENE EXPRESSION CORRELATION BETWEEN ORGANOIDS")
print("-" * 40)

# Get mean expression per gene for each organoid
organoid_expression_profiles = []

for organoid in organoids:
    mask = adata.obs['organoid'] == organoid
    organoid_data = adata[mask, :]

    # Calculate mean expression for all genes
    mean_exp = np.array(organoid_data.X.mean(axis=0)).flatten()
    organoid_expression_profiles.append(mean_exp)

# Calculate correlation matrix
expression_correlation = np.corrcoef(organoid_expression_profiles)
corr_df = pd.DataFrame(expression_correlation,
                       index=organoids,
                       columns=organoids)

print("\nGlobal Gene Expression Correlation Matrix:")
print(corr_df.round(3))

# ============ 4. CLUSTER SIMILARITY ============
print("\n4. CLUSTER COMPOSITION SIMILARITY")
print("-" * 40)

# Get cluster distribution for each organoid
cluster_distributions = []

for organoid in organoids:
    mask = adata.obs['organoid'] == organoid
    cluster_counts = adata.obs[mask]['leiden_neuronal'].value_counts()
    cluster_pct = (cluster_counts / cluster_counts.sum() * 100)

    dist_dict = {'organoid': organoid}
    for cluster in cluster_pct.index:
        dist_dict[f'cluster_{cluster}'] = cluster_pct[cluster]
    cluster_distributions.append(dist_dict)

cluster_df = pd.DataFrame(cluster_distributions).fillna(0)
print("\nCluster Distribution Matrix (%):")
print(cluster_df.round(2).to_string(index=False))

# ============ 5. MULTI-DIMENSIONAL SIMILARITY METRICS ============
print("\n5. MULTI-DIMENSIONAL SIMILARITY METRICS")
print("-" * 40)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# Combine all features for comprehensive comparison
# Features: cell type composition + marker expression + cluster distribution

# Prepare feature matrix
feature_dfs = []

# Cell type composition features
comp_features = comp_df.drop('organoid', axis=1)
feature_dfs.append(comp_features)

# Marker expression features (mean expression only, not percentages)
marker_mean_cols = [col for col in marker_df.columns if 'pct' not in col and col != 'organoid']
marker_features = marker_df[marker_mean_cols]
feature_dfs.append(marker_features)

# Cluster distribution features
cluster_features = cluster_df.drop('organoid', axis=1)
feature_dfs.append(cluster_features)

# Combine all features
all_features = pd.concat(feature_dfs, axis=1)
print(f"Total features for comparison: {all_features.shape[1]}")

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_features)

# Calculate different similarity metrics
euclidean_dist = euclidean_distances(features_scaled)
manhattan_dist = manhattan_distances(features_scaled)
cosine_sim = cosine_similarity(features_scaled)

# Create dataframes
euclidean_df = pd.DataFrame(euclidean_dist, index=organoids, columns=organoids)
manhattan_df = pd.DataFrame(manhattan_dist, index=organoids, columns=organoids)
cosine_df = pd.DataFrame(cosine_sim, index=organoids, columns=organoids)

print("\nEuclidean Distance (smaller = more similar):")
print(euclidean_df.round(3))

print("\nManhattan Distance (smaller = more similar):")
print(manhattan_df.round(3))

print("\nCosine Similarity (closer to 1 = more similar):")
print(cosine_df.round(3))

# ============ 6. HIERARCHICAL CLUSTERING ============
print("\n6. HIERARCHICAL CLUSTERING OF ORGANOIDS")
print("-" * 40)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Perform hierarchical clustering
linkage_matrix = linkage(features_scaled, method='ward')

# Plot dendrogram
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(linkage_matrix, labels=organoids, ax=ax, leaf_rotation=45)
ax.set_title('Hierarchical Clustering of Organoids (All Features)', fontsize=14, fontweight='bold')
ax.set_xlabel('Organoid')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig('organoid_comprehensive_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

# Determine clusters at different thresholds
for threshold in [5, 10, 15]:
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    print(f"\nAt distance threshold {threshold}: {len(set(clusters))} groups")
    for i, org in enumerate(organoids):
        print(f"  {org}: Group {clusters[i]}")

# ============ 7. PCA VISUALIZATION ============
print("\n7. PCA OF ORGANOIDS (All Features)")
print("-" * 40)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                     c=range(len(organoids)), s=300, cmap='Set1', alpha=0.7)

# Add labels
for i, organoid in enumerate(organoids):
    ax.annotate(organoid, (pca_result[i, 0], pca_result[i, 1]),
                fontsize=12, fontweight='bold', ha='center')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
ax.set_title('PCA of Organoids Based on All Features', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('organoid_comprehensive_pca.png', dpi=150, bbox_inches='tight')
plt.show()

print(
    f"\nExplained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

# ============ 8. SIMILARITY NETWORK ============
print("\n8. ORGANOID SIMILARITY NETWORK")
print("-" * 40)

import networkx as nx

# Create similarity graph (threshold for edges)
threshold = np.percentile(euclidean_dist[euclidean_dist > 0], 25)  # Top 25% most similar
G = nx.Graph()

# Add nodes
for i, org in enumerate(organoids):
    G.add_node(org, size=adata.obs[adata.obs['organoid'] == org].shape[0])

# Add edges for similar organoids
for i in range(len(organoids)):
    for j in range(i + 1, len(organoids)):
        if euclidean_dist[i, j] < threshold:
            G.add_edge(organoids[i], organoids[j], weight=1 / euclidean_dist[i, j])

# Draw network
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
node_sizes = [G.nodes[org]['size'] / 50 for org in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                       node_size=node_sizes, alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

ax.set_title('Organoid Similarity Network', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('organoid_similarity_network.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ 9. RANK SIMILARITY ============
print("\n9. ORGANOID SIMILARITY RANKING")
print("-" * 40)

# Create a combined similarity score (average of normalized metrics)
# Normalize distances (0-1, where 0 = most similar)
norm_euclidean = euclidean_dist / euclidean_dist.max()
norm_manhattan = manhattan_dist / manhattan_dist.max()
norm_cosine = 1 - cosine_sim  # Convert to distance

combined_dist = (norm_euclidean + norm_manhattan + norm_cosine) / 3

# Set diagonal to infinity
np.fill_diagonal(combined_dist, np.inf)

# Find all pairs and their distances
pairs = []
for i in range(len(organoids)):
    for j in range(i + 1, len(organoids)):
        pairs.append({
            'pair': f"{organoids[i]} - {organoids[j]}",
            'distance': combined_dist[i, j],
            'org1': organoids[i],
            'org2': organoids[j]
        })

pairs_df = pd.DataFrame(pairs)
pairs_df = pairs_df.sort_values('distance')

print("\nOrganoid Similarity Ranking (most similar to least):")
for idx, row in pairs_df.iterrows():
    print(f"  {row['pair']}: similarity score = {row['distance']:.3f}")

# ============ 10. FINAL ANSWER ============
print("\n" + "=" * 60)
print("FINAL ANSWER: WHICH ORGANOIDS ARE MOST SIMILAR?")
print("=" * 60)

most_similar = pairs_df.iloc[0]
second_similar = pairs_df.iloc[1]
least_similar = pairs_df.iloc[-1]

print(f"\n✅ MOST SIMILAR: {most_similar['pair']}")
print(f"   Similarity score: {most_similar['distance']:.3f} (lower = more similar)")
print(f"\n   {most_similar['org1']} vs {most_similar['org2']}:")

# Show comparison for the most similar pair
org1_data = comp_df[comp_df['organoid'] == most_similar['org1']].iloc[0]
org2_data = comp_df[comp_df['organoid'] == most_similar['org2']].iloc[0]

print("\n   Cell Type Comparison:")
for ct in comp_df.columns[1:]:
    val1 = org1_data[ct]
    val2 = org2_data[ct]
    diff = abs(val1 - val2)
    print(f"     {ct}: {val1:.1f}% vs {val2:.1f}% (diff: {diff:.1f}%)")

print(f"\n\n📊 SECOND MOST SIMILAR: {second_similar['pair']}")
print(f"   Similarity score: {second_similar['distance']:.3f}")

print(f"\n📉 LEAST SIMILAR: {least_similar['pair']}")
print(f"   Similarity score: {least_similar['distance']:.3f}")

# ============ 11. VISUALIZATION OF RESULTS ============
print("\n11. VISUALIZING SIMILARITY RESULTS")
print("-" * 40)

# Create a comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Panel 1: Cell type composition heatmap
ax1 = plt.subplot(2, 3, 1)
cell_type_data = comp_df.set_index('organoid')
im1 = ax1.imshow(cell_type_data.T, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(len(cell_type_data.index)))
ax1.set_xticklabels(cell_type_data.index, rotation=45, ha='right')
ax1.set_yticks(range(len(cell_type_data.columns)))
ax1.set_yticklabels(cell_type_data.columns, fontsize=8)
ax1.set_title('Cell Type Composition')
plt.colorbar(im1, ax=ax1)

# Panel 2: PCA plot
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(pca_result[:, 0], pca_result[:, 1],
            c=range(len(organoids)), s=200, cmap='Set1', alpha=0.7)
for i, org in enumerate(organoids):
    ax2.annotate(org, (pca_result[i, 0], pca_result[i, 1]),
                 fontsize=10, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
ax2.set_title('PCA of Organoids')
ax2.grid(True, alpha=0.3)

# Panel 3: Similarity matrix
ax3 = plt.subplot(2, 3, 3)
im3 = ax3.imshow(combined_dist, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=1)
ax3.set_xticks(range(len(organoids)))
ax3.set_yticks(range(len(organoids)))
ax3.set_xticklabels(organoids, rotation=45, ha='right')
ax3.set_yticklabels(organoids)
ax3.set_title('Combined Similarity Matrix\n(darker = more similar)')
plt.colorbar(im3, ax=ax3)

# Panel 4: Dendrogram
ax4 = plt.subplot(2, 3, 4)
dendrogram(linkage_matrix, labels=organoids, ax=ax4, leaf_rotation=45)
ax4.set_title('Hierarchical Clustering')
ax4.set_xlabel('Organoid')
ax4.set_ylabel('Distance')

# Panel 5: Marker expression heatmap (top markers)
ax5 = plt.subplot(2, 3, 5)
top_markers = ['SLC5A7', 'BCL11B', 'SNAP25', 'GRIA2', 'SLC32A1',
               'COL1A1', 'COL3A1', 'DCN', 'CSF1R', 'TREM2']
marker_heatmap_data = []
for marker in top_markers:
    if f'Motor Neuron_{marker}' in marker_df.columns:
        marker_heatmap_data.append(marker_df[f'Motor Neuron_{marker}'].values)
    elif f'Cortical_{marker}' in marker_df.columns:
        marker_heatmap_data.append(marker_df[f'Cortical_{marker}'].values)
    else:
        marker_heatmap_data.append([0, 0, 0, 0])

im5 = ax5.imshow(marker_heatmap_data, cmap='viridis', aspect='auto')
ax5.set_xticks(range(len(organoids)))
ax5.set_xticklabels(organoids, rotation=45, ha='right')
ax5.set_yticks(range(len(top_markers)))
ax5.set_yticklabels(top_markers)
ax5.set_title('Key Marker Expression')
plt.colorbar(im5, ax=ax5)

# Panel 6: Similarity ranking
ax6 = plt.subplot(2, 3, 6)
colors = ['green' if i == 0 else 'lightgreen' if i == 1 else 'lightcoral' if i == len(pairs_df) - 1 else 'lightgray'
          for i in range(len(pairs_df))]
bars = ax6.barh(range(len(pairs_df)), pairs_df['distance'].values, color=colors)
ax6.set_yticks(range(len(pairs_df)))
ax6.set_yticklabels(pairs_df['pair'].values, fontsize=9)
ax6.set_xlabel('Similarity Distance (lower = more similar)')
ax6.set_title('Organoid Similarity Ranking')
ax6.invert_xaxis()  # Most similar at top

plt.tight_layout()
plt.savefig('organoid_comprehensive_similarity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Comprehensive analysis complete!")
print("Files saved:")
print("  - organoid_cell_type_composition.csv")
print("  - organoid_marker_expression.csv")
print("  - organoid_comprehensive_clustering.png")
print("  - organoid_comprehensive_pca.png")
print("  - organoid_similarity_network.png")
print("  - organoid_comprehensive_similarity.png")
# ============ NUMPY COMPATIBILITY PATCH ============
# This must be before any other imports
import numpy as np
import sys
import warnings

# Patch numpy to provide deprecated attributes
if not hasattr(np, 'float_'):
    np.float_ = np.float64
    print("✓ Patched numpy: added np.float_ alias for np.float64")

if not hasattr(np, 'int_'):
    np.int_ = np.int64
    print("✓ Patched numpy: added np.int_ alias for np.int64")

# Suppress warnings
warnings.filterwarnings('ignore')

# Now import other packages
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import NearestNeighbors
import networkx as nx

# Try to import optional packages
try:
    import squidpy as sq
    SQUIDPY_AVAILABLE = True
except ImportError:
    SQUIDPY_AVAILABLE = False
    print("Note: Install squidpy for spatial analysis: pip install squidpy")

try:
    import mygene
    MGENE_AVAILABLE = True
except ImportError:
    MGENE_AVAILABLE = False
    print("Note: Install 'mygene' for gene descriptions: pip install mygene")

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("Note: Install 'gseapy' for pathway enrichment: pip install gseapy")

try:
    import harmonypy
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    print("Note: Install 'harmonypy' for batch correction: pip install harmonypy")

# ============ READ DATA ============
print("\n" + "=" * 60)
print("STEP 1: READING DATA")
print("=" * 60)

adata = sc.read_h5ad("C06018D5.bin50_1.0.h5ad")
print("Original data:")
print(adata)

original_gene_names = adata.var_names.copy()
print(f"First few gene names: {original_gene_names[:5]}")

if 'real_gene_name' in adata.var.columns:
    print("\nGene symbols available in var['real_gene_name']")
    print(adata.var[['real_gene_name']].head())
    missing_symbols = adata.var['real_gene_name'].isna().sum()
    print(f"Missing gene symbols: {missing_symbols}")

# ============ BASIC FILTERING ============
print("\n" + "=" * 60)
print("STEP 2: BASIC FILTERING")
print("=" * 60)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

adata = adata[adata.obs.pct_counts_mt < 10, :].copy()
print(f"\nAfter MT filtering: {adata.shape}")

# ============ NORMALIZATION ============
print("\n" + "=" * 60)
print("STEP 3: NORMALIZATION")
print("=" * 60)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ============ HIGHLY VARIABLE GENES ============
print("\n" + "=" * 60)
print("STEP 4: HIGHLY VARIABLE GENES")
print("=" * 60)

sc.pp.highly_variable_genes(
    adata, flavor='seurat', n_top_genes=8000,
    min_mean=0.0125, max_mean=3, min_disp=0.5, inplace=True
)

print("Highly variable genes column created:", 'highly_variable' in adata.var.columns)
print(f"Gene names before HVG subset: {adata.var_names[:5]}")

adata = adata[:, adata.var.highly_variable].copy()
print(f"After HVG selection: {adata.shape}")
print(f"Gene names after HVG subset: {adata.var_names[:5]}")

sc.pl.highly_variable_genes(adata)

# ============ DIMENSIONALITY REDUCTION ============
print("\n" + "=" * 60)
print("STEP 5: DIMENSIONALITY REDUCTION")
print("=" * 60)

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
sc.pl.pca_variance_ratio(adata, log=True)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata)

# ============ CLUSTERING ============
print("\n" + "=" * 60)
print("STEP 6: CLUSTERING")
print("=" * 60)

sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color="leiden")

# ============ SPATIAL VISUALIZATION ============
print("\n" + "=" * 60)
print("STEP 7: SPATIAL ANALYSIS")
print("=" * 60)

if 'spatial' in adata.obsm:
    print("Spatial coordinates found in adata.obsm['spatial']")
    spatial_coords = adata.obsm['spatial']
    print(f"Spatial coordinates shape: {spatial_coords.shape}")

    print("\n1. Creating custom spatial cluster map...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                               c=adata.obs['leiden'].astype('category').cat.codes,
                               cmap='tab10', s=5, alpha=0.7)
    axes[0].set_title('Spatial Clusters')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    axes[0].set_aspect('equal')
    legend1 = axes[0].legend(*scatter1.legend_elements(), title="Cluster",
                             bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].add_artist(legend1)

    if 'orig.ident' in adata.obs.columns:
        unique_organoids = adata.obs['orig.ident'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_organoids)))
        for i, organoid in enumerate(unique_organoids):
            mask = adata.obs['orig.ident'] == organoid
            axes[1].scatter(spatial_coords[mask, 0], spatial_coords[mask, 1],
                            c=[colors[i]], s=5, alpha=0.7, label=organoid)
        axes[1].set_title('Organoid Identity')
        axes[1].set_xlabel('X coordinate')
        axes[1].set_ylabel('Y coordinate')
        axes[1].set_aspect('equal')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('spatial_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()

    if 'orig.ident' in adata.obs.columns:
        print("\n2. Cluster composition by organoid:")
        cluster_by_organoid = pd.crosstab(adata.obs["leiden"], adata.obs["orig.ident"])
        print(cluster_by_organoid)
        cluster_by_organoid.to_csv("cluster_by_organoid.csv")

        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_by_organoid.T.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_title('Cluster Composition by Organoid')
        ax.set_xlabel('Organoid')
        ax.set_ylabel('Number of Spots')
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('cluster_composition.png', dpi=150, bbox_inches='tight')
        plt.show()

    print("\n3. Calculating spatial metrics for each cluster...")
    spatial_metrics = []
    for cluster in adata.obs['leiden'].unique():
        mask = adata.obs['leiden'] == cluster
        coords = spatial_coords[mask]
        metrics = {'cluster': cluster, 'n_spots': mask.sum()}
        if len(coords) > 3:
            try:
                hull = ConvexHull(coords)
                metrics['area_pixels'] = hull.volume
                metrics['centroid_x'] = coords[:, 0].mean()
                metrics['centroid_y'] = coords[:, 1].mean()
                metrics['spread_x'] = coords[:, 0].std()
                metrics['spread_y'] = coords[:, 1].std()
            except:
                metrics['area_pixels'] = np.nan
                metrics['centroid_x'] = coords[:, 0].mean()
                metrics['centroid_y'] = coords[:, 1].mean()
                metrics['spread_x'] = coords[:, 0].std()
                metrics['spread_y'] = coords[:, 1].std()
        else:
            metrics['area_pixels'] = np.nan
            metrics['centroid_x'] = coords[:, 0].mean() if len(coords) > 0 else np.nan
            metrics['centroid_y'] = coords[:, 1].mean() if len(coords) > 0 else np.nan
            metrics['spread_x'] = coords[:, 0].std() if len(coords) > 1 else 0
            metrics['spread_y'] = coords[:, 1].std() if len(coords) > 1 else 0
        spatial_metrics.append(metrics)
        print(f"\nCluster {cluster}:")
        print(f"  Number of spots: {metrics['n_spots']}")
        print(f"  Centroid: ({metrics['centroid_x']:.1f}, {metrics['centroid_y']:.1f})")
        if not np.isnan(metrics['area_pixels']):
            print(f"  Area: {metrics['area_pixels']:.1f} pixels²")

    spatial_metrics_df = pd.DataFrame(spatial_metrics)
    spatial_metrics_df.to_csv("spatial_metrics.csv", index=False)

    print("\n4. Plotting cluster distribution...")
    unique_clusters = adata.obs['leiden'].unique()
    n_clusters = len(unique_clusters)
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, cluster in enumerate(unique_clusters):
        mask = adata.obs['leiden'] == cluster
        coords = spatial_coords[mask]
        axes[i].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c='lightgray', s=1, alpha=0.3)
        axes[i].scatter(coords[:, 0], coords[:, 1], c='red', s=2, alpha=0.5)
        axes[i].set_title(f'Cluster {cluster} Distribution')
        axes[i].set_aspect('equal')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n5. Analyzing radial organization...")
    center = spatial_coords.mean(axis=0)
    distances = cdist([center], spatial_coords)[0]
    adata.obs['distance_from_center'] = distances
    adata.obs['spatial_zone'] = pd.cut(distances, bins=5,
                                       labels=['core', 'inner', 'mid', 'outer', 'periphery'])

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in adata.obs['leiden'].unique():
        mask = adata.obs['leiden'] == cluster
        ax.hist(adata.obs.loc[mask, 'distance_from_center'], bins=50, alpha=0.5, label=f'Cluster {cluster}')
    ax.set_xlabel('Distance from Tissue Center')
    ax.set_ylabel('Number of Spots')
    ax.set_title('Spatial Distribution by Distance from Center')
    ax.legend()
    plt.savefig('radial_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n6. Finding genes enriched in spatial zones...")
    sc.tl.rank_genes_groups(adata, 'spatial_zone', method='wilcoxon', use_raw=False)
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, title='Spatial Zone Markers')

else:
    print("No spatial coordinates found in adata.obsm['spatial']")
    print(f"Available obsm keys: {list(adata.obsm.keys())}")

# ============ MARKER GENE IDENTIFICATION ============
print("\n" + "=" * 60)
print("STEP 8: MARKER GENE ANALYSIS")
print("=" * 60)

print("\nRunning marker gene identification...")
if 'log1p' in adata.layers:
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", layer='log1p', use_raw=False)
else:
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=False)

print("\nTop marker genes per cluster:")
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names

all_markers = {}
for group in groups:
    genes = result['names'][group][:50]
    logfcs = result['logfoldchanges'][group][:50]
    scores = result['scores'][group][:50]
    all_markers[group] = {'genes': genes, 'logfcs': logfcs, 'scores': scores}
    print(f"\nCluster {group}:")
    for gene, score, logfc in zip(genes[:10], scores[:10], logfcs[:10]):
        print(f"  {gene}: score={score:.3f}, log2FC={logfc:.3f}")

marker_genes = pd.DataFrame({
    group + '_' + key: result[key][group]
    for group in groups for key in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
})
marker_genes.to_csv("marker_genes.csv")
print("\nMarker genes saved to marker_genes.csv")

sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)

print("\nCreating dotplot of top markers...")
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, title="Top Marker Genes per Cluster")

# ============ CONVERT GENE SYMBOLS ============
print("\n" + "=" * 60)
print("STEP 9: CONVERTING GENE SYMBOLS")
print("=" * 60)

if 'real_gene_name' in adata.var.columns:
    print("\nConverting ENSG IDs to gene symbols...")
    adata.var['ensg_id'] = adata.var_names.copy()
    adata.var_names = adata.var["real_gene_name"].values
    print(f"Gene names converted to symbols: {adata.var_names[:10]}")

# Re-run after symbol conversion so enrichment uses gene symbols
print("\nRe-running marker gene identification with gene symbols...")
if 'log1p' in adata.layers:
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", layer='log1p', use_raw=False)
else:
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=False)

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names

# ============ CELL TYPE MARKER ANALYSIS ============
print("\n" + "=" * 60)
print("STEP 10: CELL TYPE MARKER ANALYSIS")
print("=" * 60)

cell_type_markers = {
    'Neurons': ['SNAP25', 'SYT1', 'STMN2', 'RBFOX3', 'MAP2', 'TUBB3', 'DCX'],
    'Astrocytes': ['GFAP', 'S100B', 'AQP4', 'ALDH1L1', 'SOX9'],
    'Oligodendrocytes': ['MBP', 'PLP1', 'MOG', 'OLIG2', 'SOX10'],
    'Microglia': ['CSF1R', 'C1QB', 'PTPRC', 'CX3CR1', 'TREM2', 'AIF1'],
    'Fibroblasts': ['COL1A1', 'COL3A1', 'DCN', 'FAP', 'PDGFRA'],
    'Endothelial': ['PECAM1', 'CLDN5', 'FLT1', 'VWF', 'CDH5'],
    'Neural Progenitors': ['SOX2', 'PAX6', 'NES', 'MKI67', 'HES1'],
    'Radial Glia': ['FABP7', 'VIM', 'GLI3']
}

print("\nChecking cell type marker expression:")
cell_type_scores = {}

for cell_type, markers in cell_type_markers.items():
    available = [m for m in markers if m in adata.var_names]
    if len(available) >= 2:
        print(f"\n{cell_type}: {len(available)}/{len(markers)} markers found: {available}")
        try:
            sc.pl.matrixplot(adata, available, groupby='leiden', title=f'{cell_type} Markers',
                             cmap='Blues', standard_scale='var',
                             save=f'_{cell_type}_markers.png', show=False, use_raw=False)
        except:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                sc.pl.dotplot(adata, available, groupby='leiden',
                              title=f'{cell_type} Markers', ax=ax, show=False, use_raw=False)
                plt.savefig(f'dotplot_{cell_type}_markers.png', dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Could not create plot for {cell_type}: {e}")

        for cluster in groups:
            mask = adata.obs['leiden'] == cluster
            cluster_exp = []
            for marker in available:
                if marker in adata.var_names:
                    marker_idx = list(adata.var_names).index(marker)
                    exp_values = adata[mask, marker_idx].X
                    if hasattr(exp_values, 'toarray'):
                        exp_values = exp_values.toarray().flatten()
                    cluster_exp.append(exp_values.mean())
            if cluster_exp:
                mean_exp = np.mean(cluster_exp)
                if cell_type not in cell_type_scores:
                    cell_type_scores[cell_type] = {}
                cell_type_scores[cell_type][cluster] = mean_exp

if cell_type_scores:
    cell_type_df = pd.DataFrame(cell_type_scores).round(3)
    cell_type_df.to_csv('cell_type_scores.csv')
    print("\nCell type scores saved to cell_type_scores.csv")
    print("\nCell type scores by cluster:")
    print(cell_type_df)
else:
    print("\nNo cell type scores could be calculated.")

# ============ PATHWAY ENRICHMENT ============
print("\n" + "=" * 60)
print("STEP 11: PATHWAY ENRICHMENT ANALYSIS")
print("=" * 60)

if GSEAPY_AVAILABLE:
    for cluster in groups:
        print(f"\nRunning enrichment for Cluster {cluster}...")
        genes = result['names'][cluster][:100]
        try:
            enr = gp.enrichr(gene_list=genes.tolist(),
                             gene_sets=['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022'],
                             organism='human', outdir=f'enrichment_cluster_{cluster}')
            print(f"\nCluster {cluster} top pathways:")
            print(enr.results.head(10)[['Term', 'Adjusted P-value']])
            enr.results.to_csv(f'enrichment_cluster_{cluster}.csv', index=False)
        except Exception as e:
            print(f"Enrichment failed for cluster {cluster}: {e}")
else:
    print("Skipping pathway enrichment - install gseapy: pip install gseapy")

# ============ ADDING GENE DESCRIPTIONS ============
print("\n" + "=" * 60)
print("STEP 12: ADDING GENE DESCRIPTIONS")
print("=" * 60)

if MGENE_AVAILABLE:
    print("\nFetching gene descriptions for top markers...")
    all_top_genes = set()
    for cluster in groups:
        cluster_genes = result['names'][cluster][:20]
        all_top_genes.update(cluster_genes)

    all_top_genes = [g for g in all_top_genes if not pd.isna(g) and g != '']
    print(f"Querying {len(all_top_genes)} gene symbols...")

    try:
        mg = mygene.MyGeneInfo()
        gene_info = mg.querymany(all_top_genes, scopes='symbol',
                                 fields='name,summary,entrezgene', species='human')
        gene_info = [g for g in gene_info if 'notfound' not in g]

        if gene_info:
            gene_info_df = pd.DataFrame(gene_info)
            available_cols = ['query']
            for col in ['name', 'summary', 'entrezgene']:
                if col in gene_info_df.columns:
                    available_cols.append(col)
            gene_info_df = gene_info_df[available_cols]
            gene_info_df.columns = ['gene'] + [c for c in available_cols[1:]]

            gene_cluster = []
            for gene in gene_info_df['gene']:
                found = False
                for cluster in groups:
                    cluster_genes = result['names'][cluster][:20]
                    if gene in cluster_genes:
                        gene_cluster.append(cluster)
                        found = True
                        break
                if not found:
                    gene_cluster.append('unknown')

            gene_info_df['top_marker_in_cluster'] = gene_cluster
            gene_info_df.to_csv('gene_descriptions.csv', index=False)
            print(f"Gene descriptions saved to gene_descriptions.csv")
            print(gene_info_df.head())
        else:
            print("No gene descriptions found")
    except Exception as e:
        print(f"Error fetching gene descriptions: {e}")
else:
    print("Skipping gene descriptions - install mygene: pip install mygene")

# ============ CREATE PUBLICATION FIGURES ============
print("\n" + "=" * 60)
print("STEP 13: CREATING PUBLICATION FIGURES")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))

ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                      c=adata.obs['leiden'].astype('category').cat.codes,
                      cmap='tab10', s=3, alpha=0.7)
ax1.set_title('A: Spatial Clusters', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
sc.pl.umap(adata, color='leiden', ax=ax2, show=False, title='B: UMAP Clusters')
ax2.set_title('B: UMAP Clusters', fontsize=14, fontweight='bold')

ax3 = plt.subplot(2, 3, 3)
# rank_genes_groups_dotplot does not support ax= embedding — draw manually
_n_top = 3
_bar_genes, _bar_scores, _bar_labels = [], [], []
_colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(groups)))
for _ci, _group in enumerate(groups):
    for _rank in range(_n_top):
        _bar_genes.append(result['names'][_group][_rank])
        _bar_scores.append(result['scores'][_group][_rank])
        _bar_labels.append(f"C{_group}")
_y_pos = np.arange(len(_bar_genes))
_bar_colors = [_colors_cycle[list(groups).index(_lbl[1:])] for _lbl in _bar_labels]
ax3.barh(_y_pos, _bar_scores, color=_bar_colors, alpha=0.8)
ax3.set_yticks(_y_pos)
ax3.set_yticklabels(_bar_genes, fontsize=8)
ax3.set_xlabel('Wilcoxon score')
ax3.set_title('C: Top Marker Genes', fontsize=14, fontweight='bold')
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color=_colors_cycle[i], label=f'Cluster {g}') for i, g in enumerate(groups)],
           fontsize=7, loc='lower right')

ax4 = plt.subplot(2, 3, 4)
representative_markers = [result['names'][cluster][0] for cluster in groups[:3]]
if len(representative_markers) >= 3:
    colors_rgb = ['red', 'green', 'blue']
    rgb_image = np.zeros((spatial_coords.shape[0], 3))
    for i, (marker, color) in enumerate(zip(representative_markers[:3], colors_rgb)):
        if marker in adata.var_names:
            marker_idx = list(adata.var_names).index(marker)
            exp = adata.X[:, marker_idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            exp = (exp - exp.min()) / (exp.max() - exp.min() + 1e-8)
            if color == 'red':
                rgb_image[:, 0] = exp
            elif color == 'green':
                rgb_image[:, 1] = exp
            elif color == 'blue':
                rgb_image[:, 2] = exp
    ax4.scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=rgb_image, s=3, alpha=0.7)
    ax4.set_title('D: Marker Overlay', fontsize=14, fontweight='bold')
    ax4.set_aspect('equal')
    ax4.axis('off')

ax5 = plt.subplot(2, 3, 5)
if cell_type_scores:
    cell_type_df.T.plot(kind='bar', ax=ax5, legend=False)
    ax5.set_title('E: Cell Type Scores', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Cell Type')
    ax5.set_ylabel('Mean Expression')

ax6 = plt.subplot(2, 3, 6)
if 'cluster_by_organoid' in locals():
    cluster_by_organoid.T.plot(kind='bar', stacked=True, ax=ax6, colormap='tab10')
    ax6.set_title('F: Cluster Composition', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Organoid')
    ax6.set_ylabel('Number of Spots')
    ax6.legend(title='Cluster', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Publication figure saved to publication_figure.png")

# ============ SPATIAL EXPRESSION OF KEY MARKERS ============
print("\n" + "=" * 60)
print("STEP 14: SPATIAL EXPRESSION OF KEY MARKERS")
print("=" * 60)

neural_markers = ["SOX2", "PAX6", "MAP2", "TUBB3", "MKI67", "GFAP",
                  "NES", "DCX", "S100B", "OLIG2", "EMX1", "DLX2",
                  "CSF1R", "C1QB", "COL1A1", "COL3A1", "DCN", "SNAP25", "SYT1"]

available_markers = [m for m in neural_markers if m in adata.var_names]
print(f"\nAvailable neural markers: {available_markers}")

if available_markers and 'spatial' in adata.obsm:
    print("\nPlotting spatial expression of neural markers...")
    spatial_coords = adata.obsm['spatial']
    n_markers = len(available_markers)

    if n_markers == 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        axes = [ax]
    else:
        n_cols = 3
        n_rows = (n_markers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

    for i, marker in enumerate(available_markers):
        marker_idx = list(adata.var_names).index(marker)
        marker_exp = adata.X[:, marker_idx]
        if hasattr(marker_exp, 'toarray'):
            marker_exp = marker_exp.toarray().flatten()
        marker_exp_norm = (marker_exp - marker_exp.min()) / (marker_exp.max() - marker_exp.min() + 1e-8)
        scatter = axes[i].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                                  c=marker_exp_norm, cmap='viridis', s=5, alpha=0.7, vmin=0, vmax=1)
        axes[i].set_title(f'{marker} Expression')
        axes[i].set_aspect('equal')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(scatter, ax=axes[i])

    if n_markers > 1:
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('spatial_marker_expression.png', dpi=150, bbox_inches='tight')
    plt.show()

    if 'COL1A1' in available_markers:
        print("\nCreating detailed COL1A1 expression plot...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        marker_idx = list(adata.var_names).index('COL1A1')
        marker_exp = adata.X[:, marker_idx]
        if hasattr(marker_exp, 'toarray'):
            marker_exp = marker_exp.toarray().flatten()

        scatter = ax1.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                              c=marker_exp, cmap='Reds', s=5, alpha=0.7)
        ax1.set_title('COL1A1 Expression (Fibroblast Marker)')
        ax1.set_aspect('equal')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(scatter, ax=ax1)

        scatter2 = ax2.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                               c=adata.obs['leiden'].astype('category').cat.codes,
                               cmap='tab10', s=3, alpha=0.5)
        ax2.set_title('Clusters with COL1A1+ regions')
        ax2.set_aspect('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])

        high_exp_threshold = np.percentile(marker_exp, 90)
        high_exp_mask = marker_exp > high_exp_threshold
        ax2.scatter(spatial_coords[high_exp_mask, 0], spatial_coords[high_exp_mask, 1],
                    c='red', s=10, alpha=0.8, label='COL1A1 high (>90th %ile)')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('COL1A1_detailed.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\nCOL1A1 Expression Statistics:")
        print(f"  Mean expression: {marker_exp.mean():.4f}")
        print(f"  Max expression: {marker_exp.max():.4f}")
        print(f"  % spots expressing: {(marker_exp > 0).mean() * 100:.1f}%")

        print("\n  COL1A1 expression by cluster:")
        for cluster in groups:
            mask = adata.obs['leiden'] == cluster
            cluster_exp = marker_exp[mask]
            print(f"    Cluster {cluster}: mean={cluster_exp.mean():.4f}, "
                  f"% expressing={(cluster_exp > 0).mean() * 100:.1f}%")

# ============ ADDITIONAL VISUALIZATIONS ============
print("\n" + "=" * 60)
print("STEP 15: ADDITIONAL VISUALIZATIONS")
print("=" * 60)

print("\nCreating heatmap of top markers...")
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, use_raw=False, show=False)

print("\nCreating violin plots for top markers...")
top_genes_per_cluster = [result['names'][group][0] for group in groups[:3]]
sc.pl.violin(adata, keys=top_genes_per_cluster, groupby='leiden', rotation=45, use_raw=False)

# ============ SAVE PROCESSED DATA ============
print("\n" + "=" * 60)
print("STEP 16: SAVING RESULTS")
print("=" * 60)

print("\nConverting string columns for saving...")
for col in adata.obs.columns:
    if hasattr(adata.obs[col], 'dtype') and pd.api.types.is_string_dtype(adata.obs[col]):
        adata.obs[col] = adata.obs[col].astype('object')
        print(f"  Converted obs.{col}")

for col in adata.var.columns:
    if hasattr(adata.var[col], 'dtype') and pd.api.types.is_string_dtype(adata.var[col]):
        adata.var[col] = adata.var[col].astype('object')
        print(f"  Converted var.{col}")

if hasattr(adata.obs.index, 'dtype') and pd.api.types.is_string_dtype(adata.obs.index):
    adata.obs.index = adata.obs.index.astype('object')
    print("  Converted obs.index")

if hasattr(adata.var.index, 'dtype') and pd.api.types.is_string_dtype(adata.var.index):
    adata.var.index = adata.var.index.astype('object')
    print("  Converted var.index")

for key in list(adata.uns.keys()):
    if isinstance(adata.uns[key], pd.DataFrame):
        print(f"  Checking uns['{key}'] DataFrame")
        for col in adata.uns[key].columns:
            if hasattr(adata.uns[key][col], 'dtype') and pd.api.types.is_string_dtype(adata.uns[key][col]):
                adata.uns[key][col] = adata.uns[key][col].astype('object')
                print(f"    Converted uns['{key}'].{col}")

print("\nSaving data...")
adata.write("processed_spatial_data.h5ad")
print("Analysis complete and data saved to 'processed_spatial_data.h5ad'!")

print("\nVerifying saved data...")
test_adata = sc.read_h5ad("processed_spatial_data.h5ad", backed='r')
print(f"Gene names in saved file: {test_adata.var_names[:10]}")
test_adata.file.close()

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Final dataset: {adata.shape[0]} spots × {adata.shape[1]} genes")
print(f"Number of clusters: {len(groups)}")
print(f"Spatial coordinates: {'✓' if 'spatial' in adata.obsm else '✗'}")
print(f"Organoid identities: {'✓' if 'orig.ident' in adata.obs.columns else '✗'}")
print(f"Gene symbols converted: {'✓' if 'real_gene_name' in adata.var.columns else '✗'}")

print("\n" + "=" * 60)
print("FILES GENERATED")
print("=" * 60)
print("  - processed_spatial_data.h5ad")
print("  - marker_genes.csv")
if MGENE_AVAILABLE:
    print("  - gene_descriptions.csv")
print("  - cell_type_scores.csv")
if GSEAPY_AVAILABLE:
    print("  - enrichment_cluster_*.csv")

# ============ STEP 17: SPATIAL DOMAIN DETECTION WITH SpaGCN ============
print("\n" + "=" * 60)
print("STEP 17: SPATIAL DOMAIN DETECTION WITH SpaGCN")
print("=" * 60)

try:
    import SpaGCN as spg

    print("\nPreparing data for SpaGCN...")
    x_array = adata.obsm['spatial'][:, 0]
    y_array = adata.obsm['spatial'][:, 1]

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    print("\nCalculating spatial graph...")
    adj = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)

    # FIX: search_l takes probability p and adj matrix only
    print("Searching for optimal l parameter...")
    l = spg.search_l(p=0.5, adj=adj, start=0.01, end=1000, tol=0.01)
    print(f"Optimal l = {l}")

    # FIX: removed invalid 'seed' keyword argument
    print("Searching for optimal resolution...")
    res = spg.search_res(adata, adj, l, target_num=5, start=0.1, step=0.1,
                         tol=5e-3, lr=0.05, max_epochs=20)
    print(f"Optimal resolution = {res}")

    print(f"Running SpaGCN with l={l}, res={res}...")
    clf = spg.SpaGCN()
    clf.set_library_size(library_size=adata.obs['total_counts'].values)
    clf.train(X, adj, init_spa=True, init=None, res=res, l=l)

    y_pred = clf.predict()
    adata.obs['spatial_domain'] = y_pred.astype(str)
    print(f"Detected {len(np.unique(y_pred))} spatial domains")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(x_array, y_array,
                    c=adata.obs['spatial_domain'].astype('category').cat.codes,
                    cmap='tab10', s=5, alpha=0.7)
    axes[0].set_title('SpaGCN Spatial Domains')
    axes[0].set_aspect('equal')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(x_array, y_array,
                    c=adata.obs['leiden'].astype('category').cat.codes,
                    cmap='tab10', s=5, alpha=0.7)
    axes[1].set_title('Leiden Clusters (Transcriptomic)')
    axes[1].set_aspect('equal')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig('spatial_domains_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    cross_tab = pd.crosstab(adata.obs['leiden'], adata.obs['spatial_domain'])
    print(cross_tab)
    cross_tab.to_csv('leiden_vs_spatial_domains.csv')

    sc.tl.rank_genes_groups(adata, 'spatial_domain', method='wilcoxon', use_raw=False)
    sc.pl.rank_genes_groups_dotplot(adata, groupby='spatial_domain',
                                    n_genes=5, title='Spatial Domain Markers')

    result_spatial = adata.uns['rank_genes_groups']
    spatial_markers = pd.DataFrame({
        group + '_' + key: result_spatial[key][group]
        for group in result_spatial['names'].dtype.names
        for key in ['names', 'scores', 'pvals_adj', 'logfoldchanges']
    })
    spatial_markers.to_csv('spatial_domain_markers.csv')

except ImportError:
    print("SpaGCN not installed. Install with: pip install SpaGCN")
except Exception as e:
    print(f"Error in SpaGCN: {e}")
    import traceback
    traceback.print_exc()

# ============ STEP 18: SPATIALLY VARIABLE GENE DETECTION ============
print("\n" + "=" * 60)
print("STEP 18: SPATIALLY VARIABLE GENE DETECTION")
print("=" * 60)

try:
    import squidpy as sq

    print("\nCalculating spatial autocorrelation (Moran's I)...")
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)
    sq.gr.spatial_autocorr(adata, mode='moran', genes=adata.var_names[:1000], n_perms=100)

    moran_results = adata.uns['moranI']
    print(f"\nAvailable columns: {moran_results.columns.tolist()}")

    # FIX: always assign gene_col in every branch
    if 'gene' in moran_results.columns:
        gene_col = 'gene'
    elif 'genes' in moran_results.columns:
        gene_col = 'genes'
    else:
        print("Using index for gene names")
        moran_results = moran_results.copy()
        moran_results['gene'] = moran_results.index
        gene_col = 'gene'

    top_spatial_genes = moran_results.nsmallest(20, 'pval_norm')[gene_col].tolist()
    print("\nTop 20 spatially variable genes:")
    print(moran_results.nsmallest(20, 'pval_norm')[['I', 'pval_norm', gene_col]])
    moran_results.to_csv('spatially_variable_genes.csv')

    print("\nPlotting top spatially variable genes...")
    spatial_coords = adata.obsm['spatial']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, gene in enumerate(top_spatial_genes[:6]):
        if gene in adata.var_names:
            gene_idx = list(adata.var_names).index(gene)
            exp = adata.X[:, gene_idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            moran_val = moran_results[moran_results[gene_col] == gene]['I'].values[0]
            scatter = axes[i].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                                      c=exp, cmap='viridis', s=3, alpha=0.7)
            axes[i].set_title(f"{gene}\nMoran's I={moran_val:.3f}")
            axes[i].set_aspect('equal')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            plt.colorbar(scatter, ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, f'{gene}\nnot in dataset',
                         ha='center', va='center', transform=axes[i].transAxes)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('spatially_variable_genes.png', dpi=150, bbox_inches='tight')
    plt.show()

    top10 = moran_results.nsmallest(10, 'pval_norm')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top10)), top10['I'].values)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10[gene_col].values)
    ax.set_xlabel("Moran's I (spatial autocorrelation)")
    ax.set_title('Top 10 Spatially Variable Genes')
    for i, (bar, pval) in enumerate(zip(bars, top10['pval_norm'].values)):
        bar.set_color('darkred' if pval < 0.001 else 'red' if pval < 0.01 else 'salmon' if pval < 0.05 else 'gray')
    plt.tight_layout()
    plt.savefig('top_spatial_genes_barplot.png', dpi=150, bbox_inches='tight')
    plt.show()

except ImportError:
    print("Squidpy not installed. Install with: pip install squidpy")
except Exception as e:
    print(f"Error in spatial analysis: {e}")
    import traceback
    traceback.print_exc()

# ============ STEP 19: HOTSPOT GENE MODULES ============
print("\n" + "=" * 60)
print("STEP 19: HOTSPOT GENE MODULE DETECTION")
print("=" * 60)

try:
    import hotspot

    print("\nRunning Hotspot analysis...")
    spatial_coords = adata.obsm['spatial']
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    hs = hotspot.Hotspot(adata, layer='X', model='danb', latent_obsm='X_pca',
                         umi_counts=adata.obs['total_counts'])
    hs.create_knn_graph(latent_obsm='spatial', n_neighbors=30)
    hs.compute_autocorrelations(jobs=4)

    significant_genes = hs.results[hs.results.FDR < 0.05].index.tolist()
    print(f"\nFound {len(significant_genes)} spatially autocorrelated genes")

    hs.create_modules(min_gene_threshold=5, core_only=True, fdr_threshold=0.05)
    module_df = hs.modules
    print("\nGene modules detected:")
    print(module_df['Module'].value_counts())
    module_df.to_csv('hotspot_modules.csv')

    module_scores = hs.module_scores()
    n_modules = module_df['Module'].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(min(n_modules, 6)):
        module_name = f'module_{i}'
        if module_name in module_scores.columns:
            scatter = axes[i].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                                      c=module_scores[module_name], cmap='RdBu_r', s=3, alpha=0.7)
            axes[i].set_title(f'Spatial Module {i}')
            axes[i].set_aspect('equal')
            plt.colorbar(scatter, ax=axes[i])
    plt.tight_layout()
    plt.savefig('hotspot_modules.png', dpi=150, bbox_inches='tight')
    plt.show()

except ImportError:
    print("Hotspot not installed. Install with: pip install hotspot")

# ============ STEP 20: SPATIAL DOMAIN REPRODUCIBILITY ============
print("\n" + "=" * 60)
print("STEP 20: SPATIAL DOMAIN REPRODUCIBILITY")
print("=" * 60)

if 'spatial_domain' in adata.obs.columns and 'orig.ident' in adata.obs.columns:
    print("\nAnalyzing spatial domain reproducibility across organoids...")
    domain_by_organoid = pd.crosstab(adata.obs['spatial_domain'], adata.obs['orig.ident'])
    print(domain_by_organoid)
    domain_by_organoid.to_csv('spatial_domains_by_organoid.csv')

    domain_proportions = domain_by_organoid.div(domain_by_organoid.sum(axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    domain_by_organoid.T.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10')
    axes[0].set_title('Spatial Domain Composition (Absolute)')
    axes[0].set_xlabel('Organoid')
    axes[0].set_ylabel('Number of Spots')
    axes[0].legend(title='Domain')
    domain_proportions.T.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10')
    axes[1].set_title('Spatial Domain Composition (Proportions)')
    axes[1].set_xlabel('Organoid')
    axes[1].set_ylabel('Proportion')
    axes[1].legend(title='Domain')
    plt.tight_layout()
    plt.savefig('spatial_domain_reproducibility.png', dpi=150, bbox_inches='tight')
    plt.show()

    from scipy.stats import chi2_contingency
    chi2_val, p_val, dof, expected = chi2_contingency(domain_by_organoid)
    print(f"\nChi-square test: p-value = {p_val:.4f}")

# ============ STEP 21: SPATIAL DOMAIN MARKERS ============
print("\n" + "=" * 60)
print("STEP 21: SPATIAL DOMAIN MARKER GENES")
print("=" * 60)

if 'spatial_domain' in adata.obs.columns:
    sc.tl.rank_genes_groups(adata, 'spatial_domain', method='wilcoxon', use_raw=False)
    result_spatial = adata.uns['rank_genes_groups']
    spatial_groups = result_spatial['names'].dtype.names
    for group in spatial_groups:
        print(f"\nSpatial Domain {group}:")
        for gene, logfc in zip(result_spatial['names'][group][:10], result_spatial['logfoldchanges'][group][:10]):
            print(f"  {gene}: log2FC={logfc:.3f}")
    spatial_markers = pd.DataFrame({
        group + '_' + key: result_spatial[key][group]
        for group in spatial_groups
        for key in ['names', 'scores', 'pvals_adj', 'logfoldchanges']
    })
    spatial_markers.to_csv('spatial_domain_markers.csv')
    sc.pl.rank_genes_groups_dotplot(adata, groupby='spatial_domain',
                                    n_genes=5, title='Spatial Domain Markers')

# ============ STEP 22: NEURONAL GENE ANALYSIS ============
print("\n" + "=" * 60)
print("STEP 22: NEURONAL GENE ANALYSIS FOR CORTICOSPINAL MOTOR ORGANOIDS")
print("=" * 60)

print("\nChecking current gene names:")
print(f"First 10 gene names: {adata.var_names[:10].tolist()}")

corticospinal_markers = {
    'Cortical Layer Markers': {
        'Upper Layer (II-IV)': ['CUX1', 'CUX2', 'SATB2', 'BCL11B'],
        'Deep Layer (V-VI)': ['BCL11B', 'TBR1', 'SOX5', 'FEZF2', 'CTIP2'],
        'Layer V (Corticospinal)': ['BCL11B', 'FEZF2', 'CRYM', 'ETV1'],
        'Layer VI': ['TBR1', 'FOXP2', 'CTGF']
    },
    'Motor Neuron Markers': {
        'General Motor Neurons': ['ISL1', 'MNX1', 'CHAT', 'SLC5A7', 'SLC18A3'],
        'Spinal Motor Neurons': ['HOXC4', 'HOXC5', 'HOXA5', 'HOXB5'],
        'Motor Progenitors': ['OLIG2', 'NKX6-1', 'PAX6', 'SOX2']
    },
    'Neuronal Subtype Markers': {
        'Glutamatergic': ['SLC17A7', 'SLC17A6', 'GRIN1', 'GRIA2'],
        'GABAergic': ['GAD1', 'GAD2', 'SLC32A1', 'DLX1', 'DLX2'],
        'Cholinergic': ['CHAT', 'ACHE', 'SLC5A7']
    },
    'Synaptic Markers': {
        'Presynaptic': ['SYP', 'SNAP25', 'STX1A', 'VAMP2'],
        'Postsynaptic': ['DLG4', 'GRIN1', 'GRIA2', 'HOMER1']
    }
}

print("\n1. CHECKING NEURONAL MARKER AVAILABILITY:")
print("-" * 40)

all_found_markers = []
for category, subcategories in corticospinal_markers.items():
    print(f"\n{category}:")
    for subcat, markers in subcategories.items():
        found = [m for m in markers if m in adata.var_names]
        for m in found:
            if m not in all_found_markers:
                all_found_markers.append(m)
        print(f"  {subcat}: {len(found)}/{len(markers)} found - {found if found else 'None'}")

print(f"\nTotal unique neuronal markers found: {len(all_found_markers)}")
print(f"Markers: {all_found_markers}")

print("\n2. TESTING HIGHER RESOLUTION CLUSTERING:")
print("-" * 40)

resolutions = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res{res}')
    n_clusters = len(adata.obs[f'leiden_res{res}'].unique())
    print(f"Resolution {res}: {n_clusters} clusters")

optimal_res = 1.2
print(f"\nUsing resolution {optimal_res} for detailed neuronal analysis")

sc.tl.leiden(adata, resolution=optimal_res, key_added='leiden_neuronal')
sc.tl.rank_genes_groups(adata, 'leiden_neuronal', method='wilcoxon', use_raw=False)

result_neuronal = adata.uns['rank_genes_groups']
neuronal_groups = result_neuronal['names'].dtype.names
print(f"\nIdentified {len(neuronal_groups)} clusters at resolution {optimal_res}")

# ============ DYNAMIC CLUSTER IDENTIFICATION ============
print("\n" + "=" * 60)
print("DYNAMIC CLUSTER IDENTIFICATION")
print("=" * 60)

motor_markers_list = ['SLC5A7', 'CHAT', 'ISL1', 'MNX1']
available_motor = [m for m in motor_markers_list if m in adata.var_names]

motor_scores = {}
if available_motor:
    for marker in available_motor:
        marker_idx = list(adata.var_names).index(marker)
        exp = adata.X[:, marker_idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        for group in neuronal_groups:
            mask = adata.obs['leiden_neuronal'] == group
            if mask.sum() > 0:
                if group not in motor_scores:
                    motor_scores[group] = []
                motor_scores[group].append(exp[mask].mean())

    motor_df = pd.DataFrame([{'cluster': k, 'motor_score': np.mean(v)}
                              for k, v in motor_scores.items() if v])
    motor_df = motor_df.sort_values('motor_score', ascending=False)
    top_motor_cluster = str(motor_df.iloc[0]['cluster'])
    print(f"\n✓ Motor neuron cluster identified: {top_motor_cluster}")
    print(f"  Motor score: {motor_df.iloc[0]['motor_score']:.3f}")
    print("\nTop 10 motor neuron clusters:")
    print(motor_df.head(10))
    motor_df.to_csv('motor_neuron_clusters.csv', index=False)
else:
    top_motor_cluster = '4'
    print(f"No motor markers found, defaulting to cluster {top_motor_cluster}")

fibroblast_markers_list = ['COL1A1', 'COL3A1', 'DCN', 'COL1A2', 'FN1', 'MGP']
available_fibro = [m for m in fibroblast_markers_list if m in adata.var_names]

fibro_scores = {}
if available_fibro:
    for marker in available_fibro:
        marker_idx = list(adata.var_names).index(marker)
        exp = adata.X[:, marker_idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        for group in neuronal_groups:
            mask = adata.obs['leiden_neuronal'] == group
            if mask.sum() > 0:
                if group not in fibro_scores:
                    fibro_scores[group] = []
                fibro_scores[group].append(exp[mask].mean())

    fibro_df = pd.DataFrame([{'cluster': k, 'fibroblast_score': np.mean(v)}
                              for k, v in fibro_scores.items() if v])
    fibro_df = fibro_df.sort_values('fibroblast_score', ascending=False)
    top_fibro_cluster = str(fibro_df.iloc[0]['cluster'])
    print(f"\n✓ Fibroblast cluster identified: {top_fibro_cluster}")
    print(f"  Fibroblast score: {fibro_df.iloc[0]['fibroblast_score']:.3f}")
    print("\nTop 10 fibroblast clusters:")
    print(fibro_df.head(10))
    fibro_df.to_csv('fibroblast_clusters.csv', index=False)
else:
    top_fibro_cluster = '8'
    print(f"No fibroblast markers found, defaulting to cluster {top_fibro_cluster}")

print("\n3. TOP MARKERS FOR EACH CLUSTER:")
print("-" * 40)
for group in neuronal_groups[:10]:
    genes = result_neuronal['names'][group][:10]
    logfcs = result_neuronal['logfoldchanges'][group][:10]
    print(f"\nCluster {group} top markers:")
    if group == top_motor_cluster:
        print("  ★ MOTOR NEURON CLUSTER ★")
    elif group == top_fibro_cluster:
        print("  ◆ FIBROBLAST CLUSTER ◆")
    for i, gene in enumerate(genes[:5]):
        logfc = logfcs[i] if i < len(logfcs) else np.nan
        tag = " (FIBROBLAST MARKER)" if gene in fibroblast_markers_list else ""
        print(f"  {gene}: log2FC={logfc:.3f}{tag}")

with open('cluster_identities.txt', 'w') as f:
    f.write(f"Motor neuron cluster: {top_motor_cluster}\n")
    f.write(f"Fibroblast cluster: {top_fibro_cluster}\n")
    f.write(f"Date: {pd.Timestamp.now()}\n")

print(f"\n✅ Dynamic cluster identification complete!")
print(f"  Motor cluster: {top_motor_cluster}")
print(f"  Fibroblast cluster: {top_fibro_cluster}")

# ============ QUANTITATIVE SPATIAL ANALYSIS ============
print("\n" + "=" * 60)
print("QUANTITATIVE SPATIAL ANALYSIS OF MOTOR NEURONS")
print("=" * 60)

spatial_coords = adata.obsm['spatial']
motor_mask = adata.obs['leiden_neuronal'] == top_motor_cluster
fibro_mask = adata.obs['leiden_neuronal'] == top_fibro_cluster
motor_coords = spatial_coords[motor_mask]
fibro_coords = spatial_coords[fibro_mask]

print(f"\nMotor neuron cluster {top_motor_cluster} statistics:")
print(f"  Number of spots: {motor_mask.sum()}")
print(f"  Percentage of total: {motor_mask.sum() / len(motor_mask) * 100:.1f}%")

if len(motor_coords) > 0:
    center = motor_coords.mean(axis=0)
    print(f"  Center position: ({center[0]:.0f}, {center[1]:.0f})")
    if len(motor_coords) > 3:
        hull = ConvexHull(motor_coords)
        print(f"  Spatial area: {hull.volume:.0f} pixels²")
    if len(motor_coords) > 1:
        from scipy.spatial.distance import pdist
        pairwise_distances = pdist(motor_coords)
        print(f"  Mean distance between spots: {np.mean(pairwise_distances):.0f} pixels")

print(f"\nFibroblast cluster {top_fibro_cluster} statistics:")
print(f"  Number of spots: {fibro_mask.sum()}")

if len(motor_coords) > 0 and len(fibro_coords) > 0:
    motor_center = motor_coords.mean(axis=0)
    fibro_center = fibro_coords.mean(axis=0)
    distance = np.linalg.norm(motor_center - fibro_center)
    print(f"\nDistance between cluster centers: {distance:.0f} pixels")
    motor_min, motor_max = motor_coords.min(axis=0), motor_coords.max(axis=0)
    fibro_min, fibro_max = fibro_coords.min(axis=0), fibro_coords.max(axis=0)
    overlap_x = not (motor_max[0] < fibro_min[0] or motor_min[0] > fibro_max[0])
    overlap_y = not (motor_max[1] < fibro_min[1] or motor_min[1] > fibro_max[1])
    if overlap_x and overlap_y:
        print("  ✓ Motor neurons and fibroblasts OVERLAP spatially")
    else:
        print("  ✗ Motor neurons and fibroblasts are SEPARATE")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c='lightgray', s=3, alpha=0.3)
    axes[0].scatter(motor_coords[:, 0], motor_coords[:, 1], c='red', s=10, alpha=0.8)
    axes[0].set_title(f'Motor Neuron Cluster {top_motor_cluster}\n({motor_mask.sum()} spots)')
    axes[0].set_aspect('equal')
    axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c='lightgray', s=3, alpha=0.3)
    axes[1].scatter(fibro_coords[:, 0], fibro_coords[:, 1], c='blue', s=10, alpha=0.8)
    axes[1].set_title(f'Fibroblast Cluster {top_fibro_cluster}\n({fibro_mask.sum()} spots)')
    axes[1].set_aspect('equal')
    axes[2].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c='lightgray', s=3, alpha=0.3)
    axes[2].scatter(motor_coords[:, 0], motor_coords[:, 1], c='red', s=10, alpha=0.8, label='Motor neurons')
    axes[2].scatter(fibro_coords[:, 0], fibro_coords[:, 1], c='blue', s=10, alpha=0.8, label='Fibroblasts')
    axes[2].set_title('Motor Neurons vs Fibroblasts')
    axes[2].set_aspect('equal')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('motor_vs_fibroblast_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

if 'SLC5A7' in adata.var_names:
    idx = list(adata.var_names).index('SLC5A7')
    exp = adata.X[:, idx]
    if hasattr(exp, 'toarray'):
        exp = exp.toarray().flatten()
    motor_exp = exp[motor_mask]
    print(f"\nSLC5A7 expression in motor cluster {top_motor_cluster}:")
    print(f"  Mean: {motor_exp.mean():.4f}")
    print(f"  % expressing: {(motor_exp > 0).mean() * 100:.1f}%")

print(f"\nTop marker genes for motor neuron cluster {top_motor_cluster}:")
motor_genes = []
for i, gene in enumerate(result_neuronal['names'][top_motor_cluster][:20]):
    score = result_neuronal['scores'][top_motor_cluster][i]
    logfc = result_neuronal['logfoldchanges'][top_motor_cluster][i]
    pval = result_neuronal['pvals'][top_motor_cluster][i]
    motor_genes.append({'gene': gene, 'score': score, 'log2FC': logfc, 'p_value': pval})
    print(f"  {gene}: score={score:.3f}")

motor_df_cluster = pd.DataFrame(motor_genes)
motor_df_cluster.to_csv(f'motor_neuron_cluster_{top_motor_cluster}_markers.csv', index=False)

print("\n✅ Analysis complete!")
print("=" * 60)

print("\nLayer marker expression in motor neuron cluster:")
layer_markers = {
    'Upper Layer (CUX1, CUX2)': ['CUX1', 'CUX2'],
    'Layer V/VI (BCL11B, TBR1)': ['BCL11B', 'TBR1', 'FEZF2'],
    'Motor Neuron (SLC5A7, MNX1)': ['SLC5A7', 'MNX1', 'CHAT']
}

motor_data = adata[adata.obs['leiden_neuronal'] == top_motor_cluster, :]
for layer, markers in layer_markers.items():
    print(f"\n{layer}:")
    for marker in markers:
        if marker in adata.var_names:
            idx = list(adata.var_names).index(marker)
            exp = motor_data.X[:, idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            print(f"  {marker}: {(exp > 0).mean() * 100:.1f}% expressing, mean={exp.mean():.3f}")
        else:
            print(f"  {marker}: not detected")

print(f"\nTop marker genes for motor neuron cluster {top_motor_cluster}:")
motor_genes_full = []
for i, gene in enumerate(result_neuronal['names'][top_motor_cluster][:50]):
    score = result_neuronal['scores'][top_motor_cluster][i]
    logfc = result_neuronal['logfoldchanges'][top_motor_cluster][i]
    pval = result_neuronal['pvals'][top_motor_cluster][i]
    motor_genes_full.append({'gene': gene, 'score': score, 'log2FC': logfc, 'p_value': pval})

motor_df_full = pd.DataFrame(motor_genes_full)
motor_df_full.to_csv(f'motor_neuron_cluster_{top_motor_cluster}_markers.csv', index=False)
print(f"\nSaved {len(motor_df_full)} marker genes")
print(motor_df_full.head(10))

upper_mn = ['BCL11B', 'FEZF2', 'CRYM', 'ETV1', 'SOX5']
lower_mn = ['ISL1', 'MNX1', 'HOXC4', 'HOXC5', 'HOXA5']

print("\nMotor neuron subtype analysis:")
print("Upper motor neuron markers:")
for marker in upper_mn:
    if marker in adata.var_names:
        idx = list(adata.var_names).index(marker)
        exp = adata[adata.obs['leiden_neuronal'] == top_motor_cluster, idx].X
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        print(f"  {marker}: {exp.mean():.3f}")

print("\nLower motor neuron markers:")
for marker in lower_mn:
    if marker in adata.var_names:
        idx = list(adata.var_names).index(marker)
        exp = adata[adata.obs['leiden_neuronal'] == top_motor_cluster, idx].X
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        print(f"  {marker}: {exp.mean():.3f}")

# Publication figure
fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 3, 1)
mask = adata.obs['leiden_neuronal'] == top_motor_cluster
ax1.scatter(adata.obsm['spatial'][~mask, 0], adata.obsm['spatial'][~mask, 1],
            c='lightgray', s=3, alpha=0.3, label='Other cells')
ax1.scatter(adata.obsm['spatial'][mask, 0], adata.obsm['spatial'][mask, 1],
            c='red', s=10, alpha=0.8, label='Motor neurons')
ax1.set_title('A: Spatial Distribution of Motor Neurons', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')
ax1.legend()

ax2 = plt.subplot(2, 3, 2)
if 'SLC5A7' in adata.var_names:
    idx = list(adata.var_names).index('SLC5A7')
    exp = adata.X[:, idx]
    if hasattr(exp, 'toarray'):
        exp = exp.toarray().flatten()
    scatter = ax2.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                          c=exp, cmap='Reds', s=5, alpha=0.7)
    ax2.set_title('B: SLC5A7 Expression (Motor Neuron Marker)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2)

ax3 = plt.subplot(2, 3, 3)
top10_markers = motor_df_full.head(10)
bars = ax3.barh(range(10), top10_markers['score'].values)
ax3.set_yticks(range(10))
ax3.set_yticklabels(top10_markers['gene'].values)
ax3.set_xlabel('Wilcoxon Score')
ax3.set_title('C: Top Motor Neuron Markers', fontsize=14, fontweight='bold')
for i, (bar, p) in enumerate(zip(bars, top10_markers['p_value'].values)):
    bar.set_color('darkred' if p < 0.001 else 'red' if p < 0.01 else 'salmon' if p < 0.05 else 'gray')

ax4 = plt.subplot(2, 3, 4)
motor_mask = adata.obs['leiden_neuronal'] == top_motor_cluster
fib_mask = adata.obs['leiden_neuronal'] == top_fibro_cluster
ax4.scatter(adata.obsm['spatial'][motor_mask, 0], adata.obsm['spatial'][motor_mask, 1],
            c='red', s=10, alpha=0.8, label='Motor neurons')
ax4.scatter(adata.obsm['spatial'][fib_mask, 0], adata.obsm['spatial'][fib_mask, 1],
            c='blue', s=10, alpha=0.8, label='Fibroblasts')
ax4.set_title('D: Motor Neuron-Fibroblast Interaction', fontsize=14, fontweight='bold')
ax4.set_aspect('equal')
ax4.legend()

ax5 = plt.subplot(2, 3, 5)
if 'GRIA2' in adata.var_names:
    idx = list(adata.var_names).index('GRIA2')
    exp = adata.X[:, idx]
    if hasattr(exp, 'toarray'):
        exp = exp.toarray().flatten()
    scatter = ax5.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                          c=exp, cmap='Blues', s=5, alpha=0.7)
    ax5.set_title('E: GRIA2 Expression (Glutamatergic)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax5)

ax6 = plt.subplot(2, 3, 6)
cluster_sizes = adata.obs['leiden_neuronal'].value_counts()
motor_pct = (cluster_sizes[top_motor_cluster] / len(adata)) * 100
fib_pct = (cluster_sizes[top_fibro_cluster] / len(adata)) * 100
other_pct = 100 - motor_pct - fib_pct
ax6.pie([motor_pct, fib_pct, other_pct], labels=['Motor Neurons', 'Fibroblasts', 'Other'],
        colors=['red', 'blue', 'lightgray'], autopct='%1.1f%%')
ax6.set_title('F: Tissue Composition', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('motor_neuron_publication_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Publication figure saved to motor_neuron_publication_figure.png")

# ============ ADDITIONAL NEURONAL SUBTYPE ANALYSIS ============
print("\n" + "=" * 60)
print("ADDITIONAL NEURONAL SUBTYPE ANALYSIS")
print("=" * 60)

subtype_markers = {
    'Sensory Spinal Neurons': {
        'General Sensory': ['NTRK1', 'NTRK2', 'NTRK3', 'RET', 'RUNX1', 'RUNX3'],
        'Mechanoreceptors': ['MECOM', 'MAFA', 'MAFB', 'NEFH', 'PVALB'],
        'Proprioceptors': ['ETV1', 'ETV4', 'RUNX3', 'PARV', 'NEFH'],
        'Nociceptors': ['TRPV1', 'TRPA1', 'SCN9A', 'SCN10A', 'TAC1', 'CALCA'],
        'Pruriceptors': ['SST', 'NPPB', 'NPY2R', 'MRGPRD']
    },
    'Cortical Neurons': {
        'General Cortical': ['SATB2', 'BCL11B', 'TBR1', 'CUX1', 'CUX2', 'RORB'],
        'Upper Layer (II-IV)': ['CUX1', 'CUX2', 'SATB2', 'RORB'],
        'Deep Layer (V-VI)': ['BCL11B', 'TBR1', 'SOX5', 'FEZF2', 'CTIP2'],
        'Layer V Corticospinal': ['BCL11B', 'FEZF2', 'CRYM', 'ETV1'],
        'Layer VI': ['TBR1', 'FOXP2', 'CTGF']
    },
    'GABAergic Neurons': {
        'General GABAergic': ['GAD1', 'GAD2', 'SLC32A1', 'GABRA1', 'GABRG2'],
        'Parvalbumin+ (PV)': ['PVALB', 'SST', 'VIP', 'CCK', 'NPY'],
        'Somatostatin+ (SST)': ['SST', 'NPY', 'CALB2'],
        'VIP+': ['VIP', 'CCK', 'CALB2'],
        'Calretinin+': ['CALB2', 'VIP'],
        'Calbindin+': ['CALB1']
    },
    'Glutamatergic Neurons': {
        'General Glutamatergic': ['SLC17A7', 'SLC17A6', 'GRIN1', 'GRIA2', 'GRIK1'],
        'AMPA Receptors': ['GRIA1', 'GRIA2', 'GRIA3', 'GRIA4'],
        'NMDA Receptors': ['GRIN1', 'GRIN2A', 'GRIN2B', 'GRIN2D']
    }
}

print("\n1. CHECKING NEURONAL SUBTYPE MARKER AVAILABILITY:")
print("-" * 60)

all_subtype_markers = {}
for category, subcategories in subtype_markers.items():
    print(f"\n{category}:")
    category_markers = []
    for subcat, markers in subcategories.items():
        found = [m for m in markers if m in adata.var_names]
        category_markers.extend(found)
        print(f"  {subcat}: {len(found)}/{len(markers)} found - {found if found else 'None'}")
    all_subtype_markers[category] = list(set(category_markers))

print("\n2. CALCULATING SUBTYPE SCORES PER CLUSTER:")
print("-" * 60)

subtype_scores = {}
for category, markers in all_subtype_markers.items():
    if len(markers) >= 2:
        print(f"\n{category} ({len(markers)} markers):")
        category_scores = []
        for group in neuronal_groups[:15]:
            mask = adata.obs['leiden_neuronal'] == group
            if mask.sum() > 0:
                cluster_exp = []
                for marker in markers:
                    if marker in adata.var_names:
                        marker_idx = list(adata.var_names).index(marker)
                        exp = adata[mask, marker_idx].X
                        if hasattr(exp, 'toarray'):
                            exp = exp.toarray().flatten()
                        cluster_exp.append(exp.mean())
                if cluster_exp:
                    category_scores.append({'cluster': group, 'score': np.mean(cluster_exp)})
        if category_scores:
            score_df = pd.DataFrame(category_scores).sort_values('score', ascending=False)
            subtype_scores[category] = score_df
            print(f"  Top 3 clusters:")
            for _, row in score_df.head(3).iterrows():
                star = "★" if row['cluster'] == top_motor_cluster else ""
                print(f"    Cluster {row['cluster']}{star}: {row['score']:.4f}")

print("\n3. CREATING SUBTYPE SCORE HEATMAP...")
heatmap_data, heatmap_rows = [], []
heatmap_cols = list(neuronal_groups[:15])

for category, score_df in subtype_scores.items():
    if not score_df.empty:
        heatmap_rows.append(category)
        row_data = []
        for cluster in heatmap_cols:
            score_val = score_df[score_df['cluster'] == cluster]['score'].values
            row_data.append(score_val[0] if len(score_val) > 0 else 0)
        heatmap_data.append(row_data)

if heatmap_data:
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_xticklabels(heatmap_cols, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_rows)))
    ax.set_yticklabels(heatmap_rows)
    if top_motor_cluster in heatmap_cols:
        motor_idx = heatmap_cols.index(top_motor_cluster)
        ax.axvline(x=motor_idx, color='red', linewidth=2, linestyle='--', alpha=0.5)
        ax.text(motor_idx, -0.5, 'Motor\nNeuron', ha='center', va='top', color='red', fontsize=9)
    plt.colorbar(im, ax=ax, label='Mean Expression')
    ax.set_title('Neuronal Subtype Scores by Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('neuronal_subtype_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Heatmap saved to neuronal_subtype_heatmap.png")

print("\n4. MOTOR NEURON CLUSTER SUBTYPE ANALYSIS:")
print("-" * 60)
print(f"Analyzing motor neuron cluster {top_motor_cluster}")

motor_data = adata[adata.obs['leiden_neuronal'] == top_motor_cluster, :]
for category, markers in all_subtype_markers.items():
    if markers:
        print(f"\n{category}:")
        for marker in markers[:10]:
            if marker in adata.var_names:
                idx = list(adata.var_names).index(marker)
                exp = motor_data.X[:, idx]
                if hasattr(exp, 'toarray'):
                    exp = exp.toarray().flatten()
                if exp.mean() > 0.1:
                    print(f"  {marker}: {(exp > 0).mean() * 100:.1f}% expressing, mean={exp.mean():.3f}")

print("\n5. CREATING SUBTYPE SCORE SUMMARY...")
all_scores_list = []
for category, score_df in subtype_scores.items():
    for _, row in score_df.iterrows():
        all_scores_list.append({'cluster': row['cluster'], 'subtype': category, 'score': row['score']})

all_scores_df = pd.DataFrame(all_scores_list)
pivot_scores = all_scores_df.pivot(index='cluster', columns='subtype', values='score').fillna(0)
pivot_scores.to_csv('neuronal_subtype_scores_all.csv')
print("Subtype scores saved to neuronal_subtype_scores_all.csv")
print("\nTop clusters for each subtype:")
print(pivot_scores.head(10))

print("\n6. DOMINANT SUBTYPE IN MOTOR NEURON CLUSTER:")
print("-" * 60)
motor_subtypes = pivot_scores.loc[top_motor_cluster] if top_motor_cluster in pivot_scores.index else None
if motor_subtypes is not None:
    motor_subtypes = motor_subtypes.sort_values(ascending=False)
    print(f"\nMotor neuron cluster {top_motor_cluster} subtype scores:")
    for subtype, score in motor_subtypes.head(5).items():
        print(f"  {subtype}: {score:.4f}")
    dominant = motor_subtypes.index[0]
    print(f"\n→ Dominant subtype: {dominant}")

print("\n7. VISUALIZING TOP SUBTYPES...")
top_subtypes = all_scores_df.groupby('subtype')['score'].mean().nlargest(5).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, subtype in enumerate(top_subtypes[:5]):
    subtype_data = all_scores_df[all_scores_df['subtype'] == subtype]
    axes[i].bar(range(len(subtype_data)), subtype_data['score'].values)
    axes[i].set_xticks(range(len(subtype_data)))
    axes[i].set_xticklabels(subtype_data['cluster'].values, rotation=45, ha='right')
    axes[i].set_title(f'{subtype} Scores')
    axes[i].set_xlabel('Cluster')
    axes[i].set_ylabel('Score')
    if top_motor_cluster in subtype_data['cluster'].values:
        motor_idx = list(subtype_data['cluster']).index(top_motor_cluster)
        axes[i].patches[motor_idx].set_color('red')
if len(top_subtypes) < 6:
    axes[5].set_visible(False)
plt.tight_layout()
plt.savefig('subtype_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Additional neuronal subtype analysis complete!")

# ============ SKELETAL MUSCLE ANALYSIS ============
print("\n" + "=" * 60)
print("SKELETAL MUSCLE ANALYSIS")
print("=" * 60)

skeletal_muscle_markers = {
    'Myogenic regulators': ['MYOD1', 'MYF5', 'MYOG', 'PAX3', 'PAX7'],
    'Structural proteins': ['MYH1', 'MYH2', 'MYH3', 'MYH7', 'MYH8', 'ACTA1', 'DES', 'TNNT1', 'TNNT3', 'TNNC1', 'TNNC2'],
    'Sarcomeric': ['TTN', 'NEB', 'ACTN2', 'MYBPC1', 'MYBPC2', 'MYL1', 'MYL2', 'MYL3'],
    'Muscle-specific': ['CKMT2', 'CKM', 'CHRNG', 'CHRND', 'CHRNE']
}

print("\n1. CHECKING SKELETAL MUSCLE MARKER AVAILABILITY:")
print("-" * 40)
all_muscle_markers = []
for category, markers in skeletal_muscle_markers.items():
    found = [m for m in markers if m in adata.var_names]
    all_muscle_markers.extend(found)
    print(f"{category}: {len(found)}/{len(markers)} found - {found if found else 'None'}")

print(f"\nTotal skeletal muscle markers found: {len(all_muscle_markers)}")
print(f"Markers: {all_muscle_markers}")

if len(all_muscle_markers) >= 2:
    print("\n2. CALCULATING SKELETAL MUSCLE SCORES PER CLUSTER:")
    print("-" * 40)

    muscle_scores_dict = {}
    for group in neuronal_groups:
        mask = adata.obs['leiden_neuronal'] == group
        if mask.sum() > 0:
            cluster_exp = []
            for marker in all_muscle_markers:
                if marker in adata.var_names:
                    marker_idx = list(adata.var_names).index(marker)
                    exp = adata[mask, marker_idx].X
                    if hasattr(exp, 'toarray'):
                        exp = exp.toarray().flatten()
                    cluster_exp.append(exp.mean())
            if cluster_exp:
                muscle_scores_dict[group] = np.mean(cluster_exp)

    if muscle_scores_dict:
        muscle_score_df = pd.DataFrame([{'cluster': k, 'muscle_score': v}
                                        for k, v in muscle_scores_dict.items()]).sort_values('muscle_score', ascending=False)
        print("\nTop clusters with skeletal muscle signature:")
        print(muscle_score_df.head(10))
        muscle_score_df.to_csv('skeletal_muscle_clusters.csv', index=False)

        top_muscle_cluster = muscle_score_df.iloc[0]['cluster']
        print(f"\n✓ Top skeletal muscle cluster: {top_muscle_cluster}")
        print(f"  Score: {muscle_score_df.iloc[0]['muscle_score']:.4f}")
        if top_muscle_cluster == top_motor_cluster:
            print("  → This is the SAME as your motor neuron cluster!")
        else:
            print(f"  → Different from motor neuron cluster {top_motor_cluster}")

        fig, ax = plt.subplots(figsize=(8, 8))
        mask = adata.obs['leiden_neuronal'] == top_muscle_cluster
        ax.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], c='lightgray', s=3, alpha=0.3)
        ax.scatter(adata.obsm['spatial'][mask, 0], adata.obsm['spatial'][mask, 1],
                   c='green', s=10, alpha=0.8, label=f'Skeletal Muscle Cluster {top_muscle_cluster}')
        ax.set_title(f'Skeletal Muscle Cluster {top_muscle_cluster}')
        ax.set_aspect('equal')
        ax.legend()
        plt.savefig('skeletal_muscle_spatial.png', dpi=150, bbox_inches='tight')
        plt.show()

# ============ TARDBP ENRICHMENT ANALYSIS ============
print("\n" + "=" * 60)
print("TARDBP ENRICHMENT ANALYSIS")
print("=" * 60)

c9_gene = 'TARDBP'
if c9_gene in adata.var_names:
    print(f"\n✓ {c9_gene} found in dataset")
    c9_idx = list(adata.var_names).index(c9_gene)
    c9_exp = adata.X[:, c9_idx]
    if hasattr(c9_exp, 'toarray'):
        c9_exp = c9_exp.toarray().flatten()
    adata.obs['TARDBP_expression'] = c9_exp

    if 'orig.ident' in adata.obs.columns:
        organoids_tardbp = adata.obs['orig.ident'].unique()
        c9_by_organoid = []
        for organoid in organoids_tardbp:
            mask = adata.obs['orig.ident'] == organoid
            organoid_exp = c9_exp[mask]
            stats = {'organoid': organoid, 'mean_expression': organoid_exp.mean(),
                     'median_expression': np.median(organoid_exp), 'max_expression': organoid_exp.max(),
                     'percent_expressing': (organoid_exp > 0).mean() * 100, 'n_cells': mask.sum()}
            c9_by_organoid.append(stats)
            print(f"\n{organoid}: mean={stats['mean_expression']:.4f}, "
                  f"% expressing={stats['percent_expressing']:.1f}%")
        pd.DataFrame(c9_by_organoid).to_csv('TARDBP_by_organoid.csv', index=False)
else:
    print(f"\n✗ {c9_gene} NOT found in dataset")
    similar = [g for g in adata.var_names if 'C9orf' in g or 'C9' in g]
    print(f"Available genes similar to TARDBP:")
    print(similar[:10] if similar else "None found")

# ============ ALS/NEURODEGENERATION GENE ANALYSIS ============
print("\n" + "=" * 60)
print("ALS/NEURODEGENERATION GENE ANALYSIS")
print("=" * 60)

als_gene_groups = {
    'ALS_Core': ['TARDBP', 'FUS', 'OPTN', 'SOD1', 'NEK1', 'TBK1', 'CHMP2B', 'UNC13A'],
    'FTD_Core': ['MAPT', 'GRN', 'TMEM106B'],
    'HOX_Spinal': ['HOXA7', 'HOXA10', 'HOXA11'],
    'Signaling': ['STK10', 'MAP4K3', 'EFR3A', 'EPHA4'],
    'Other': ['SHOX2', 'CAMTA1', 'CDH22', 'MTX2']
}

print("\n1. CHECKING ALS GENE AVAILABILITY:")
print("-" * 60)

all_als_markers = []
for group_name, genes in als_gene_groups.items():
    available = [g for g in genes if g in adata.var_names]
    all_als_markers.extend(available)
    print(f"\n{group_name}: {len(available)}/{len(genes)} found")
    if available:
        print(f"  → {', '.join(available)}")
    else:
        print(f"  → None found")

print(f"\nTotal ALS-related genes found: {len(all_als_markers)}")
if all_als_markers:
    print(f"Genes: {', '.join(all_als_markers)}")

if all_als_markers:
    print(f"\n2. ANALYZING ALS GENE EXPRESSION:")
    print("-" * 60)

    als_stats = []
    for gene in all_als_markers:
        idx = list(adata.var_names).index(gene)
        exp = adata.X[:, idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        stats = {'gene': gene, 'mean': exp.mean(), 'median': np.median(exp),
                 'std': exp.std(), 'sem': exp.std() / np.sqrt(len(exp)),
                 'min': exp.min(), 'max': exp.max(),
                 'pct_expressing': (exp > 0).mean() * 100, 'n_cells': len(exp)}
        als_stats.append(stats)
        print(f"\n📊 {gene}:")
        print(f"    Mean ± SEM: {stats['mean']:.4f} ± {stats['sem']:.4f}")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    % expressing: {stats['pct_expressing']:.1f}%")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    als_stats_df = pd.DataFrame(als_stats)
    als_stats_df.to_csv('ALS_genes_statistics.csv', index=False)
    print("\n✅ ALS gene statistics saved to 'ALS_genes_statistics.csv'")

    print("\n3. CREATING VISUALIZATIONS:")
    print("-" * 60)

    genes_list = als_stats_df['gene'].values
    means = als_stats_df['mean'].values
    errors = als_stats_df['sem'].values
    x_pos = np.arange(len(genes_list))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genes_list, rotation=45, ha='right')
    ax.set_ylabel('Mean Expression ± SEM')
    ax.set_title('ALS Gene Expression Levels', fontweight='bold')
    plt.tight_layout()
    plt.savefig('ALS_genes_barplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Bar plot saved to 'ALS_genes_barplot.png'")

    fig, ax = plt.subplots(figsize=(12, 6))
    pct = als_stats_df['pct_expressing'].values
    bars = ax.bar(x_pos, pct, color='coral', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genes_list, rotation=45, ha='right')
    ax.set_ylabel('% Expressing Cells')
    ax.set_ylim(0, 100)
    ax.set_title('Percentage of Cells Expressing ALS Genes', fontweight='bold')
    plt.tight_layout()
    plt.savefig('ALS_genes_percentage.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Percentage plot saved to 'ALS_genes_percentage.png'")

    if len(all_als_markers) >= 2:
        als_exp_matrix = []
        for gene in all_als_markers:
            idx = list(adata.var_names).index(gene)
            exp = adata.X[:, idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            als_exp_matrix.append(exp)
        als_exp_matrix = np.array(als_exp_matrix).T
        corr_matrix = np.corrcoef(als_exp_matrix.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(all_als_markers)))
        ax.set_xticklabels(all_als_markers, rotation=45, ha='right')
        ax.set_yticks(range(len(all_als_markers)))
        ax.set_yticklabels(all_als_markers)
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('ALS Gene Expression Correlation Matrix', fontsize=14, fontweight='bold')
        for i in range(len(all_als_markers)):
            for j in range(len(all_als_markers)):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
        plt.tight_layout()
        plt.savefig('ALS_genes_correlation.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✅ Correlation matrix saved to 'ALS_genes_correlation.png'")

    print("\n4. CLUSTER-SPECIFIC ENRICHMENT:")
    print("-" * 60)
    for gene in all_als_markers:
        print(f"\n{gene} expression by cluster:")
        idx = list(adata.var_names).index(gene)
        exp = adata.X[:, idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        threshold = np.percentile(exp, 75)
        cluster_means = []
        for cluster in neuronal_groups[:10]:
            mask = adata.obs['leiden_neuronal'] == cluster
            if mask.sum() > 0:
                cluster_exp = exp[mask]
                mean_exp = cluster_exp.mean()
                pct_high = (cluster_exp > threshold).mean() * 100
                if mean_exp > 0.01:
                    cluster_means.append((cluster, mean_exp, pct_high))
        cluster_means.sort(key=lambda x: x[1], reverse=True)
        for cluster, mean_exp, pct_high in cluster_means[:5]:
            star = "★" if cluster == top_motor_cluster else ""
            print(f"  Cluster {cluster}{star}: mean={mean_exp:.4f}, {pct_high:.1f}% high expressors")

    print("\n5. SPATIAL VISUALIZATION OF ALS GENES:")
    print("-" * 60)
    spatial_coords_als = adata.obsm['spatial']
    n_genes = len(all_als_markers)
    n_cols = min(3, n_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_genes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, gene in enumerate(all_als_markers):
        idx = list(adata.var_names).index(gene)
        exp = adata.X[:, idx]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        vmax = np.percentile(exp[exp > 0], 95) if (exp > 0).any() else 1
        scatter = axes[i].scatter(spatial_coords_als[:, 0], spatial_coords_als[:, 1],
                                  c=exp, cmap='Reds', s=5, alpha=0.7, vmin=0, vmax=vmax)
        axes[i].set_title(f'{gene} Expression', fontweight='bold')
        axes[i].set_aspect('equal')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(scatter, ax=axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Spatial Expression of ALS-Related Genes', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ALS_genes_spatial.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Spatial plots saved to 'ALS_genes_spatial.png'")

    print("\n" + "=" * 60)
    print("ALS GENE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nTotal ALS-related genes analyzed: {len(all_als_markers)}")
    print(f"\nGenes detected: {', '.join(all_als_markers)}")
    print("\nExpression Summary:")
    print(als_stats_df[['gene', 'mean', 'pct_expressing']].round(4).to_string(index=False))

else:
    print("\n❌ No ALS-related genes found in dataset")

print("\n✅ ALS gene analysis complete!")

# ============ SEPARATING INDIVIDUAL ORGANOIDS ============
print("\n" + "=" * 60)
print("SEPARATING INDIVIDUAL ORGANOIDS")
print("=" * 60)

from sklearn.cluster import KMeans

spatial_coords = adata.obsm['spatial']
print("\n1. Using K-means to separate organoids...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
spatial_clusters = kmeans.fit_predict(spatial_coords)
adata.obs['organoid'] = [f'Organoid_{i + 1}' for i in spatial_clusters]

print("Organoid distribution:")
print(adata.obs['organoid'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                c=adata.obs['leiden'].astype('category').cat.codes, cmap='tab10', s=5, alpha=0.7)
axes[0].set_title('Original Clusters')
axes[0].set_aspect('equal')
scatter = axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                          c=spatial_clusters, cmap='Set1', s=5, alpha=0.7)
axes[1].set_title('K-means Separated Organoids')
axes[1].set_aspect('equal')
plt.colorbar(scatter, ax=axes[1])
plt.tight_layout()
plt.savefig('organoid_separation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ STATISTICAL COMPARISON BETWEEN ORGANOIDS ============
print("\n" + "=" * 60)
print("STATISTICAL COMPARISON BETWEEN ORGANOIDS")
print("=" * 60)

print(f"Using motor neuron cluster: {top_motor_cluster}")

organoids = adata.obs['organoid'].unique()
print(f"\nOrganoids identified: {organoids.tolist()}")

print("\n1. MOTOR NEURON ABUNDANCE BY ORGANOID:")
print("-" * 40)

motor_abundance = []
for organoid in organoids:
    mask_organoid = adata.obs['organoid'] == organoid
    mask_motor = adata.obs['leiden_neuronal'] == top_motor_cluster
    mask = mask_organoid & mask_motor
    total_cells = mask_organoid.sum()
    motor_cells = mask.sum()
    stats = {'organoid': organoid, 'total_cells': total_cells,
             'motor_neuron_cells': motor_cells,
             'percentage_motor': (motor_cells / total_cells * 100) if total_cells > 0 else 0}
    motor_abundance.append(stats)
    print(f"\n{organoid}:")
    print(f"  Total cells: {total_cells}")
    print(f"  Motor neurons: {motor_cells}")
    print(f"  % Motor neurons: {stats['percentage_motor']:.2f}%")

motor_df_organoid = pd.DataFrame(motor_abundance)
motor_df_organoid.to_csv('motor_neuron_abundance_by_organoid.csv', index=False)

from scipy.stats import chi2_contingency

print("\n2. STATISTICAL TEST FOR MOTOR NEURON ABUNDANCE:")
print("-" * 40)

contingency_data = []
for organoid in organoids:
    mask_organoid = adata.obs['organoid'] == organoid
    mask_motor = adata.obs['leiden_neuronal'] == top_motor_cluster
    motor_cells = (mask_organoid & mask_motor).sum()
    other_cells = (mask_organoid & ~mask_motor).sum()
    contingency_data.append([motor_cells, other_cells])

contingency = pd.DataFrame(contingency_data, index=organoids,
                           columns=['Motor Neurons', 'Other Cells'])
print("\nContingency table:")
print(contingency)

chi2_val, p_val, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test for independence:")
print(f"  χ² = {chi2_val:.3f}")
print(f"  p-value = {p_val:.6f}")
if p_val < 0.05:
    print("  → SIGNIFICANT difference in motor neuron abundance between organoids")
else:
    print("  → No significant difference")

print("\n3. MOTOR NEURON MARKER EXPRESSION BY ORGANOID:")
print("-" * 40)

if 'SLC5A7' in adata.var_names:
    marker_idx = list(adata.var_names).index('SLC5A7')
    slc5a7_exp = adata.X[:, marker_idx]
    if hasattr(slc5a7_exp, 'toarray'):
        slc5a7_exp = slc5a7_exp.toarray().flatten()

    marker_stats = []
    expression_groups = []

    for organoid in organoids:
        mask = adata.obs['organoid'] == organoid
        organoid_exp = slc5a7_exp[mask]
        expression_groups.append(organoid_exp)
        stats = {'organoid': organoid, 'mean_expression': organoid_exp.mean(),
                 'median_expression': np.median(organoid_exp),
                 'std_expression': organoid_exp.std(),
                 'sem_expression': organoid_exp.std() / np.sqrt(len(organoid_exp)),
                 'percent_expressing': (organoid_exp > 0).mean() * 100}
        marker_stats.append(stats)
        print(f"\n{organoid}:")
        print(f"  Mean SLC5A7: {stats['mean_expression']:.4f} ± {stats['sem_expression']:.4f}")
        print(f"  % expressing: {stats['percent_expressing']:.1f}%")

    marker_df = pd.DataFrame(marker_stats)
    marker_df.to_csv('SLC5A7_expression_by_organoid.csv', index=False)

    from scipy.stats import f_oneway
    f_stat, p_val_anova = f_oneway(*expression_groups)
    print(f"\nANOVA test for SLC5A7 expression:")
    print(f"  F-statistic = {f_stat:.3f}")
    print(f"  p-value = {p_val_anova:.6f}")
    if p_val_anova < 0.05:
        print("  → SIGNIFICANT difference in SLC5A7 expression between organoids")
    else:
        print("  → No significant difference")

    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    print("\nPairwise comparisons (t-test with Bonferroni correction):")
    p_values = []
    comparisons = []
    for i in range(len(organoids)):
        for j in range(i + 1, len(organoids)):
            t_stat, p = ttest_ind(expression_groups[i], expression_groups[j])
            p_values.append(p)
            comparisons.append(f"{organoids[i]} vs {organoids[j]}")

    rejected, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
    for comp, p_raw, p_corr in zip(comparisons, p_values, p_corrected):
        sig_star = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        print(f"  {comp}: p={p_raw:.4f} (adj={p_corr:.4f}) {sig_star}")

print("\n4. VISUALIZING MOTOR NEURONS BY ORGANOID:")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()
colors_org = ['red', 'blue', 'green', 'purple']

for i, organoid in enumerate(organoids):
    if i < 4:
        organoid_mask = adata.obs['organoid'] == organoid
        axes[i].scatter(spatial_coords[organoid_mask, 0], spatial_coords[organoid_mask, 1],
                        c='lightgray', s=3, alpha=0.3)
        motor_mask_org = organoid_mask & (adata.obs['leiden_neuronal'] == top_motor_cluster)
        axes[i].scatter(spatial_coords[motor_mask_org, 0], spatial_coords[motor_mask_org, 1],
                        c=colors_org[i % len(colors_org)], s=10, alpha=0.8, label='Motor Neurons')
        axes[i].set_title(f'{organoid}')
        axes[i].set_aspect('equal')
        axes[i].legend()

plt.tight_layout()
plt.savefig('motor_neurons_by_organoid.png', dpi=150, bbox_inches='tight')
plt.show()

if 'SLC5A7' in adata.var_names:
    print("\n5. CREATING BOXPLOT OF SLC5A7 EXPRESSION BY ORGANOID:")
    print("-" * 40)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = [slc5a7_exp[adata.obs['organoid'] == org] for org in organoids]
    bp = ax.boxplot(plot_data, patch_artist=True, labels=organoids)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors_org[i % len(colors_org)])
    ax.set_ylabel('SLC5A7 Expression')
    ax.set_title('Motor Neuron Marker Expression by Organoid')
    if 'p_val_anova' in locals() and p_val_anova < 0.05:
        ax.text(0.5, 0.95, f'ANOVA: p={p_val_anova:.4f}', transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig('slc5a7_by_organoid_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "=" * 60)
print("SUMMARY: ORGANOID COMPARISON")
print("=" * 60)

summary_df = motor_df_organoid
if 'marker_df' in locals():
    summary_df = summary_df.merge(marker_df, on='organoid', how='outer')

print("\nOrganoid Comparison Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv('organoid_comparison_summary.csv', index=False)

print("\n✅ Organoid separation and comparison complete!")
print("Files saved:")
print("  - organoid_separation.png")
print("  - motor_neuron_abundance_by_organoid.csv")
if 'marker_df' in locals():
    print("  - SLC5A7_expression_by_organoid.csv")
    print("  - slc5a7_by_organoid_boxplot.png")
print("  - motor_neurons_by_organoid.png")
print("  - organoid_comparison_summary.csv")

# ============ UMAPS FOR EACH ORGANOID ============
# FIX: all sc.pl.umap calls with gene color now include use_raw=False
print("\n" + "=" * 60)
print("CREATING INDIVIDUAL UMAPS FOR EACH ORGANOID")
print("=" * 60)

organoids = adata.obs['organoid'].unique()
print(f"Organoids: {organoids.tolist()}")

key_genes = ['SLC5A7', 'BCL11B', 'SNAP25', 'GRIA2', 'SLC32A1',
             'COL1A1', 'COL3A1', 'DCN', 'RBFOX3', 'CSF1R', 'TREM2']

available_genes = [g for g in key_genes if g in adata.var_names]
print(f"\nGenes to visualize: {available_genes}")

print("\n1. CREATING INDIVIDUAL UMAPS WITH GENE OVERLAYS...")

for organoid in organoids:
    organoid_mask = adata.obs['organoid'] == organoid
    adata_organoid = adata[organoid_mask, :].copy()

    if adata_organoid.shape[0] < 10:
        print(f"  Skipping {organoid} - only {adata_organoid.shape[0]} cells")
        continue

    print(f"\n  Processing {organoid}: {adata_organoid.shape[0]} cells")

    # Check which genes are expressed in this subset
    genes_in_organoid = []
    for gene in available_genes:
        if gene in adata_organoid.var_names:
            gene_idx = list(adata_organoid.var_names).index(gene)
            exp = adata_organoid.X[:, gene_idx]
            if hasattr(exp, 'toarray'):
                exp = exp.toarray().flatten()
            if exp.max() > 0:
                genes_in_organoid.append(gene)

    print(f"    Genes expressed in this organoid: {genes_in_organoid}")

    if not genes_in_organoid:
        print(f"    No marker genes expressed in {organoid}, skipping...")
        continue

    n_genes = len(genes_in_organoid) + 1
    n_cols = min(3, n_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_genes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Panel 1: clusters (obs column, no use_raw needed)
    try:
        sc.pl.umap(adata_organoid, color='leiden_neuronal', ax=axes[0],
                   title=f'{organoid} - Clusters', show=False, legend_loc='on data')
    except Exception as e:
        print(f"    Warning: Could not plot clusters - {e}")
        axes[0].text(0.5, 0.5, 'Clusters\nnot available',
                     ha='center', va='center', transform=axes[0].transAxes)

    # Remaining panels: gene expression — FIX: use_raw=False
    for i, gene in enumerate(genes_in_organoid, start=1):
        if i < len(axes):
            try:
                sc.pl.umap(adata_organoid, color=gene, ax=axes[i],
                           title=f'{organoid} - {gene}', show=False,
                           cmap='Reds', vmax='p95', use_raw=False)   # ← FIX
            except Exception as e:
                print(f"    Warning: Could not plot {gene} - {e}")
                axes[i].text(0.5, 0.5, f'{gene}\nnot available',
                             ha='center', va='center', transform=axes[i].transAxes)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{organoid} UMAP Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'umap_{organoid}_genes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: umap_{organoid}_genes.png")

print("\n2. CREATING SIDE-BY-SIDE UMAP COMPARISONS FOR EACH GENE...")

for gene in available_genes:
    print(f"\n  Creating UMAPs for {gene} across organoids...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, organoid in enumerate(organoids):
        if i < 4:
            organoid_mask = adata.obs['organoid'] == organoid
            adata_organoid = adata[organoid_mask, :].copy()

            if adata_organoid.shape[0] >= 5:
                try:
                    if gene in adata_organoid.var_names:
                        # FIX: use_raw=False
                        sc.pl.umap(adata_organoid, color=gene, ax=axes[i],
                                   title=f'{organoid} - {gene}', show=False,
                                   cmap='Reds', vmax='p95', use_raw=False)   # ← FIX
                    else:
                        axes[i].text(0.5, 0.5, f'{gene}\nnot in dataset',
                                     ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f'{organoid}')
                except Exception as e:
                    print(f"    Warning: Could not plot {organoid} - {e}")
                    axes[i].text(0.5, 0.5, f'Error\n{str(e)[:30]}',
                                 ha='center', va='center', transform=axes[i].transAxes)
            else:
                axes[i].text(0.5, 0.5, f'{organoid}\n(insufficient cells)',
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{organoid}')

    plt.suptitle(f'{gene} Expression Across Organoids', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'umap_comparison_{gene}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: umap_comparison_{gene}.png")

print("\n3. CREATING SUMMARY PLOTS FOR TOP MOTOR NEURON MARKERS...")

top_motor_genes = []
if 'rank_genes_groups' in adata.uns:
    result_neuronal_check = adata.uns['rank_genes_groups']
    if top_motor_cluster in result_neuronal_check['names'].dtype.names:
        for gene in result_neuronal_check['names'][top_motor_cluster][:5]:
            if gene in adata.var_names:
                top_motor_genes.append(gene)

if not top_motor_genes:
    top_motor_genes = [g for g in ['SLC5A7', 'BCL11B', 'SNAP25'] if g in adata.var_names]

if top_motor_genes:
    print(f"Top motor neuron markers: {top_motor_genes}")

    for organoid in organoids:
        organoid_mask = adata.obs['organoid'] == organoid
        adata_organoid = adata[organoid_mask, :].copy()

        if adata_organoid.shape[0] < 10:
            continue

        motor_in_organoid = [g for g in top_motor_genes if g in adata_organoid.var_names]
        if not motor_in_organoid:
            print(f"    No motor markers expressed in {organoid}")
            continue

        n_plots = len(motor_in_organoid) + 1
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        try:
            sc.pl.umap(adata_organoid, color='leiden_neuronal', ax=axes[0],
                       title=f'{organoid} - Clusters', show=False, legend_loc='on data')
        except:
            axes[0].text(0.5, 0.5, 'Clusters', ha='center', va='center')

        for i, gene in enumerate(motor_in_organoid):
            try:
                # FIX: use_raw=False
                sc.pl.umap(adata_organoid, color=gene, ax=axes[i + 1],
                           title=f'{organoid} - {gene}', show=False,
                           cmap='Reds', use_raw=False)   # ← FIX
            except:
                axes[i + 1].text(0.5, 0.5, f'{gene}', ha='center', va='center')

        plt.suptitle(f'{organoid} - Motor Neuron Markers', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'umap_{organoid}_motor_markers.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  ✅ Saved: umap_{organoid}_motor_markers.png")
else:
    print("  No motor neuron markers available")

print("\n✅ UMAP generation complete!")

# ============ CUSTOM COLORED UMAPS FOR EACH ORGANOID WITH CELL TYPE LABELS ============
print("\n" + "=" * 60)
print("CREATING CUSTOM COLORED UMAPS FOR EACH ORGANOID")
print("=" * 60)

# Define cell type colors based on your image
cell_type_colors = {
    'Motor Neurons': '#FF0000',  # Red
    'Cortical Neurons': '#FFA500',  # Orange
    'Glutamatergic': '#FFFF00',  # Yellow
    'GABAergic': '#00FF00',  # Green
    'Neural Progenitors': '#ADD8E6',  # Light Blue
    'Fibroblasts': '#FFC0CB',  # Pink
    'Glial': '#808080',  # Gray
    'Other': '#FFFFFF'  # White
}

# First, assign cell type labels to each cluster based on your marker analysis
print("\n1. ASSIGNING CELL TYPE LABELS TO CLUSTERS...")

# Create a mapping from cluster to cell type
cluster_to_celltype = {}

# From your analysis results:
# - Motor neuron cluster is {top_motor_cluster}
# - Fibroblast cluster is {top_fibro_cluster}
# - Other clusters need to be assigned based on marker expression

# Calculate scores for each cluster to determine cell type
cell_type_scores = {}

# Define marker genes for each cell type
cell_type_markers_custom = {
    'Motor Neurons': ['SLC5A7', 'BCL11B', 'CHAT'],
    'Cortical Neurons': ['SNAP25', 'RBFOX3', 'MAP2'],
    'Glutamatergic': ['GRIA2', 'SLC17A7', 'GRIN1'],
    'GABAergic': ['SLC32A1', 'GAD1', 'GAD2'],
    'Neural Progenitors': ['SOX2', 'PAX6', 'NES'],
    'Fibroblasts': ['COL1A1', 'COL3A1', 'DCN'],
    'Glial': ['GFAP', 'S100B', 'OLIG2']
}

# For each cluster, calculate score for each cell type
for cluster in neuronal_groups:
    mask = adata.obs['leiden_neuronal'] == cluster
    if mask.sum() == 0:
        continue

    scores = {}
    for cell_type, markers in cell_type_markers_custom.items():
        available_markers = [m for m in markers if m in adata.var_names]
        if available_markers:
            cluster_exp = []
            for marker in available_markers:
                marker_idx = list(adata.var_names).index(marker)
                exp = adata[mask, marker_idx].X
                if hasattr(exp, 'toarray'):
                    exp = exp.toarray().flatten()
                cluster_exp.append(exp.mean())
            scores[cell_type] = np.mean(cluster_exp)
        else:
            scores[cell_type] = 0

    # Assign cell type based on highest score
    if scores:
        best_cell_type = max(scores, key=scores.get)
        # Only assign if score is meaningful
        if scores[best_cell_type] > 0.01:
            cluster_to_celltype[cluster] = best_cell_type
        else:
            cluster_to_celltype[cluster] = 'Other'

# Force known clusters
cluster_to_celltype[top_motor_cluster] = 'Motor Neurons'
cluster_to_celltype[top_fibro_cluster] = 'Fibroblasts'

print("\nCluster to cell type mapping:")
for cluster in sorted(cluster_to_celltype.keys()):
    print(f"  Cluster {cluster}: {cluster_to_celltype[cluster]}")

# Add cell type labels to adata
adata.obs['cell_type'] = adata.obs['leiden_neuronal'].map(cluster_to_celltype).fillna('Other')

# Create a color map for all unique cell types
unique_cell_types = adata.obs['cell_type'].unique()
cell_type_color_map = {}
for ct in unique_cell_types:
    if ct in cell_type_colors:
        cell_type_color_map[ct] = cell_type_colors[ct]
    else:
        # Assign random color for any missing types
        cell_type_color_map[ct] = '#{:06x}'.format(np.random.randint(0, 0xFFFFFF))

print(f"\nCell types found: {list(unique_cell_types)}")

# ============ CREATE UMAP FOR EACH ORGANOID WITH CUSTOM COLORS ============
print("\n2. CREATING CUSTOM UMAP FOR EACH ORGANOID...")

for organoid in organoids:
    # Subset data for this organoid
    organoid_mask = adata.obs['organoid'] == organoid
    adata_organoid = adata[organoid_mask, :].copy()

    if adata_organoid.shape[0] < 10:
        print(f"  Skipping {organoid} - only {adata_organoid.shape[0]} cells")
        continue

    print(f"\n  Processing {organoid}: {adata_organoid.shape[0]} cells")

    # Get cell types present in this organoid
    organoid_cell_types = adata_organoid.obs['cell_type'].unique()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot each cell type separately with its own color
    for cell_type in organoid_cell_types:
        mask = adata_organoid.obs['cell_type'] == cell_type
        color = cell_type_color_map[cell_type]

        ax.scatter(adata_organoid.obsm['X_umap'][mask, 0],
                   adata_organoid.obsm['X_umap'][mask, 1],
                   c=color, s=10, alpha=0.8, label=cell_type, edgecolors='none')

    # Customize the plot
    ax.set_title(f'{organoid} - UMAP', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP_1', fontsize=12)
    ax.set_ylabel('UMAP_2', fontsize=12)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                       frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    plt.savefig(f'umap_{organoid}_colored.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: umap_{organoid}_colored.png")

# ============ CREATE SIDE-BY-SIDE UMAPS FOR ALL ORGANOIDS ============
print("\n3. CREATING SIDE-BY-SIDE UMAP COMPARISON...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i, organoid in enumerate(organoids):
    if i >= 4:
        break

    organoid_mask = adata.obs['organoid'] == organoid
    adata_organoid = adata[organoid_mask, :].copy()

    if adata_organoid.shape[0] < 5:
        axes[i].text(0.5, 0.5, f'{organoid}\n(insufficient cells)',
                     ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(f'{organoid}')
        continue

    # Plot each cell type
    organoid_cell_types = adata_organoid.obs['cell_type'].unique()

    for cell_type in organoid_cell_types:
        mask = adata_organoid.obs['cell_type'] == cell_type
        color = cell_type_color_map[cell_type]

        axes[i].scatter(adata_organoid.obsm['X_umap'][mask, 0],
                        adata_organoid.obsm['X_umap'][mask, 1],
                        c=color, s=8, alpha=0.8, label=cell_type if i == 0 else "",
                        edgecolors='none')

    axes[i].set_title(f'{organoid}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('UMAP_1', fontsize=10)
    axes[i].set_ylabel('UMAP_2', fontsize=10)
    axes[i].tick_params(axis='both', which='both', length=0)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

# Add a single legend for all plots
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cell_type_color_map[ct],
                      markersize=10, label=ct) for ct in unique_cell_types]
fig.legend(handles=handles, bbox_to_anchor=(0.5, 0.02), loc='lower center',
           ncol=min(4, len(unique_cell_types)), frameon=True, fancybox=True, shadow=True)

plt.suptitle('Cell Type Distribution Across Organoids', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('umap_all_organoids_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("  ✅ Saved: umap_all_organoids_comparison.png")

# ============ CREATE UMAP WITH TOP MARKER GENES FOR EACH ORGANOID ============
print("\n4. CREATING UMAPS WITH TOP MARKER GENES FOR EACH ORGANOID...")

# Get top 3 marker genes for each cell type
top_markers_by_type = {}

for cell_type, markers in cell_type_markers_custom.items():
    available = [m for m in markers if m in adata.var_names]
    if available:
        top_markers_by_type[cell_type] = available[:3]

print("\nTop markers by cell type:")
for ct, markers in top_markers_by_type.items():
    print(f"  {ct}: {markers}")

for organoid in organoids:
    organoid_mask = adata.obs['organoid'] == organoid
    adata_organoid = adata[organoid_mask, :].copy()

    if adata_organoid.shape[0] < 10:
        continue

    # Collect all unique markers to plot
    all_markers_to_plot = []
    for markers in top_markers_by_type.values():
        all_markers_to_plot.extend([m for m in markers if m in adata_organoid.var_names])
    all_markers_to_plot = list(set(all_markers_to_plot))

    if not all_markers_to_plot:
        continue

    n_plots = len(all_markers_to_plot) + 1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Panel 1: Colored by cell type
    for cell_type in adata_organoid.obs['cell_type'].unique():
        mask = adata_organoid.obs['cell_type'] == cell_type
        color = cell_type_color_map[cell_type]
        axes[0].scatter(adata_organoid.obsm['X_umap'][mask, 0],
                        adata_organoid.obsm['X_umap'][mask, 1],
                        c=color, s=5, alpha=0.8, label=cell_type, edgecolors='none')

    axes[0].set_title(f'{organoid} - Cell Types', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Remaining panels: Marker genes
    for i, gene in enumerate(all_markers_to_plot, start=1):
        if i < len(axes):
            try:
                sc.pl.umap(adata_organoid, color=gene, ax=axes[i],
                           title=f'{organoid} - {gene}', show=False,
                           cmap='Reds', vmax='p95', use_raw=False)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            except Exception as e:
                axes[i].text(0.5, 0.5, f'{gene}\nerror', ha='center', va='center')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{organoid} - Cell Types and Markers', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'umap_{organoid}_with_markers.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: umap_{organoid}_with_markers.png")

# ============ CREATE LEGEND ONLY FIGURE ============
print("\n5. CREATING LEGEND FIGURE...")

fig, ax = plt.subplots(figsize=(8, len(unique_cell_types) * 0.5))
ax.axis('off')

# Create legend elements
legend_elements = []
for cell_type, color in cell_type_color_map.items():
    if cell_type in cell_type_colors:  # Only show defined types
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=15,
                                          label=cell_type))

ax.legend(handles=legend_elements, loc='center', frameon=True,
          fancybox=True, shadow=True, fontsize=12, ncol=2)

plt.title('Cell Type Legend', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('cell_type_legend.png', dpi=300, bbox_inches='tight')
plt.show()
print("  ✅ Saved: cell_type_legend.png")

print("\n" + "=" * 60)
print("CUSTOM UMAP GENERATION SUMMARY")
print("=" * 60)
print(f"\nOrganoids processed: {len(organoids)}")
print(f"Cell types identified: {len(unique_cell_types)}")
print("\nFiles created:")
for organoid in organoids:
    print(f"  - umap_{organoid}_colored.png")
    print(f"  - umap_{organoid}_with_markers.png")
print("  - umap_all_organoids_comparison.png")
print("  - cell_type_legend.png")
print("\n✅ All custom UMAPs generated successfully!")

# ============ COMPREHENSIVE ORGANOID SIMILARITY ANALYSIS ============
print("\n" + "=" * 60)
print("COMPREHENSIVE ORGANOID SIMILARITY ANALYSIS")
print("=" * 60)

# ============ 1. CELL TYPE COMPOSITION ============
print("\n1. CELL TYPE COMPOSITION BY ORGANOID")
print("-" * 40)

# Get cell type distribution for each organoid
cell_type_distributions = []

for organoid in organoids:
    mask = adata.obs['organoid'] == organoid
    cell_type_counts = adata.obs[mask]['cell_type'].value_counts()
    cell_type_pct = (cell_type_counts / cell_type_counts.sum() * 100).round(2)

    print(f"\n{organoid} Cell Type Composition:")
    dist_dict = {'organoid': organoid}
    for ct in cell_type_pct.index:
        dist_dict[ct] = cell_type_pct[ct]
        print(f"  {ct}: {cell_type_pct[ct]}%")
    cell_type_distributions.append(dist_dict)

# Create composition dataframe
comp_df = pd.DataFrame(cell_type_distributions).fillna(0)
print("\nCell Type Composition Matrix (%):")
print(comp_df.to_string(index=False))
comp_df.to_csv('organoid_cell_type_composition.csv', index=False)

# ============ 2. ALL MARKER GENE EXPRESSION ============
print("\n2. MARKER GENE EXPRESSION BY ORGANOID")
print("-" * 40)

# Define all marker gene categories
all_marker_categories = {
    'Motor Neuron': ['SLC5A7', 'BCL11B', 'CHAT', 'ISL1', 'MNX1'],
    'Cortical': ['SNAP25', 'RBFOX3', 'MAP2', 'TUBB3', 'DCX'],
    'Glutamatergic': ['GRIA2', 'SLC17A7', 'GRIN1', 'GRIK1'],
    'GABAergic': ['SLC32A1', 'GAD1', 'GAD2', 'PVALB', 'SST'],
    'Fibroblast': ['COL1A1', 'COL3A1', 'DCN', 'FAP', 'PDGFRA'],
    'Microglia': ['CSF1R', 'TREM2', 'AIF1', 'CX3CR1'],
    'Astrocyte': ['GFAP', 'S100B', 'AQP4', 'ALDH1L1'],
    'Oligodendrocyte': ['MBP', 'PLP1', 'MOG', 'OLIG2'],
    'Neural Progenitor': ['SOX2', 'PAX6', 'NES', 'MKI67'],
    'Synaptic': ['SYP', 'STX1A', 'VAMP2', 'DLG4']
}

# Collect expression data for all markers
marker_expression = []

for organoid in organoids:
    mask = adata.obs['organoid'] == organoid
    organoid_data = adata[mask, :]

    exp_dict = {'organoid': organoid}

    for category, markers in all_marker_categories.items():
        for marker in markers:
            if marker in organoid_data.var_names:
                marker_idx = list(organoid_data.var_names).index(marker)
                exp = organoid_data.X[:, marker_idx]
                if hasattr(exp, 'toarray'):
                    exp = exp.toarray().flatten()
                exp_dict[f'{category}_{marker}'] = exp.mean()
                exp_dict[f'{category}_{marker}_pct'] = (exp > 0).mean() * 100

    marker_expression.append(exp_dict)

marker_df = pd.DataFrame(marker_expression).fillna(0)
print(f"\nCollected expression data for {len(marker_df.columns) - 1} markers")
marker_df.to_csv('organoid_marker_expression.csv', index=False)

# ============ 3. GLOBAL GENE EXPRESSION CORRELATION ============
print("\n3. GLOBAL GENE EXPRESSION CORRELATION BETWEEN ORGANOIDS")
print("-" * 40)

# Get mean expression per gene for each organoid
organoid_expression_profiles = []

for organoid in organoids:
    mask = adata.obs['organoid'] == organoid
    organoid_data = adata[mask, :]

    # Calculate mean expression for all genes
    mean_exp = np.array(organoid_data.X.mean(axis=0)).flatten()
    organoid_expression_profiles.append(mean_exp)

# Calculate correlation matrix
expression_correlation = np.corrcoef(organoid_expression_profiles)
corr_df = pd.DataFrame(expression_correlation,
                       index=organoids,
                       columns=organoids)

print("\nGlobal Gene Expression Correlation Matrix:")
print(corr_df.round(3))

# ============ 4. CLUSTER SIMILARITY ============
print("\n4. CLUSTER COMPOSITION SIMILARITY")
print("-" * 40)

# Get cluster distribution for each organoid
cluster_distributions = []

for organoid in organoids:
    mask = adata.obs['organoid'] == organoid
    cluster_counts = adata.obs[mask]['leiden_neuronal'].value_counts()
    cluster_pct = (cluster_counts / cluster_counts.sum() * 100)

    dist_dict = {'organoid': organoid}
    for cluster in cluster_pct.index:
        dist_dict[f'cluster_{cluster}'] = cluster_pct[cluster]
    cluster_distributions.append(dist_dict)

cluster_df = pd.DataFrame(cluster_distributions).fillna(0)
print("\nCluster Distribution Matrix (%):")
print(cluster_df.round(2).to_string(index=False))

# ============ 5. MULTI-DIMENSIONAL SIMILARITY METRICS ============
print("\n5. MULTI-DIMENSIONAL SIMILARITY METRICS")
print("-" * 40)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# Combine all features for comprehensive comparison
# Features: cell type composition + marker expression + cluster distribution

# Prepare feature matrix
feature_dfs = []

# Cell type composition features
comp_features = comp_df.drop('organoid', axis=1)
feature_dfs.append(comp_features)

# Marker expression features (mean expression only, not percentages)
marker_mean_cols = [col for col in marker_df.columns if 'pct' not in col and col != 'organoid']
marker_features = marker_df[marker_mean_cols]
feature_dfs.append(marker_features)

# Cluster distribution features
cluster_features = cluster_df.drop('organoid', axis=1)
feature_dfs.append(cluster_features)

# Combine all features
all_features = pd.concat(feature_dfs, axis=1)
print(f"Total features for comparison: {all_features.shape[1]}")

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_features)

# Calculate different similarity metrics
euclidean_dist = euclidean_distances(features_scaled)
manhattan_dist = manhattan_distances(features_scaled)
cosine_sim = cosine_similarity(features_scaled)

# Create dataframes
euclidean_df = pd.DataFrame(euclidean_dist, index=organoids, columns=organoids)
manhattan_df = pd.DataFrame(manhattan_dist, index=organoids, columns=organoids)
cosine_df = pd.DataFrame(cosine_sim, index=organoids, columns=organoids)

print("\nEuclidean Distance (smaller = more similar):")
print(euclidean_df.round(3))

print("\nManhattan Distance (smaller = more similar):")
print(manhattan_df.round(3))

print("\nCosine Similarity (closer to 1 = more similar):")
print(cosine_df.round(3))

# ============ 6. HIERARCHICAL CLUSTERING ============
print("\n6. HIERARCHICAL CLUSTERING OF ORGANOIDS")
print("-" * 40)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Perform hierarchical clustering
linkage_matrix = linkage(features_scaled, method='ward')

# Plot dendrogram
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(linkage_matrix, labels=organoids, ax=ax, leaf_rotation=45)
ax.set_title('Hierarchical Clustering of Organoids (All Features)', fontsize=14, fontweight='bold')
ax.set_xlabel('Organoid')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig('organoid_comprehensive_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

# Determine clusters at different thresholds
for threshold in [5, 10, 15]:
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    print(f"\nAt distance threshold {threshold}: {len(set(clusters))} groups")
    for i, org in enumerate(organoids):
        print(f"  {org}: Group {clusters[i]}")

# ============ 7. PCA VISUALIZATION ============
print("\n7. PCA OF ORGANOIDS (All Features)")
print("-" * 40)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                     c=range(len(organoids)), s=300, cmap='Set1', alpha=0.7)

# Add labels
for i, organoid in enumerate(organoids):
    ax.annotate(organoid, (pca_result[i, 0], pca_result[i, 1]),
                fontsize=12, fontweight='bold', ha='center')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
ax.set_title('PCA of Organoids Based on All Features', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('organoid_comprehensive_pca.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nExplained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

# ============ 8. SIMILARITY NETWORK ============
print("\n8. ORGANOID SIMILARITY NETWORK")
print("-" * 40)

import networkx as nx

# Create similarity graph (threshold for edges)
threshold = np.percentile(euclidean_dist[euclidean_dist > 0], 25)  # Top 25% most similar
G = nx.Graph()

# Add nodes
for i, org in enumerate(organoids):
    G.add_node(org, size=adata.obs[adata.obs['organoid'] == org].shape[0])

# Add edges for similar organoids
for i in range(len(organoids)):
    for j in range(i + 1, len(organoids)):
        if euclidean_dist[i, j] < threshold:
            G.add_edge(organoids[i], organoids[j], weight=1 / euclidean_dist[i, j])

# Draw network
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
node_sizes = [G.nodes[org]['size'] / 50 for org in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                       node_size=node_sizes, alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

ax.set_title('Organoid Similarity Network', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('organoid_similarity_network.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ 9. RANK SIMILARITY ============
print("\n9. ORGANOID SIMILARITY RANKING")
print("-" * 40)

# Create a combined similarity score (average of normalized metrics)
# Normalize distances (0-1, where 0 = most similar)
norm_euclidean = euclidean_dist / euclidean_dist.max()
norm_manhattan = manhattan_dist / manhattan_dist.max()
norm_cosine = 1 - cosine_sim  # Convert to distance

combined_dist = (norm_euclidean + norm_manhattan + norm_cosine) / 3

# Set diagonal to infinity
np.fill_diagonal(combined_dist, np.inf)

# Find all pairs and their distances
pairs = []
for i in range(len(organoids)):
    for j in range(i + 1, len(organoids)):
        pairs.append({
            'pair': f"{organoids[i]} - {organoids[j]}",
            'distance': combined_dist[i, j],
            'org1': organoids[i],
            'org2': organoids[j]
        })

pairs_df = pd.DataFrame(pairs)
pairs_df = pairs_df.sort_values('distance')

print("\nOrganoid Similarity Ranking (most similar to least):")
for idx, row in pairs_df.iterrows():
    print(f"  {row['pair']}: similarity score = {row['distance']:.3f}")

# ============ 10. FINAL ANSWER ============
print("\n" + "=" * 60)
print("FINAL ANSWER: WHICH ORGANOIDS ARE MOST SIMILAR?")
print("=" * 60)

most_similar = pairs_df.iloc[0]
second_similar = pairs_df.iloc[1]
least_similar = pairs_df.iloc[-1]

print(f"\n✅ MOST SIMILAR: {most_similar['pair']}")
print(f"   Similarity score: {most_similar['distance']:.3f} (lower = more similar)")
print(f"\n   {most_similar['org1']} vs {most_similar['org2']}:")

# Show comparison for the most similar pair
org1_data = comp_df[comp_df['organoid'] == most_similar['org1']].iloc[0]
org2_data = comp_df[comp_df['organoid'] == most_similar['org2']].iloc[0]

print("\n   Cell Type Comparison:")
for ct in comp_df.columns[1:]:
    val1 = org1_data[ct]
    val2 = org2_data[ct]
    diff = abs(val1 - val2)
    print(f"     {ct}: {val1:.1f}% vs {val2:.1f}% (diff: {diff:.1f}%)")

print(f"\n\n📊 SECOND MOST SIMILAR: {second_similar['pair']}")
print(f"   Similarity score: {second_similar['distance']:.3f}")

print(f"\n📉 LEAST SIMILAR: {least_similar['pair']}")
print(f"   Similarity score: {least_similar['distance']:.3f}")

# ============ 11. VISUALIZATION OF RESULTS ============
print("\n11. VISUALIZING SIMILARITY RESULTS")
print("-" * 40)

# Create a comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Panel 1: Cell type composition heatmap
ax1 = plt.subplot(2, 3, 1)
cell_type_data = comp_df.set_index('organoid')
im1 = ax1.imshow(cell_type_data.T, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(len(cell_type_data.index)))
ax1.set_xticklabels(cell_type_data.index, rotation=45, ha='right')
ax1.set_yticks(range(len(cell_type_data.columns)))
ax1.set_yticklabels(cell_type_data.columns, fontsize=8)
ax1.set_title('Cell Type Composition')
plt.colorbar(im1, ax=ax1)

# Panel 2: PCA plot
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(pca_result[:, 0], pca_result[:, 1],
            c=range(len(organoids)), s=200, cmap='Set1', alpha=0.7)
for i, org in enumerate(organoids):
    ax2.annotate(org, (pca_result[i, 0], pca_result[i, 1]),
                 fontsize=10, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
ax2.set_title('PCA of Organoids')
ax2.grid(True, alpha=0.3)

# Panel 3: Similarity matrix
ax3 = plt.subplot(2, 3, 3)
im3 = ax3.imshow(combined_dist, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=1)
ax3.set_xticks(range(len(organoids)))
ax3.set_yticks(range(len(organoids)))
ax3.set_xticklabels(organoids, rotation=45, ha='right')
ax3.set_yticklabels(organoids)
ax3.set_title('Combined Similarity Matrix\n(darker = more similar)')
plt.colorbar(im3, ax=ax3)

# Panel 4: Dendrogram
ax4 = plt.subplot(2, 3, 4)
dendrogram(linkage_matrix, labels=organoids, ax=ax4, leaf_rotation=45)
ax4.set_title('Hierarchical Clustering')
ax4.set_xlabel('Organoid')
ax4.set_ylabel('Distance')

# Panel 5: Marker expression heatmap (top markers)
ax5 = plt.subplot(2, 3, 5)
top_markers = ['SLC5A7', 'BCL11B', 'SNAP25', 'GRIA2', 'SLC32A1',
               'COL1A1', 'COL3A1', 'DCN', 'CSF1R', 'TREM2']
marker_heatmap_data = []
for marker in top_markers:
    if f'Motor Neuron_{marker}' in marker_df.columns:
        marker_heatmap_data.append(marker_df[f'Motor Neuron_{marker}'].values)
    elif f'Cortical_{marker}' in marker_df.columns:
        marker_heatmap_data.append(marker_df[f'Cortical_{marker}'].values)
    else:
        marker_heatmap_data.append([0, 0, 0, 0])

im5 = ax5.imshow(marker_heatmap_data, cmap='viridis', aspect='auto')
ax5.set_xticks(range(len(organoids)))
ax5.set_xticklabels(organoids, rotation=45, ha='right')
ax5.set_yticks(range(len(top_markers)))
ax5.set_yticklabels(top_markers)
ax5.set_title('Key Marker Expression')
plt.colorbar(im5, ax=ax5)

# Panel 6: Similarity ranking
ax6 = plt.subplot(2, 3, 6)
colors = ['green' if i == 0 else 'lightgreen' if i == 1 else 'lightcoral' if i == len(pairs_df) - 1 else 'lightgray'
          for i in range(len(pairs_df))]
bars = ax6.barh(range(len(pairs_df)), pairs_df['distance'].values, color=colors)
ax6.set_yticks(range(len(pairs_df)))
ax6.set_yticklabels(pairs_df['pair'].values, fontsize=9)
ax6.set_xlabel('Similarity Distance (lower = more similar)')
ax6.set_title('Organoid Similarity Ranking')
ax6.invert_xaxis()  # Most similar at top

plt.tight_layout()
plt.savefig('organoid_comprehensive_similarity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Comprehensive analysis complete!")
print("Files saved:")
print("  - organoid_cell_type_composition.csv")
print("  - organoid_marker_expression.csv")
print("  - organoid_comprehensive_clustering.png")
print("  - organoid_comprehensive_pca.png")
print("  - organoid_similarity_network.png")
print("  - organoid_comprehensive_similarity.png")

print("\n✅ Analysis complete!")
print("=" * 60)