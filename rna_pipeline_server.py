# ============================================================
# RNA SPATIAL ANALYSIS PIPELINE  (condensed, full analysis)
# ============================================================
import warnings; warnings.filterwarnings('ignore')
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist
from matplotlib.patches import Patch

try: import mygene;  MGENE_AVAILABLE = True
except ImportError:  MGENE_AVAILABLE = False

try: import gseapy as gp; GSEAPY_AVAILABLE = True
except ImportError:        GSEAPY_AVAILABLE = False

import os

# ============================================================
# SERVER PATH CONFIGURATION  ← edit these two lines only
# ============================================================
FILE_PATH    = r'/home/s.tesema1/C06018D5.bin50_1.0.h5ad'
RESULTS_BASE = r'/home/s.tesema1/figures_for_RNA'
os.makedirs(RESULTS_BASE, exist_ok=True)

def R(fname):
    """Return full path inside RESULTS_BASE."""
    return os.path.join(RESULTS_BASE, fname)


# ── helpers ──────────────────────────────────────────────────
def step(n, title):
    print(f"\n{'='*60}\nSTEP {n}: {title}\n{'='*60}")

def get_exp(adata, gene):
    """Return dense 1-D expression array for a gene."""
    x = adata.X[:, list(adata.var_names).index(gene)]
    return x.toarray().flatten() if hasattr(x, 'toarray') else np.asarray(x).flatten()

def score_clusters(adata, groupby, markers):
    """Mean expression score per cluster for a list of markers."""
    available = [m for m in markers if m in adata.var_names]
    if not available:
        return None, available
    scores = {}
    for grp in adata.obs[groupby].unique():
        mask = adata.obs[groupby] == grp
        scores[grp] = np.mean([get_exp(adata, m)[mask].mean() for m in available])
    return pd.Series(scores).sort_values(ascending=False), available

def spatial_scatter(ax, coords, c, title, s=5, cmap='tab10', **kw):
    sc_ = ax.scatter(coords[:, 0], coords[:, 1], c=c, cmap=cmap, s=s, alpha=0.7, **kw)
    ax.set_title(title); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    return sc_

def save_show(fig, R(fname), dpi=150:
    fig.tight_layout(pad=2.5)
    fig.savefig(fname, dpi=dpi, bbox_inches='tight'); plt.show(); plt.close(fig)

# ── STEP 1-6: load / filter / normalise / HVG / dimred / cluster ─
step(1, "READING DATA")
adata = sc.read_h5ad(FILE_PATH)
print(adata)
if 'real_gene_name' in adata.var.columns:
    print(adata.var[['real_gene_name']].head())

step(2, "BASIC FILTERING")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs.pct_counts_mt < 10].copy()
print(f"After MT filter: {adata.shape}")

step(3, "NORMALIZATION")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

step(4, "HIGHLY VARIABLE GENES")
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=4000,
                             min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable].copy()
sc.pl.highly_variable_genes(adata)
print(f"After HVG: {adata.shape}")

step(5, "DIMENSIONALITY REDUCTION")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata, log=True)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)

step(6, "CLUSTERING")
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color="leiden")

# ── STEP 7: spatial analysis ──────────────────────────────────
step(7, "SPATIAL ANALYSIS")
if 'spatial' in adata.obsm:
    coords = adata.obsm['spatial']

    # 7a: cluster + organoid maps
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sc1 = spatial_scatter(axes[0], coords,
                          adata.obs['leiden'].astype('category').cat.codes, 'Spatial Clusters')
    axes[0].legend(*sc1.legend_elements(), title="Cluster", bbox_to_anchor=(1.05,1), loc='upper left')
    if 'orig.ident' in adata.obs.columns:
        uniq = adata.obs['orig.ident'].unique()
        clrs = plt.cm.tab10(np.linspace(0, 1, len(uniq)))
        for i, org in enumerate(uniq):
            m = adata.obs['orig.ident'] == org
            axes[1].scatter(coords[m, 0], coords[m, 1], c=[clrs[i]], s=5, alpha=0.7, label=org)
        axes[1].set_title('Organoid Identity'); axes[1].set_aspect('equal')
        axes[1].legend(bbox_to_anchor=(1.05,1), loc='upper left')
    save_show(fig, R('spatial_clusters.png')

    # 7b: cluster composition
    if 'orig.ident' in adata.obs.columns:
        cluster_by_organoid = pd.crosstab(adata.obs["leiden"], adata.obs["orig.ident"])
        cluster_by_organoid.to_csv(R("cluster_by_organoid.csv"))
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_by_organoid.T.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_title('Cluster Composition by Organoid'); ax.legend(title='Cluster', bbox_to_anchor=(1.05,1))
        save_show(fig, R('cluster_composition.png')

    # 7c: spatial metrics
    spatial_metrics = []
    for cluster in adata.obs['leiden'].unique():
        m = adata.obs['leiden'] == cluster; cr = coords[m]
        d = {'cluster': cluster, 'n_spots': m.sum(),
             'centroid_x': cr[:, 0].mean(), 'centroid_y': cr[:, 1].mean(),
             'spread_x': cr[:, 0].std(), 'spread_y': cr[:, 1].std()}
        try:    d['area_pixels'] = ConvexHull(cr).volume if len(cr) > 3 else np.nan
        except: d['area_pixels'] = np.nan
        spatial_metrics.append(d)
        print(f"Cluster {cluster}: {d['n_spots']} spots, centroid ({d['centroid_x']:.1f},{d['centroid_y']:.1f})")
    pd.DataFrame(spatial_metrics).to_csv(R("spatial_metrics.csv"), index=False)

    # 7d: individual cluster distribution
    uniq_cl = adata.obs['leiden'].unique()
    n_cols = min(3, len(uniq_cl)); n_rows = (len(uniq_cl) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
    axes = np.array(axes).flatten()
    for i, cl in enumerate(uniq_cl):
        m = adata.obs['leiden'] == cl
        axes[i].scatter(coords[:, 0], coords[:, 1], c='lightgray', s=1, alpha=0.3)
        axes[i].scatter(coords[m, 0], coords[m, 1], c='red', s=2, alpha=0.5)
        axes[i].set_title(f'Cluster {cl}'); axes[i].set_aspect('equal')
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    save_show(fig, R('cluster_distribution.png')

    # 7e: radial zones
    center = coords.mean(axis=0)
    adata.obs['distance_from_center'] = cdist([center], coords)[0]
    adata.obs['spatial_zone'] = pd.cut(adata.obs['distance_from_center'], bins=5,
                                       labels=['core','inner','mid','outer','periphery'])
    fig, ax = plt.subplots(figsize=(10, 6))
    for cl in adata.obs['leiden'].unique():
        ax.hist(adata.obs.loc[adata.obs['leiden']==cl, 'distance_from_center'],
                bins=50, alpha=0.5, label=f'Cluster {cl}')
    ax.set(xlabel='Distance from Center', ylabel='Spots'); ax.legend()
    save_show(fig, R('radial_distribution.png')

    sc.tl.rank_genes_groups(adata, 'spatial_zone', method='wilcoxon', use_raw=False)
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, title='Spatial Zone Markers')
else:
    print(f"No spatial coords found. Available: {list(adata.obsm.keys())}")

# ── STEP 8: marker genes ──────────────────────────────────────
step(8, "MARKER GENE ANALYSIS")
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=False)
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
for g in groups:
    for gene, sc_, lfc in zip(result['names'][g][:10], result['scores'][g][:10], result['logfoldchanges'][g][:10]):
        print(f"  Cluster {g} | {gene}: score={sc_:.3f} log2FC={lfc:.3f}")
pd.DataFrame({g+'_'+k: result[k][g] for g in groups
              for k in ['names','scores','pvals','pvals_adj','logfoldchanges']}
             ).to_csv(R("marker_genes.csv"))
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, title="Top Marker Genes per Cluster")

# ── STEP 9: convert gene symbols ─────────────────────────────
step(9, "CONVERTING GENE SYMBOLS")
if 'real_gene_name' in adata.var.columns:
    adata.var['ensg_id'] = adata.var_names.copy()
    adata.var_names = adata.var["real_gene_name"].values
    print(f"Converted: {adata.var_names[:10]}")
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=False)
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names

# ── STEP 10: cell type scoring ────────────────────────────────
step(10, "CELL TYPE MARKER ANALYSIS")
sc.settings.figdir = RESULTS_BASE
cell_type_markers = {
    'Neurons':          ['SNAP25','SYT1','STMN2','RBFOX3','MAP2','TUBB3','DCX'],
    'Astrocytes':       ['GFAP','S100B','AQP4','ALDH1L1','SOX9'],
    'Oligodendrocytes': ['MBP','PLP1','MOG','OLIG2','SOX10'],
    'Microglia':        ['CSF1R','C1QB','PTPRC','CX3CR1','TREM2','AIF1'],
    'Fibroblasts':      ['COL1A1','COL3A1','DCN','FAP','PDGFRA'],
    'Endothelial':      ['PECAM1','CLDN5','FLT1','VWF','CDH5'],
    'Neural Progenitors':['SOX2','PAX6','NES','MKI67','HES1'],
    'Radial Glia':      ['FABP7','VIM','GLI3'],
}
cell_type_scores = {}
for ct, mkrs in cell_type_markers.items():
    avail = [m for m in mkrs if m in adata.var_names]
    if len(avail) < 2: continue
    print(f"{ct}: {len(avail)}/{len(mkrs)} found: {avail}")
    try:
        sc.pl.matrixplot(adata, avail, groupby='leiden', title=f'{ct} Markers',
                         cmap='Blues', standard_scale='var', save=f'_{ct}_markers.png', show=False, use_raw=False)
    except:
        pass
    for g in groups:
        m = adata.obs['leiden'] == g
        cell_type_scores.setdefault(ct, {})[g] = np.mean([get_exp(adata, mk)[m].mean() for mk in avail])
if cell_type_scores:
    cell_type_df = pd.DataFrame(cell_type_scores).round(3)
    cell_type_df.to_csv(R('cell_type_scores.csv'))
    print(cell_type_df)

# ── STEP 11: pathway enrichment ───────────────────────────────
step(11, "PATHWAY ENRICHMENT")
if GSEAPY_AVAILABLE:
    for cluster in groups:
        try:
            enr = gp.enrichr(gene_list=result['names'][cluster][:100].tolist(),
                             gene_sets=['GO_Biological_Process_2023','KEGG_2021_Human','Reactome_2022'],
                             organism='human', outdir=R(f'enrichment_cluster_{cluster}'))
            print(f"Cluster {cluster}:\n{enr.results.head(5)[['Term','Adjusted P-value']]}")
            enr.results.to_csv(R(f'enrichment_cluster_{cluster}.csv'), index=False)
        except Exception as e:
            print(f"Cluster {cluster} enrichment failed: {e}")

# ── STEP 12: gene descriptions ────────────────────────────────
step(12, "GENE DESCRIPTIONS")
if MGENE_AVAILABLE:
    all_top = list({g for cl in groups for g in result['names'][cl][:20]
                    if not pd.isna(g) and g != ''})
    try:
        mg = mygene.MyGeneInfo()
        ginfo = [g for g in mg.querymany(all_top, scopes='symbol',
                                          fields='name,summary,entrezgene', species='human')
                 if 'notfound' not in g]
        if ginfo:
            gdf = pd.DataFrame(ginfo)[[c for c in ['query','name','summary','entrezgene'] if c in pd.DataFrame(ginfo).columns]]
            gdf.columns = ['gene'] + list(gdf.columns[1:])
            gdf['top_marker_in_cluster'] = [next((cl for cl in groups if gene in result['names'][cl][:20]), 'unknown')
                                             for gene in gdf['gene']]
            gdf.to_csv(R('gene_descriptions.csv'), index=False)
            print(gdf.head())
    except Exception as e:
        print(f"Gene description error: {e}")

# ── STEP 13: publication figure ───────────────────────────────
step(13, "PUBLICATION FIGURES")
fig = plt.figure(figsize=(26, 20))

ax1 = plt.subplot(2,3,1)
ax1.scatter(coords[:,0], coords[:,1], c=adata.obs['leiden'].astype('category').cat.codes,
            cmap='tab10', s=3, alpha=0.7); ax1.set_title('A: Spatial Clusters', fontweight='bold')
ax1.set_aspect('equal'); ax1.axis('off')

ax2 = plt.subplot(2,3,2)
sc.pl.umap(adata, color='leiden', ax=ax2, show=False)
ax2.set_title('B: UMAP Clusters', fontweight='bold')

ax3 = plt.subplot(2,3,3)
_nc = plt.cm.tab10(np.linspace(0,1,len(groups)))
_bg, _bs, _bl = zip(*[(result['names'][g][r], result['scores'][g][r], g)
                       for g in groups for r in range(3)])
_yp = np.arange(len(_bg))
ax3.barh(_yp, _bs, color=[_nc[list(groups).index(l)] for l in _bl], alpha=0.8)
ax3.set_yticks(_yp); ax3.set_yticklabels(_bg, fontsize=8); ax3.set_xlabel('Wilcoxon score')
ax3.set_title('C: Top Marker Genes', fontweight='bold')
ax3.legend(handles=[Patch(color=_nc[i], label=f'Cluster {g}') for i,g in enumerate(groups)],
           fontsize=7, loc='lower right')

ax4 = plt.subplot(2,3,4)
rep_mkrs = [result['names'][g][0] for g in groups[:3]]
if len(rep_mkrs) >= 3:
    rgb = np.zeros((coords.shape[0], 3))
    for i, (mk, ch) in enumerate(zip(rep_mkrs[:3], range(3))):
        if mk in adata.var_names:
            e = get_exp(adata, mk); e = (e-e.min())/(e.max()-e.min()+1e-8); rgb[:,ch] = e
    ax4.scatter(coords[:,0], coords[:,1], c=rgb, s=3, alpha=0.7)
ax4.set_title('D: Marker Overlay', fontweight='bold'); ax4.set_aspect('equal'); ax4.axis('off')

ax5 = plt.subplot(2,3,5)
if cell_type_scores:
    cell_type_df.T.plot(kind='bar', ax=ax5, legend=False)
    ax5.set_title('E: Cell Type Scores', fontweight='bold')

ax6 = plt.subplot(2,3,6)
if 'cluster_by_organoid' in dir():
    cluster_by_organoid.T.plot(kind='bar', stacked=True, ax=ax6, colormap='tab10')
    ax6.set_title('F: Cluster Composition', fontweight='bold')
    ax6.legend(title='Cluster', bbox_to_anchor=(1.05,1))

save_show(fig, R('publication_figure.png'), dpi=300

# ── STEP 14: spatial marker expression ───────────────────────
step(14, "SPATIAL EXPRESSION OF KEY MARKERS")
neural_markers = ["SOX2","PAX6","MAP2","TUBB3","MKI67","GFAP","NES","DCX","S100B",
                  "OLIG2","EMX1","DLX2","CSF1R","C1QB","COL1A1","COL3A1","DCN","SNAP25","SYT1"]
avail_nm = [m for m in neural_markers if m in adata.var_names]
print(f"Available neural markers: {avail_nm}")

if avail_nm and 'spatial' in adata.obsm:
    n_cols = 3; n_rows = (len(avail_nm)+2)//3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
    axes = axes.flatten()
    for i, mk in enumerate(avail_nm):
        e = get_exp(adata, mk); e = (e-e.min())/(e.max()-e.min()+1e-8)
        sc_ = spatial_scatter(axes[i], coords, e, f'{mk} Expression', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(sc_, ax=axes[i])
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    save_show(fig, R('spatial_marker_expression.png')

    if 'COL1A1' in avail_nm:
        e = get_exp(adata, 'COL1A1')
        fig, (a1,a2) = plt.subplots(1,2,figsize=(16,7))
        sc_ = spatial_scatter(a1, coords, e, 'COL1A1 Expression', cmap='Reds'); plt.colorbar(sc_, ax=a1)
        spatial_scatter(a2, coords, adata.obs['leiden'].astype('category').cat.codes, 'Clusters + COL1A1 High')
        hi = e > np.percentile(e, 90)
        a2.scatter(coords[hi,0], coords[hi,1], c='red', s=10, alpha=0.8, label='COL1A1 >90th %ile')
        a2.legend(); save_show(fig, R('COL1A1_detailed.png')
        print(f"COL1A1: mean={e.mean():.4f}, max={e.max():.4f}, %expressing={(e>0).mean()*100:.1f}%")

# ── STEP 15: additional visualisations ───────────────────────
step(15, "ADDITIONAL VISUALIZATIONS")
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, use_raw=False, show=False)
sc.pl.violin(adata, keys=[result['names'][g][0] for g in groups[:3]],
             groupby='leiden', rotation=45, use_raw=False)

# ── STEP 16: save ─────────────────────────────────────────────
step(16, "SAVING RESULTS")
for col in adata.obs.columns:
    if pd.api.types.is_string_dtype(adata.obs[col]): adata.obs[col] = adata.obs[col].astype('object')
for col in adata.var.columns:
    if pd.api.types.is_string_dtype(adata.var[col]): adata.var[col] = adata.var[col].astype('object')
adata.write(R("processed_spatial_data.h5ad"))
test = sc.read_h5ad(R("processed_spatial_data.h5ad"), backed='r')
print(f"Saved. Genes: {test.var_names[:10]}"); test.file.close()

# ── STEP 17: SpaGCN ───────────────────────────────────────────
step(17, "SPATIAL DOMAIN DETECTION (SpaGCN)")
try:
    import SpaGCN as spg
    x_arr, y_arr = adata.obsm['spatial'][:,0], adata.obsm['spatial'][:,1]
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    adj = spg.calculate_adj_matrix(x=x_arr, y=y_arr, histology=False)
    l = spg.search_l(p=0.5, adj=adj, start=0.01, end=1000, tol=0.01)
    res = spg.search_res(adata, adj, l, target_num=5, start=0.1, step=0.1, tol=5e-3, lr=0.05, max_epochs=20)
    clf = spg.SpaGCN(); clf.set_library_size(adata.obs['total_counts'].values)
    clf.train(X, adj, init_spa=True, init=None, res=res, l=l)
    y_pred = clf.predict()
    adata.obs['spatial_domain'] = y_pred.astype(str)
    print(f"Detected {len(np.unique(y_pred))} spatial domains")
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    for ax, col, title in zip(axes, ['spatial_domain','leiden'],
                               ['SpaGCN Domains','Leiden Clusters']):
        spatial_scatter(ax, adata.obsm['spatial'], adata.obs[col].astype('category').cat.codes, title)
    save_show(fig, R('spatial_domains_comparison.png')
    cross_tab = pd.crosstab(adata.obs['leiden'], adata.obs['spatial_domain'])
    cross_tab.to_csv(R('leiden_vs_spatial_domains.csv')); print(cross_tab)
    sc.tl.rank_genes_groups(adata, 'spatial_domain', method='wilcoxon', use_raw=False)
    sc.pl.rank_genes_groups_dotplot(adata, groupby='spatial_domain', n_genes=5)
    result_spatial = adata.uns['rank_genes_groups']
    pd.DataFrame({g+'_'+k: result_spatial[k][g] for g in result_spatial['names'].dtype.names
                  for k in ['names','scores','pvals_adj','logfoldchanges']
                  }).to_csv(R('spatial_domain_markers.csv'))
except ImportError: print("SpaGCN not installed: pip install SpaGCN")
except Exception as e: import traceback; print(e); traceback.print_exc()

# ── STEP 18: Moran's I ────────────────────────────────────────
step(18, "SPATIALLY VARIABLE GENES (Moran's I)")
try:
    import squidpy as sq
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)
    sq.gr.spatial_autocorr(adata, mode='moran', genes=adata.var_names[:1000], n_perms=100)
    mr = adata.uns['moranI']
    if 'gene' not in mr.columns and 'genes' not in mr.columns:
        mr = mr.copy(); mr['gene'] = mr.index
    gene_col = 'gene' if 'gene' in mr.columns else 'genes'
    top20 = mr.nsmallest(20, 'pval_norm'); print(top20[['I','pval_norm',gene_col]])
    mr.to_csv(R('spatially_variable_genes.csv'))
    fig, axes = plt.subplots(2,3,figsize=(21,14)); axes = axes.flatten()
        if gene in adata.var_names:
            e = get_exp(adata, gene)
            sc_ = axes[i].scatter(coords[:,0], coords[:,1], c=e, cmap='viridis', s=3, alpha=0.7)
            axes[i].set_title(f"{gene}\nI={mr.loc[mr[gene_col]==gene,'I'].values[0]:.3f}")
            axes[i].set_aspect('equal'); plt.colorbar(sc_, ax=axes[i])
    for j in range(i+1,6): axes[j].set_visible(False)
    save_show(fig, R('spatially_variable_genes.png')
    top10 = mr.nsmallest(10,'pval_norm')
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.barh(range(10), top10['I'].values)
    ax.set_yticks(range(10)); ax.set_yticklabels(top10[gene_col].values)
    ax.set_xlabel("Moran's I"); ax.set_title('Top 10 Spatially Variable Genes')
    for bar, pv in zip(bars, top10['pval_norm'].values):
        bar.set_color('darkred' if pv<.001 else 'red' if pv<.01 else 'salmon' if pv<.05 else 'gray')
    save_show(fig, R('top_spatial_genes_barplot.png')
except ImportError: print("squidpy not installed: pip install squidpy")
except Exception as e: import traceback; print(e); traceback.print_exc()

# ── STEP 19: Hotspot ──────────────────────────────────────────
step(19, "HOTSPOT GENE MODULE DETECTION")
try:
    import hotspot
    hs = hotspot.Hotspot(adata, layer='X', model='danb', latent_obsm='X_pca',
                         umi_counts=adata.obs['total_counts'])
    hs.create_knn_graph(latent_obsm='spatial', n_neighbors=30)
    hs.compute_autocorrelations(jobs=4)
    print(f"Sig genes: {len(hs.results[hs.results.FDR < 0.05])}")
    hs.create_modules(min_gene_threshold=5, core_only=True, fdr_threshold=0.05)
    hs.modules.to_csv(R('hotspot_modules.csv'))
    ms = hs.module_scores()
    fig, axes = plt.subplots(2,3,figsize=(21,14)); axes = axes.flatten()
    for i, col in enumerate(ms.columns[:6]):
        sc_ = axes[i].scatter(coords[:,0], coords[:,1], c=ms[col], cmap='RdBu_r', s=3, alpha=0.7)
        axes[i].set_title(f'Module {i}'); axes[i].set_aspect('equal'); plt.colorbar(sc_, ax=axes[i])
    save_show(fig, R('hotspot_modules.png')
except ImportError: print("hotspot not installed: pip install hotspot")

# ── STEP 20: spatial domain reproducibility ───────────────────
step(20, "SPATIAL DOMAIN REPRODUCIBILITY")
if 'spatial_domain' in adata.obs.columns and 'orig.ident' in adata.obs.columns:
    dbo = pd.crosstab(adata.obs['spatial_domain'], adata.obs['orig.ident'])
    dbo.to_csv(R('spatial_domains_by_organoid.csv'))
    dbp = dbo.div(dbo.sum(axis=0), axis=1)
    fig, axes = plt.subplots(1,2,figsize=(18,8))
    dbp.T.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10', title='Proportions')
    save_show(fig, R('spatial_domain_reproducibility.png')
    from scipy.stats import chi2_contingency
    _, p, *_ = chi2_contingency(dbo)
    print(f"Chi-square p={p:.4f} → {'sig diff' if p<0.05 else 'reproducible'}")

# ── STEP 21: spatial domain markers ──────────────────────────
step(21, "SPATIAL DOMAIN MARKERS")
if 'spatial_domain' in adata.obs.columns:
    sc.tl.rank_genes_groups(adata, 'spatial_domain', method='wilcoxon', use_raw=False)
    rs = adata.uns['rank_genes_groups']; sg = rs['names'].dtype.names
    for g in sg:
        print(f"Domain {g}: " + ", ".join(f"{n}({l:.2f})"
              for n,l in zip(rs['names'][g][:5], rs['logfoldchanges'][g][:5])))
    pd.DataFrame({g+'_'+k: rs[k][g] for g in sg
                  for k in ['names','scores','pvals_adj','logfoldchanges']
                  }).to_csv(R('spatial_domain_markers.csv'))
    sc.pl.rank_genes_groups_dotplot(adata, groupby='spatial_domain', n_genes=5)

# ── STEP 22: neuronal / motor analysis ───────────────────────
step(22, "NEURONAL GENE ANALYSIS")
corticospinal_markers = {
    'Cortical Layer': {
        'Upper (II-IV)':    ['CUX1','CUX2','SATB2','BCL11B'],
        'Deep (V-VI)':      ['BCL11B','TBR1','SOX5','FEZF2','CTIP2'],
        'Layer V CST':      ['BCL11B','FEZF2','CRYM','ETV1'],
        'Layer VI':         ['TBR1','FOXP2','CTGF'],
    },
    'Motor Neuron': {
        'General MN':       ['ISL1','MNX1','CHAT','SLC5A7','SLC18A3'],
        'Spinal MN':        ['HOXC4','HOXC5','HOXA5','HOXB5'],
        'Motor Progenitors':['OLIG2','NKX6-1','PAX6','SOX2'],
    },
    'Neuronal Subtype': {
        'Glutamatergic':    ['SLC17A7','SLC17A6','GRIN1','GRIA2'],
        'GABAergic':        ['GAD1','GAD2','SLC32A1','DLX1','DLX2'],
        'Cholinergic':      ['CHAT','ACHE','SLC5A7'],
    },
    'Synaptic': {
        'Presynaptic':      ['SYP','SNAP25','STX1A','VAMP2'],
        'Postsynaptic':     ['DLG4','GRIN1','GRIA2','HOMER1'],
    },
}
all_found = []
for cat, subs in corticospinal_markers.items():
    print(f"\n{cat}:")
    for sc_name, mkrs in subs.items():
        found = [m for m in mkrs if m in adata.var_names]
        all_found.extend(f for f in found if f not in all_found)
        print(f"  {sc_name}: {found or 'None'}")
print(f"Total unique found: {len(all_found)}")

# Higher-res clustering
for res in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res{res}')
    print(f"Res {res}: {len(adata.obs[f'leiden_res{res}'].unique())} clusters")

optimal_res = 2.5
sc.tl.leiden(adata, resolution=optimal_res, key_added='leiden_neuronal')
sc.tl.rank_genes_groups(adata, 'leiden_neuronal', method='wilcoxon', use_raw=False)
result_neuronal = adata.uns['rank_genes_groups']
neuronal_groups = result_neuronal['names'].dtype.names

# Dynamic cluster ID
motor_scores, avail_motor = score_clusters(adata, 'leiden_neuronal',
                                            ['SLC5A7','CHAT','ISL1','MNX1'])
fibro_scores,  avail_fibro = score_clusters(adata, 'leiden_neuronal',
                                             ['COL1A1','COL3A1','DCN','COL1A2','FN1','MGP'])

top_motor_cluster = str(motor_scores.index[0]) if motor_scores is not None else '4'
top_fibro_cluster  = str(fibro_scores.index[0])  if fibro_scores  is not None else '8'
print(f"✓ Motor: cluster {top_motor_cluster}  (score {motor_scores.iloc[0]:.3f})")
print(f"✓ Fibro:  cluster {top_fibro_cluster}  (score {fibro_scores.iloc[0]:.3f})")

if motor_scores is not None:
    motor_scores.reset_index().rename(columns={'index':'cluster',0:'motor_score'}).to_csv(R('motor_neuron_clusters.csv'), index=False)
if fibro_scores is not None:
    fibro_scores.reset_index().rename(columns={'index':'cluster',0:'fibroblast_score'}).to_csv(R('fibroblast_clusters.csv'), index=False)
with open(R('cluster_identities.txt'),'w') as f:
    f.write(f"Motor cluster: {top_motor_cluster}\nFibroblast cluster: {top_fibro_cluster}\nDate: {pd.Timestamp.now()}\n")

# Quantitative spatial analysis
motor_mask = adata.obs['leiden_neuronal'] == top_motor_cluster
fibro_mask  = adata.obs['leiden_neuronal'] == top_fibro_cluster
motor_coords = coords[motor_mask]; fibro_coords = coords[fibro_mask]
print(f"Motor: {motor_mask.sum()} spots | Fibro: {fibro_mask.sum()} spots")
if len(motor_coords) > 3:
    mc = motor_coords.mean(axis=0)
    print(f"Motor center: ({mc[0]:.0f},{mc[1]:.0f}), area: {ConvexHull(motor_coords).volume:.0f} px²")
    print(f"Mean inter-spot dist: {pdist(motor_coords).mean():.0f} px")
if len(motor_coords) > 0 and len(fibro_coords) > 0:
    d = np.linalg.norm(motor_coords.mean(axis=0) - fibro_coords.mean(axis=0))
    print(f"Centre-to-centre dist: {d:.0f} px")
    mm, mM = motor_coords.min(0), motor_coords.max(0)
    fm, fM = fibro_coords.min(0), fibro_coords.max(0)
    overlap = not (mM[0]<fm[0] or mm[0]>fM[0] or mM[1]<fm[1] or mm[1]>fM[1])
    print("✓ OVERLAP" if overlap else "✗ SEPARATE")

fig, axes = plt.subplots(1,3,figsize=(18,5))
for ax, mask_, col, title_ in [
    (axes[0], motor_mask, 'red',   f'Motor Cluster {top_motor_cluster}'),
    (axes[1], fibro_mask,  'blue',  f'Fibro Cluster {top_fibro_cluster}'),
    (axes[2], None,        None,    'Motor vs Fibro'),
]:
    ax.scatter(coords[:,0], coords[:,1], c='lightgray', s=3, alpha=0.3)
    if mask_ is not None:
        ax.scatter(coords[mask_,0], coords[mask_,1], c=col, s=10, alpha=0.8)
    else:
        ax.scatter(coords[motor_mask,0], coords[motor_mask,1], c='red',  s=10, alpha=0.8, label='Motor')
        ax.scatter(coords[fibro_mask,0],  coords[fibro_mask,1],  c='blue', s=10, alpha=0.8, label='Fibro')
        ax.legend()
    ax.set_title(title_); ax.set_aspect('equal')
save_show(fig, R('motor_vs_fibroblast_analysis.png')

# Motor marker stats
motor_genes_rows = [{'gene': g, 'score': result_neuronal['scores'][top_motor_cluster][i],
                     'log2FC': result_neuronal['logfoldchanges'][top_motor_cluster][i],
                     'p_value': result_neuronal['pvals'][top_motor_cluster][i]}
                    for i, g in enumerate(result_neuronal['names'][top_motor_cluster][:50])]
motor_df_full = pd.DataFrame(motor_genes_rows)
motor_df_full.to_csv(R(f'motor_neuron_cluster_{top_motor_cluster}_markers.csv'), index=False)
print(motor_df_full.head(10))

# Layer markers
layer_markers = {
    'Upper Layer':    ['CUX1','CUX2'],
    'Layer V/VI':     ['BCL11B','TBR1','FEZF2'],
    'Motor Neuron':   ['SLC5A7','MNX1','CHAT'],
}
motor_sub = adata[adata.obs['leiden_neuronal'] == top_motor_cluster]
for layer, mkrs in layer_markers.items():
    for mk in mkrs:
        if mk in adata.var_names:
            e = get_exp(motor_sub, mk)
            print(f"  {layer} | {mk}: {(e>0).mean()*100:.1f}% expressing, mean={e.mean():.3f}")

# Subtype (UMN vs LMN)
for label, mkrs in [('Upper MN', ['BCL11B','FEZF2','CRYM','ETV1','SOX5']),
                    ('Lower MN', ['ISL1','MNX1','HOXC4','HOXC5','HOXA5'])]:
    print(f"\n{label}:")
    for mk in mkrs:
        if mk in adata.var_names:
            e = get_exp(adata[adata.obs['leiden_neuronal'] == top_motor_cluster], mk)
            print(f"  {mk}: {e.mean():.3f}")

# Publication figure (motor)
fig = plt.figure(figsize=(16,12))
for sp, mk_gene, title_, cmap_ in [(2,'SLC5A7','B: SLC5A7','Reds'), (5,'GRIA2','E: GRIA2','Blues')]:
    ax = plt.subplot(2,3,sp)
    if mk_gene in adata.var_names:
        e = get_exp(adata, mk_gene)
        sc_ = ax.scatter(coords[:,0], coords[:,1], c=e, cmap=cmap_, s=5, alpha=0.7)
        ax.set_title(title_, fontweight='bold'); plt.colorbar(sc_, ax=ax)

ax1 = plt.subplot(2,3,1)
ax1.scatter(coords[~motor_mask,0], coords[~motor_mask,1], c='lightgray', s=3, alpha=0.3)
ax1.scatter(coords[motor_mask,0], coords[motor_mask,1], c='red', s=10, alpha=0.8, label='Motor')
ax1.set_title('A: Motor Spatial', fontweight='bold'); ax1.set_aspect('equal'); ax1.legend()

ax3 = plt.subplot(2,3,3)
top10 = motor_df_full.head(10)
bars = ax3.barh(range(10), top10['score'].values)
ax3.set_yticks(range(10)); ax3.set_yticklabels(top10['gene'].values); ax3.set_xlabel('Wilcoxon Score')
ax3.set_title('C: Top Motor Markers', fontweight='bold')
for bar, p in zip(bars, top10['p_value'].values):
    bar.set_color('darkred' if p<.001 else 'red' if p<.01 else 'salmon' if p<.05 else 'gray')

ax4 = plt.subplot(2,3,4)
ax4.scatter(coords[motor_mask,0], coords[motor_mask,1], c='red',  s=10, alpha=0.8, label='Motor')
ax4.scatter(coords[fibro_mask,0],  coords[fibro_mask,1],  c='blue', s=10, alpha=0.8, label='Fibro')
ax4.set_title('D: Motor-Fibro Interaction', fontweight='bold'); ax4.set_aspect('equal'); ax4.legend()

ax6 = plt.subplot(2,3,6)
cs = adata.obs['leiden_neuronal'].value_counts()
m_p = cs[top_motor_cluster]/len(adata)*100; f_p = cs[top_fibro_cluster]/len(adata)*100
ax6.pie([m_p, f_p, 100-m_p-f_p], labels=['Motor','Fibro','Other'],
        colors=['red','blue','lightgray'], autopct='%1.1f%%')
ax6.set_title('F: Tissue Composition', fontweight='bold')
save_show(fig, R('motor_neuron_publication_figure.png'), dpi=300

# ── STEP 22b: Neuronal subtype analysis ──────────────────────
subtype_markers = {
    'Sensory Spinal':   {'General': ['NTRK1','NTRK2','NTRK3','RET','RUNX1','RUNX3'],
                         'Mechanoreceptors': ['MAFA','MAFB','NEFH','PVALB'],
                         'Nociceptors': ['TRPV1','TRPA1','SCN9A','TAC1','CALCA']},
    'Cortical Neurons': {'General': ['SATB2','BCL11B','TBR1','CUX1','CUX2','RORB'],
                         'Upper (II-IV)': ['CUX1','CUX2','SATB2','RORB'],
                         'Deep (V-VI)':  ['BCL11B','TBR1','SOX5','FEZF2','CTIP2']},
    'GABAergic':        {'General': ['GAD1','GAD2','SLC32A1','GABRA1'],
                         'PV+': ['PVALB','SST','VIP','NPY']},
    'Glutamatergic':    {'General': ['SLC17A7','SLC17A6','GRIN1','GRIA2'],
                         'AMPA': ['GRIA1','GRIA2','GRIA3','GRIA4']},
}
all_subtype_markers, subtype_scores_d = {}, {}
for cat, subs in subtype_markers.items():
    all_subtype_markers[cat] = list({m for sub in subs.values() for m in sub if m in adata.var_names})
    sc_, _ = score_clusters(adata, 'leiden_neuronal', all_subtype_markers[cat])
    if sc_ is not None: subtype_scores_d[cat] = sc_

if subtype_scores_d:
    pivot_scores = pd.DataFrame(subtype_scores_d).fillna(0)
    pivot_scores.to_csv(R('neuronal_subtype_scores_all.csv'))
    if top_motor_cluster in pivot_scores.index:
        dom = pivot_scores.loc[top_motor_cluster].sort_values(ascending=False)
        print(f"Motor cluster dominant subtype: {dom.index[0]} ({dom.iloc[0]:.4f})")
    # Heatmap
    fig, ax = plt.subplots(figsize=(14,8))
    im = ax.imshow(pivot_scores.T, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(pivot_scores))); ax.set_xticklabels(pivot_scores.index, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot_scores.columns))); ax.set_yticklabels(pivot_scores.columns)
    if top_motor_cluster in pivot_scores.index:
        xi = list(pivot_scores.index).index(top_motor_cluster)
        ax.axvline(xi, color='red', lw=2, ls='--', alpha=0.5)
    plt.colorbar(im, ax=ax, label='Mean Expression')
    ax.set_title('Neuronal Subtype Scores', fontweight='bold')
    save_show(fig, R('neuronal_subtype_heatmap.png')

# ── Skeletal muscle ───────────────────────────────────────────
skeletal_markers = {
    'Myogenic regulators': ['MYOD1','MYF5','MYOG','PAX3','PAX7'],
    'Structural proteins': ['MYH1','MYH2','MYH3','MYH7','ACTA1','DES','TNNT1','TNNT3'],
    'Sarcomeric':           ['TTN','NEB','ACTN2','MYBPC1','MYL1','MYL2'],
    'Muscle-specific':      ['CKMT2','CKM','CHRNG','CHRND','CHRNE'],
}
all_muscle = [m for mkrs in skeletal_markers.values() for m in mkrs if m in adata.var_names]
print(f"Skeletal muscle markers found: {all_muscle}")
if len(all_muscle) >= 2:
    ms_, _ = score_clusters(adata, 'leiden_neuronal', all_muscle)
    top_muscle = str(ms_.index[0])
    print(f"Top muscle cluster: {top_muscle} (score {ms_.iloc[0]:.4f})")
    ms_.reset_index().rename(columns={'index':'cluster',0:'muscle_score'}).to_csv(R('skeletal_muscle_clusters.csv'), index=False)
    fig, ax = plt.subplots(figsize=(8,8))
    mm = adata.obs['leiden_neuronal'] == top_muscle
    ax.scatter(coords[:,0], coords[:,1], c='lightgray', s=3, alpha=0.3)
    ax.scatter(coords[mm,0], coords[mm,1], c='green', s=10, alpha=0.8, label=f'Muscle {top_muscle}')
    ax.set_title(f'Skeletal Muscle Cluster {top_muscle}'); ax.legend()
    save_show(fig, R('skeletal_muscle_spatial.png')

# ── TARDBP / ALS analysis ────────────────────────────────────
step(22, "TARDBP + ALS GENE ANALYSIS")
als_groups = {
    'ALS_Core':   ['TARDBP','FUS','OPTN','SOD1','NEK1','TBK1','CHMP2B','UNC13A'],
    'FTD_Core':   ['MAPT','GRN','TMEM106B'],
    'HOX_Spinal': ['HOXA7','HOXA10','HOXA11'],
    'Signaling':  ['STK10','MAP4K3','EFR3A','EPHA4'],
    'Other':      ['SHOX2','CAMTA1','CDH22','MTX2'],
}
all_als = [g for genes in als_groups.values() for g in genes if g in adata.var_names]
print(f"ALS genes found: {all_als}")

if all_als:
    als_stats = []
    for gene in all_als:
        e = get_exp(adata, gene)
        als_stats.append({'gene': gene, 'mean': e.mean(), 'median': np.median(e),
                          'sem': e.std()/np.sqrt(len(e)), 'min': e.min(), 'max': e.max(),
                          'pct_expressing': (e>0).mean()*100})
    als_df = pd.DataFrame(als_stats); als_df.to_csv(R('ALS_genes_statistics.csv'), index=False)
    print(als_df[['gene','mean','pct_expressing']].round(4).to_string(index=False))

    gl = als_df['gene'].values; means = als_df['mean'].values; errs = als_df['sem'].values
    xp = np.arange(len(gl))
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    axes[0].bar(xp, means, yerr=errs, capsize=5, color='steelblue', alpha=0.7)
    axes[0].set_xticks(xp); axes[0].set_xticklabels(gl, rotation=45, ha='right')
    axes[0].set_title('ALS Gene Expression ± SEM')
    axes[1].bar(xp, als_df['pct_expressing'].values, color='coral', alpha=0.7)
    axes[1].set_xticks(xp); axes[1].set_xticklabels(gl, rotation=45, ha='right')
    axes[1].set_title('% Expressing Cells')
    if len(all_als) >= 2:
        mat = np.column_stack([get_exp(adata, g) for g in all_als])
        corr = np.corrcoef(mat.T)
        im = axes[2].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_xticks(range(len(all_als))); axes[2].set_xticklabels(all_als, rotation=45, ha='right')
        axes[2].set_yticks(range(len(all_als))); axes[2].set_yticklabels(all_als)
        plt.colorbar(im, ax=axes[2])
        axes[2].set_title('ALS Gene Correlation')
    save_show(fig, R('ALS_genes_barplot.png')

    n_cols = min(3, len(all_als)); n_rows = (len(all_als)+n_cols-1)//n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
    axes = np.array(axes).flatten()
    for i, gene in enumerate(all_als):
        e = get_exp(adata, gene); vmax = np.percentile(e[e>0], 95) if (e>0).any() else 1
        sc_ = spatial_scatter(axes[i], coords, e, f'{gene} Expression', cmap='Reds', vmin=0, vmax=vmax)
        plt.colorbar(sc_, ax=axes[i])
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    save_show(fig, R('ALS_genes_spatial.png')

    if 'orig.ident' in adata.obs.columns and 'TARDBP' in adata.var_names:
        e = get_exp(adata, 'TARDBP')
        rows = [{'organoid': org, 'mean': e[adata.obs['orig.ident']==org].mean(),
                 'pct': (e[adata.obs['orig.ident']==org]>0).mean()*100}
                for org in adata.obs['orig.ident'].unique()]
        pd.DataFrame(rows).to_csv(R('TARDBP_by_organoid.csv'), index=False)
        print(pd.DataFrame(rows))

# ── Organoid separation ───────────────────────────────────────
step(23, "SEPARATING + COMPARING ORGANOIDS")
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
adata.obs['organoid'] = [f'Organoid_{i+1}' for i in kmeans.fit_predict(coords)]
organoids = adata.obs['organoid'].unique()
print(adata.obs['organoid'].value_counts())

fig, axes = plt.subplots(1,2,figsize=(14,6))
spatial_scatter(axes[0], coords, adata.obs['leiden'].astype('category').cat.codes, 'Original Clusters')
sc_ = spatial_scatter(axes[1], coords, adata.obs['organoid'].astype('category').cat.codes, 'K-means Organoids', cmap='Set1')
plt.colorbar(sc_, ax=axes[1]); save_show(fig, R('organoid_separation.png')

# Motor abundance by organoid
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
abund = pd.DataFrame([{'organoid': org,
    'total_cells': (adata.obs['organoid']==org).sum(),
    'motor_cells': ((adata.obs['organoid']==org)&(adata.obs['leiden_neuronal']==top_motor_cluster)).sum(),
    'pct_motor': ((adata.obs['organoid']==org)&(adata.obs['leiden_neuronal']==top_motor_cluster)).sum() /
                  (adata.obs['organoid']==org).sum()*100}
    for org in organoids])
abund.to_csv(R('motor_neuron_abundance_by_organoid.csv'), index=False); print(abund)

ctab = pd.DataFrame([[((adata.obs['organoid']==o)&(adata.obs['leiden_neuronal']==top_motor_cluster)).sum(),
                       ((adata.obs['organoid']==o)&(adata.obs['leiden_neuronal']!=top_motor_cluster)).sum()]
                      for o in organoids], index=organoids, columns=['Motor','Other'])
chi2_val, p_chi, *_ = chi2_contingency(ctab)
print(f"Chi-square: χ²={chi2_val:.3f}, p={p_chi:.6f} → {'SIG' if p_chi<.05 else 'NS'}")

if 'SLC5A7' in adata.var_names:
    slc = get_exp(adata, 'SLC5A7')
    grps = [slc[adata.obs['organoid']==o] for o in organoids]
    slc_stats = pd.DataFrame([{'organoid':o,'mean':g.mean(),'sem':g.std()/np.sqrt(len(g)),
                                'pct_expressing':(g>0).mean()*100} for o,g in zip(organoids,grps)])
    slc_stats.to_csv(R('SLC5A7_expression_by_organoid.csv'), index=False); print(slc_stats)
    F, p_anova = f_oneway(*grps)
    print(f"ANOVA: F={F:.3f}, p={p_anova:.6f}")
    pvs = [ttest_ind(grps[i],grps[j])[1] for i in range(len(organoids)) for j in range(i+1,len(organoids))]
    comps = [f"{organoids[i]} vs {organoids[j]}" for i in range(len(organoids)) for j in range(i+1,len(organoids))]
    _, pc, *_ = multipletests(pvs, method='bonferroni')
    for c, pr, pc_ in zip(comps, pvs, pc):
        print(f"  {c}: p={pr:.4f} adj={pc_:.4f} {'***' if pc_<.001 else '**' if pc_<.01 else '*' if pc_<.05 else 'ns'}")

org_colors = ['red','blue','green','purple']
fig, axes = plt.subplots(2,2,figsize=(12,12)); axes = axes.flatten()
for i, org in enumerate(organoids[:4]):
    om = adata.obs['organoid']==org; mm = om & (adata.obs['leiden_neuronal']==top_motor_cluster)
    axes[i].scatter(coords[om,0], coords[om,1], c='lightgray', s=3, alpha=0.3)
    axes[i].scatter(coords[mm,0], coords[mm,1], c=org_colors[i], s=10, alpha=0.8, label='Motor')
    axes[i].set_title(org); axes[i].set_aspect('equal'); axes[i].legend()
save_show(fig, R('motor_neurons_by_organoid.png')

if 'SLC5A7' in adata.var_names:
    fig, ax = plt.subplots(figsize=(10,6))
    bp = ax.boxplot(grps, patch_artist=True, labels=organoids)
    for bx, col in zip(bp['boxes'], org_colors): bx.set_facecolor(col)
    ax.set_ylabel('SLC5A7 Expression'); ax.set_title('Motor Marker by Organoid')
    if p_anova < .05:
        ax.text(0.5, 0.95, f'ANOVA: p={p_anova:.4f}', transform=ax.transAxes,
                ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    save_show(fig, R('slc5a7_by_organoid_boxplot.png')

# ── Cell type assignment + custom UMAPs ──────────────────────
step(24, "CUSTOM COLORED UMAPS PER ORGANOID")
ct_marker_map = {
    'Motor Neurons':    ['SLC5A7','BCL11B','CHAT'],
    'Cortical Neurons': ['SNAP25','RBFOX3','MAP2'],
    'Glutamatergic':    ['GRIA2','SLC17A7','GRIN1'],
    'GABAergic':        ['SLC32A1','GAD1','GAD2'],
    'Neural Progenitors':['SOX2','PAX6','NES'],
    'Fibroblasts':      ['COL1A1','COL3A1','DCN'],
    'Glial':            ['GFAP','S100B','OLIG2'],
}
ct_colors = {'Motor Neurons':'#FF0000','Cortical Neurons':'#FFA500','Glutamatergic':'#FFFF00',
             'GABAergic':'#00FF00','Neural Progenitors':'#ADD8E6','Fibroblasts':'#FFC0CB',
             'Glial':'#808080','Other':'#FFFFFF'}

cluster_to_ct = {}
for cl in neuronal_groups:
    m = adata.obs['leiden_neuronal'] == cl
    sc_ = {ct: np.mean([get_exp(adata, mk)[m].mean() for mk in mks if mk in adata.var_names])
           for ct, mks in ct_marker_map.items()
           if any(mk in adata.var_names for mk in mks)}
    best = max(sc_, key=sc_.get) if sc_ else 'Other'
    cluster_to_ct[cl] = best if sc_.get(best, 0) > 0.01 else 'Other'
cluster_to_ct[top_motor_cluster] = 'Motor Neurons'
cluster_to_ct[top_fibro_cluster]  = 'Fibroblasts'
adata.obs['cell_type'] = adata.obs['leiden_neuronal'].map(cluster_to_ct).fillna('Other')
unique_cts = adata.obs['cell_type'].unique()
ct_color_map = {ct: ct_colors.get(ct, f'#{np.random.randint(0,0xFFFFFF):06x}') for ct in unique_cts}

for org in organoids:
    om = adata.obs['organoid'] == org; sub = adata[om].copy()
    if sub.shape[0] < 10: continue
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: leiden_neuronal clusters
    sc.pl.umap(sub, color='leiden_neuronal', ax=axes[0], show=False,
               title=f'{org} – Clusters', legend_loc='on data', legend_fontsize=8)

    # Right: cell types (custom colours)
    for ct in sub.obs['cell_type'].unique():
        m = sub.obs['cell_type'] == ct
        axes[1].scatter(sub.obsm['X_umap'][m,0], sub.obsm['X_umap'][m,1],
                        c=ct_color_map[ct], s=10, alpha=0.8, label=ct, edgecolors='none')
    axes[1].set_title(f'{org} – Cell Types', fontsize=14, fontweight='bold')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    save_show(fig, R(f'umap_{org}_colored.png'), dpi=300

# ── Side-by-side: CLUSTERS ──
fig, axes = plt.subplots(2, 2, figsize=(20, 18)); axes = axes.flatten()
for i, org in enumerate(organoids[:4]):
    om = adata.obs['organoid'] == org; sub = adata[om].copy()
    if sub.shape[0] < 5:
        axes[i].text(0.5, 0.5, f'{org}\n(insufficient)', ha='center', va='center'); continue
    sc.pl.umap(sub, color='leiden_neuronal', ax=axes[i], show=False,
               title=org, legend_loc='on data', legend_fontsize=7)
plt.suptitle('Clusters Across Organoids', fontsize=18, fontweight='bold', y=1.01)
save_show(fig, R('umap_all_organoids_clusters.png'), dpi=300

# ── Side-by-side: CELL TYPES ──
fig, axes = plt.subplots(2, 2, figsize=(20, 18)); axes = axes.flatten()
for i, org in enumerate(organoids[:4]):
    om = adata.obs['organoid'] == org; sub = adata[om].copy()
    if sub.shape[0] < 5:
        axes[i].text(0.5, 0.5, f'{org}\n(insufficient)', ha='center', va='center'); continue
    for ct in sub.obs['cell_type'].unique():
        m = sub.obs['cell_type'] == ct
        axes[i].scatter(sub.obsm['X_umap'][m,0], sub.obsm['X_umap'][m,1],
                        c=ct_color_map[ct], s=8, alpha=0.8, edgecolors='none')
    axes[i].set_title(org, fontsize=14, fontweight='bold')
    axes[i].set_xticks([]); axes[i].set_yticks([])
handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=ct_color_map[ct],
                      markersize=10, label=ct) for ct in unique_cts]
fig.legend(handles=handles, bbox_to_anchor=(0.5, 0.0), loc='lower center',
           ncol=min(4, len(unique_cts)), frameon=True)
plt.suptitle('Cell Types Across Organoids', fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
save_show(fig, R('umap_all_organoids_comparison.png'), dpi=300

# Per-gene comparison
key_genes = [g for g in ['SLC5A7','BCL11B','SNAP25','GRIA2','SLC32A1','COL1A1','COL3A1',
                          'DCN','RBFOX3','CSF1R','TREM2'] if g in adata.var_names]
for gene in key_genes:
    fig, axes = plt.subplots(2,2,figsize=(12,10)); axes = axes.flatten()
    for i, org in enumerate(organoids[:4]):
        sub = adata[adata.obs['organoid']==org].copy()
        try:
            sc.pl.umap(sub, color=gene, ax=axes[i], title=f'{org} - {gene}',
                       show=False, cmap='Reds', vmax='p95', use_raw=False)
        except:
            axes[i].text(0.5,0.5,f'{gene}\nerror',ha='center',va='center')
    plt.suptitle(f'{gene} Across Organoids', fontsize=16, fontweight='bold')
    plt.tight_layout(); save_show(fig, R(f'umap_comparison_{gene}.png'), dpi=300

# ── Organoid similarity ───────────────────────────────────────
step(25, "COMPREHENSIVE ORGANOID SIMILARITY ANALYSIS")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx

# Cell type composition
comp_rows = [{'organoid': org,
              **{ct: pct for ct, pct in (adata.obs[adata.obs['organoid']==org]['cell_type'].value_counts(normalize=True)*100).items()}}
             for org in organoids]
comp_df = pd.DataFrame(comp_rows).fillna(0); comp_df.to_csv(R('organoid_cell_type_composition.csv'), index=False)

# Marker expression profiles
expr_rows = [{'organoid': org,
              **{f"{ct}_{mk}": get_exp(adata[adata.obs['organoid']==org], mk).mean()
                 for ct, mks in ct_marker_map.items() for mk in mks if mk in adata.var_names}}
             for org in organoids]
marker_df = pd.DataFrame(expr_rows).fillna(0); marker_df.to_csv(R('organoid_marker_expression.csv'), index=False)

# Global correlations
profiles = [np.asarray(adata[adata.obs['organoid']==org].X.mean(axis=0)).flatten() for org in organoids]
corr_df = pd.DataFrame(np.corrcoef(profiles), index=organoids, columns=organoids)
print("Global expression correlation:\n", corr_df.round(3))

# Multi-dimensional similarity
all_feats = pd.concat([comp_df.drop('organoid',axis=1), marker_df.drop('organoid',axis=1)], axis=1)
fs = StandardScaler().fit_transform(all_feats)
ed = euclidean_distances(fs); md = manhattan_distances(fs); cs = cosine_similarity(fs)
edf = pd.DataFrame(ed, index=organoids, columns=organoids)
print("Euclidean dist:\n", edf.round(3))

lm = linkage(fs, method='ward')
fig, ax = plt.subplots(figsize=(10,5))
dendrogram(lm, labels=list(organoids), ax=ax)
ax.set_title('Organoid Hierarchical Clustering'); save_show(fig, R('organoid_comprehensive_clustering.png')

pca = PCA(n_components=2); pr = pca.fit_transform(fs)
fig, ax = plt.subplots(figsize=(8,7))
ax.scatter(pr[:,0], pr[:,1], c=range(len(organoids)), s=200, cmap='Set1', alpha=0.7)
for i, org in enumerate(organoids): ax.annotate(org, pr[i], fontsize=12, fontweight='bold', ha='center')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA of Organoids'); ax.grid(alpha=0.3)
save_show(fig, R('organoid_comprehensive_pca.png')

# Combined similarity + ranking
norm_dist = (ed/ed.max() + md/md.max() + (1-cs)/1) / 3
np.fill_diagonal(norm_dist, np.inf)
pairs_df = pd.DataFrame([{'pair': f"{organoids[i]} - {organoids[j]}",
                           'distance': norm_dist[i,j]}
                          for i in range(len(organoids)) for j in range(i+1,len(organoids))]
                        ).sort_values('distance')
print("\nSimilarity ranking:\n", pairs_df.to_string(index=False))
print(f"\n✅ MOST SIMILAR: {pairs_df.iloc[0]['pair']}")
print(f"📉 LEAST SIMILAR: {pairs_df.iloc[-1]['pair']}")

# ── STEP 26: SVG detection (multi-method) ─────────────────────
step(26, "SPATIALLY VARIABLE GENE DETECTION (Multi-method)")
try:
    import squidpy as sq
    if 'spatial_neighbors' not in adata.uns:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)
    n_test = min(2000, adata.shape[1])
    sq.gr.spatial_autocorr(adata, mode='moran', genes=adata.var_names[:n_test], n_perms=100)
    mr = adata.uns['moranI']
    if 'gene' not in mr.columns: mr = mr.copy(); mr['gene'] = mr.index
    gene_col = 'gene' if 'gene' in mr.columns else 'genes'
    print(mr.nsmallest(20,'pval_norm')[['I','pval_norm',gene_col]])
    mr.to_csv(R('spatially_variable_genes_moran.csv'))
except Exception as e: print(f"Moran error: {e}")

# SPARK-X (memory-efficient)
try:
    from scipy.stats import chi2
    def sparkx_test(expr, coords, n_sample=5000):
        n = len(expr)
        if n > n_sample:
            idx = np.random.choice(n, n_sample, replace=False)
            expr, coords = expr[idx], coords[idx]
        cd = cdist(coords, coords, 'euclidean')
        K = np.exp(-cd**2 / (2 * np.median(cd[cd>0])**2))
        H = np.eye(len(expr)) - 1/len(expr)
        Kc = H @ K @ H; ec = expr - expr.mean()
        stat = ec @ Kc @ ec / (ec @ ec)
        df = np.trace(Kc @ Kc) / np.trace(Kc)
        return stat, 1 - chi2.cdf(stat*df, df)

    n_spark = min(500, adata.shape[1]); sparkx_res = []
    for i in range(n_spark):
        e = get_exp(adata, adata.var_names[i])
        if e.max() > 0:
            stat, p = sparkx_test(e, coords)
            sparkx_res.append({'gene': adata.var_names[i], 'statistic': stat, 'p_value': p})
    if sparkx_res:
        sx = pd.DataFrame(sparkx_res)
        sx['q_value'] = multipletests(sx['p_value'], method='fdr_bh')[1]
        sx.to_csv(R('spatially_variable_genes_sparkx.csv'), index=False)
        print("Top SPARK-X genes:\n", sx.nsmallest(10,'p_value')[['gene','statistic','p_value']])
except Exception as e: print(f"SPARK-X error: {e}")

# ── STEP 27: Spatial domain detection ─────────────────────────
step(27, "SPATIAL DOMAIN DETECTION")
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=10, metric='euclidean').fit(coords)
distances_, indices_ = nn.kneighbors(coords)
G_sp = nx.Graph()
for i in range(adata.shape[0]):
    for j, d in zip(indices_[i,1:], distances_[i,1:]):
        G_sp.add_edge(i, j, weight=1/(d+1e-6))

louvain_done = False
for import_str, call_fn in [
    ("from community import best_partition", "best_partition"),
    ("from community import community_louvain; best_partition=community_louvain.best_partition", "best_partition"),
    ("import community as cm; best_partition=cm.best_partition", "best_partition"),
]:
    try:
        exec(import_str)
        partition = eval(f"{call_fn}(G_sp)")
        adata.obs['spatial_domain_louvain'] = [str(partition[i]) for i in range(adata.shape[0])]
        louvain_done = True; break
    except: pass
if not louvain_done:
    adata.obs['spatial_domain_louvain'] = [f"Domain_{x}" for x in KMeans(n_clusters=8, random_state=42).fit_predict(coords)]
sc.pp.neighbors(adata, n_neighbors=10, use_rep='spatial')
sc.tl.leiden(adata, resolution=0.5, key_added='spatial_domain_leiden')
print(f"Louvain domains: {len(adata.obs['spatial_domain_louvain'].unique())}  "
      f"Leiden domains: {len(adata.obs['spatial_domain_leiden'].unique())}")

# Always plot the 3-panel comparison (does not require squidpy)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, col, title_ in zip(axes,
    ['spatial_domain_louvain', 'spatial_domain_leiden', 'leiden'],
    ['Spatial Domains (Louvain)', 'Spatial Domains (Leiden)', 'Transcriptomic Clusters']):
    try:
        sc.pl.spatial(adata, color=col, ax=ax, show=False, title=title_, spot_size=50)
    except Exception:
        # fallback: plain scatter if sc.pl.spatial not available
        spatial_scatter(ax, coords, adata.obs[col].astype('category').cat.codes, title_)
save_show(fig, R('spatial_domains_comparison.png')

sc.tl.rank_genes_groups(adata, 'spatial_domain_leiden', method='wilcoxon')
sc.pl.rank_genes_groups_dotplot(adata, groupby='spatial_domain_leiden', n_genes=5)

# ── STEP 28: Disease vs Healthy ───────────────────────────────
step(28, "DISEASE VS HEALTHY COMPARISON")
disease_map = {'Organoid_1':'Control','Organoid_2':'Control','Organoid_3':'Disease','Organoid_4':'Disease'}
adata.obs['condition'] = adata.obs['organoid'].map(disease_map).fillna('Unknown')
print(adata.obs['condition'].value_counts())

# Pseudobulk DE
pb_means = {org: np.asarray(adata[adata.obs['organoid']==org].X.mean(axis=0)).flatten()
            for org in organoids}
pb_conditions = {org: adata.obs.loc[adata.obs['organoid']==org,'condition'].iloc[0] for org in organoids}
ctrl_idx = [i for i,o in enumerate(organoids) if pb_conditions[o]=='Control']
dis_idx  = [i for i,o in enumerate(organoids) if pb_conditions[o]=='Disease']
pb_arr = np.array([pb_means[o] for o in organoids])

if len(ctrl_idx) >= 2 and len(dis_idx) >= 2:
    de_res = []
    for i, gene in enumerate(adata.var_names):
        t, p = ttest_ind(pb_arr[ctrl_idx, i], pb_arr[dis_idx, i])
        lfc = np.log2(pb_arr[dis_idx,i].mean()+1) - np.log2(pb_arr[ctrl_idx,i].mean()+1)
        de_res.append({'gene': gene, 'log2FC': lfc, 'p_value': p, 't_stat': t})
    de_df = pd.DataFrame(de_res); de_df['p_adj'] = multipletests(de_df['p_value'], method='fdr_bh')[1]
    de_df = de_df.sort_values('p_adj'); de_df.to_csv(R('disease_vs_healthy_pseudobulk.csv'), index=False)
    print(de_df.head(20)[['gene','log2FC','p_adj']])

    de_df['-log10p'] = -np.log10(de_df['p_adj'])
    sig = (de_df['p_adj']<.05) & (de_df['log2FC'].abs()>.5)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(de_df['log2FC'], de_df['-log10p'],
               c=['gray' if not s else 'red' if f>0 else 'blue'
                  for s,f in zip(sig, de_df['log2FC'])], alpha=0.6, s=10)
    ax.axhline(-np.log10(.05), color='gray', ls='--', alpha=.5)
    ax.axvline(0.5, color='gray', ls='--', alpha=.5); ax.axvline(-0.5, color='gray', ls='--', alpha=.5)
    ax.set(xlabel='log2FC', ylabel='-log10(adj p)'); ax.set_title('Disease vs Healthy Volcano')
    for _, row in de_df.nsmallest(10,'p_adj').iterrows():
        ax.annotate(row['gene'], (row['log2FC'], row['-log10p']), fontsize=8)
    save_show(fig, R('disease_vs_healthy_volcano.png')

if len(adata.obs['condition'].unique()) >= 2:
    sc.tl.rank_genes_groups(adata, 'condition', groups=['Disease'], reference='Control', method='wilcoxon')
    sc.pl.rank_genes_groups_dotplot(adata, groupby='condition', n_genes=10)
    rs = adata.uns['rank_genes_groups']
    pd.DataFrame({'gene': rs['names']['Disease'], 'log2FC': rs['logfoldchanges']['Disease'],
                  'p_adj': rs['pvals_adj']['Disease']}).to_csv(R('disease_vs_healthy_scranpy.csv'), index=False)

# ── STEP 29: Pathway enrichment (full) ────────────────────────
step(29, "PATHWAY ENRICHMENT")
if GSEAPY_AVAILABLE:
    gene_sets = ['GO_Biological_Process_2023','KEGG_2021_Human','Reactome_2022']
    if 'de_df' in dir():
        for name, subset in [('Upregulated', de_df[(de_df['log2FC']>.5)&(de_df['p_adj']<.05)]['gene'].tolist()),
                              ('Downregulated', de_df[(de_df['log2FC']<-.5)&(de_df['p_adj']<.05)]['gene'].tolist())]:
            if len(subset) >= 5:
                try:
                    enr = gp.enrichr(subset, gene_sets=gene_sets, organism='human', outdir=R(f'enrichment_{name}'))
                    print(f"{name}:\n{enr.results.head(5)[['Term','Adjusted P-value']]}")
                    enr.results.to_csv(R(f'enrichment_{name}.csv'), index=False)
                except Exception as e: print(f"{name} enrichment error: {e}")
    for cl in neuronal_groups[:5]:
        try:
            enr = gp.enrichr(result_neuronal['names'][cl][:100].tolist(), gene_sets=gene_sets,
                             organism='human', outdir=R(f'enrichment_cluster_{cl}_detailed'))
            enr.results.to_csv(R(f'enrichment_cluster_{cl}_detailed.csv'), index=False)
        except Exception as e: print(f"Cluster {cl}: {e}")

# ── STEP 30: Cell-cell interactions ──────────────────────────
step(30, "CELL-CELL INTERACTION ANALYSIS")
try:
    import squidpy as sq
    if 'spatial_neighbors' not in adata.uns:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)
    sq.gr.nhood_enrichment(adata, cluster_key='cell_type')
    fig, ax = plt.subplots(figsize=(10,8))
    sq.pl.nhood_enrichment(adata, cluster_key='cell_type', ax=ax)
    save_show(fig, R('neighborhood_enrichment.png')

    lr_pairs = [('NGF','NGFR'),('BDNF','NTRK2'),('FGF2','FGFR1'),('NLGN1','NRXN1'),('CNTF','CNTFR')]
    interactions = []
    for L, R in lr_pairs:
        if L not in adata.var_names or R not in adata.var_names: continue
        le = get_exp(adata, L); re = get_exp(adata, R)
        for i, ct1 in enumerate(adata.obs['cell_type'].unique()):
            for j, ct2 in enumerate(adata.obs['cell_type'].unique()):
                if i == j: continue
                m1 = adata.obs['cell_type']==ct1; m2 = adata.obs['cell_type']==ct2
                score = le[m1].mean() * re[m2].mean()
                if score > 0.01:
                    interactions.append({'L': L, 'R': R, 'sender': ct1, 'receiver': ct2, 'score': score})
    if interactions:
        idf = pd.DataFrame(interactions).sort_values('score', ascending=False)
        idf.to_csv(R('cell_cell_interactions.csv'), index=False)
        print(idf.head(10))
        fig, ax = plt.subplots(figsize=(12,8))
        top15 = idf.head(15)
        ax.barh(range(len(top15)), top15['score'])
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels([f"{r['sender']}→{r['receiver']}: {r['L']}-{r['R']}" for _,r in top15.iterrows()])
        ax.set_title('Top Cell-Cell Interactions'); save_show(fig, R('cell_cell_interactions.png')

    # Motor-Fibro proximity
    from scipy.spatial import cKDTree
    mt = cKDTree(coords[motor_mask]); ft = cKDTree(coords[fibro_mask])
    nearby = mt.query_ball_tree(ft, 50)
    int_m = np.zeros(adata.shape[0], bool); int_f = np.zeros(adata.shape[0], bool)
    for i, nb in enumerate(nearby):
        if nb:
            int_m[np.where(motor_mask)[0][i]] = True
            for j in nb: int_f[np.where(fibro_mask)[0][j]] = True
    print(f"Motor near fibro: {int_m.sum()} | Fibro near motor: {int_f.sum()}")
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    axes[0].scatter(coords[:,0], coords[:,1], c='lightgray', s=3, alpha=0.3)
    axes[0].scatter(coords[motor_mask,0], coords[motor_mask,1], c='red',  s=5, alpha=0.5, label='Motor')
    axes[0].scatter(coords[fibro_mask,0],  coords[fibro_mask,1],  c='blue', s=5, alpha=0.5, label='Fibro')
    axes[0].set_title('All Motor + Fibro'); axes[0].legend()
    axes[1].scatter(coords[:,0], coords[:,1], c='lightgray', s=3, alpha=0.3)
    axes[1].scatter(coords[int_m,0], coords[int_m,1], c='red',  s=10, alpha=0.8, label='Interacting Motor')
    axes[1].scatter(coords[int_f,0], coords[int_f,1], c='blue', s=10, alpha=0.8, label='Interacting Fibro')
    axes[1].set_title('Motor-Fibro Interactions (<50px)'); axes[1].legend()
    save_show(fig, R('motor_fibroblast_interactions.png')
except Exception as e: print(f"CCI error: {e}")

# ── STEP 31: Spatial neighborhood analysis ────────────────────
step(31, "SPATIAL NEIGHBORHOOD ANALYSIS")
try:
    import squidpy as sq
    sq.gr.nhood_enrichment(adata, cluster_key='leiden_neuronal')
    fig, ax = plt.subplots(figsize=(10,8))
    sq.pl.nhood_enrichment(adata, cluster_key='leiden_neuronal', ax=ax)
    save_show(fig, R('neighborhood_enrichment_heatmap.png')
    sq.gr.co_occurrence(adata, cluster_key='leiden_neuronal')
    if top_motor_cluster and top_fibro_cluster:
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        sq.pl.co_occurrence(adata, cluster_key='leiden_neuronal', clusters=[top_motor_cluster], ax=axes[0])
        sq.pl.co_occurrence(adata, cluster_key='leiden_neuronal', clusters=[top_fibro_cluster],  ax=axes[1])
        save_show(fig, R('co_occurrence_patterns.png')
    # Spatial niches
    nb_mat = adata.obsp['spatial_connectivities']
    n_cl = len(adata.obs['leiden_neuronal'].unique())
    nbcomp = np.zeros((adata.shape[0], n_cl))
    for i in range(adata.shape[0]):
        ni = nb_mat[i].nonzero()[1]
        if len(ni):
            for j, cl in enumerate(adata.obs['leiden_neuronal'].cat.categories):
                nbcomp[i,j] = (adata.obs['leiden_neuronal'].iloc[ni]==cl).sum()/len(ni)
    adata.obs['spatial_niche'] = [f'Niche_{n}' for n in KMeans(n_clusters=5, random_state=42).fit_predict(nbcomp)]
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    sc.pl.spatial(adata, color='spatial_niche',      ax=axes[0], show=False, spot_size=50)
    sc.pl.spatial(adata, color='leiden_neuronal',    ax=axes[1], show=False, spot_size=50)
    save_show(fig, R('spatial_niches.png')
    nc = pd.crosstab(adata.obs['spatial_niche'], adata.obs['leiden_neuronal'])
    (nc.div(nc.sum(axis=1), axis=0)*100).round(1).to_csv(R('niche_composition.csv'))
    sc.tl.rank_genes_groups(adata, 'spatial_niche', method='wilcoxon')
    sc.pl.rank_genes_groups_dotplot(adata, groupby='spatial_niche', n_genes=5)
except Exception as e: print(f"Neighborhood error: {e}")

# ── STEP 32: Batch correction ────────────────────────────────
step(32, "BATCH CORRECTION")
fig, axes = plt.subplots(1,2,figsize=(14,6))
sc.pl.umap(adata, color='organoid',        ax=axes[0], show=False, title='Before: by Organoid')
sc.pl.umap(adata, color='leiden_neuronal', ax=axes[1], show=False, title='Before: by Cluster')
save_show(fig, R('batch_effects_before.png')

for method, pkg, install in [('Harmony','harmonypy','pip install harmonypy'),
                               ('BBKNN','bbknn','pip install bbknn')]:
    try:
        if method == 'Harmony':
            import harmonypy
            meta = adata.obs[['organoid']].copy()
            meta['organoid_code'] = meta['organoid'].astype('category').cat.codes
            ho = harmonypy.HarmonyIntegration(X_pca=adata.obsm['X_pca'], meta_data=meta, vars_use=['organoid_code'])
            ho.run(max_iter_harmony=10)
            adata.obsm['X_pca_harmony'] = ho.Z_corr.T
            sc.pp.neighbors(adata, use_rep='X_pca_harmony'); sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=0.5, key_added='leiden_harmony')
            key_added = 'leiden_harmony'
        else:
            import bbknn
            bbknn.bbknn(adata, batch_key='organoid'); sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=0.5, key_added='leiden_bbknn')
            key_added = 'leiden_bbknn'
        fig, axes = plt.subplots(1,2,figsize=(14,6))
        sc.pl.umap(adata, color='organoid',  ax=axes[0], show=False, title=f'{method}: by Organoid')
        sc.pl.umap(adata, color=key_added,   ax=axes[1], show=False, title=f'{method}: by Cluster')
        save_show(fig, R(f'batch_effects_after_{method.lower()}.png')
    except ImportError: print(f"{method} not installed: {install}")
    except Exception as e: print(f"{method} error: {e}")

# ── STEP 33: Spot deconvolution ───────────────────────────────
step(33, "SPOT DECONVOLUTION")
decon_markers = {
    'Motor_Neurons':     ['SLC5A7','BCL11B','CHAT','ISL1'],
    'Cortical_Neurons':  ['SNAP25','RBFOX3','MAP2'],
    'GABAergic':         ['SLC32A1','GAD1','GAD2'],
    'Glutamatergic':     ['GRIA2','SLC17A7'],
    'Fibroblasts':       ['COL1A1','COL3A1','DCN'],
    'Microglia':         ['CSF1R','TREM2'],
    'Astrocytes':        ['GFAP','S100B'],
    'Neural_Progenitors':['SOX2','PAX6','NES'],
}
scores_mat = {}
for ct, mks in decon_markers.items():
    avail = [m for m in mks if m in adata.var_names]
    if avail:
        scores_mat[ct] = np.mean(np.column_stack([get_exp(adata, m) for m in avail]), axis=1)
    else:
        scores_mat[ct] = np.zeros(adata.shape[0])
prop_df = pd.DataFrame(scores_mat, index=adata.obs_names)
prop_df = prop_df.div(prop_df.sum(axis=1).replace(0, 1), axis=0)
prop_df.to_csv(R('spot_cell_type_proportions.csv'))
key_ct = [ct for ct in ['Motor_Neurons','Fibroblasts','Cortical_Neurons'] if ct in prop_df.columns]
if key_ct and 'spatial' in adata.obsm:
    fig, axes = plt.subplots(1,len(key_ct),figsize=(5*len(key_ct),5))
    if len(key_ct)==1: axes=[axes]
    for ax, ct in zip(axes, key_ct):
        sc_ = spatial_scatter(ax, coords, prop_df[ct].values, f'{ct} Proportion', cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(sc_, ax=ax)
    save_show(fig, R('spot_deconvolution.png')

# ── STEP 34: Spatial trajectories ────────────────────────────
step(34, "SPATIAL TRAJECTORY ANALYSIS")
sc.tl.diffmap(adata, n_comps=10)
sc.pl.diffmap(adata, color='leiden_neuronal')
for i in range(3):
    cx = np.corrcoef(adata.obsm['X_diffmap'][:,i], coords[:,0])[0,1]
    cy = np.corrcoef(adata.obsm['X_diffmap'][:,i], coords[:,1])[0,1]
    print(f"DC{i+1}: corr_x={cx:.3f}, corr_y={cy:.3f}")

if 'distance_from_center' not in adata.obs.columns:
    adata.obs['distance_from_center'] = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
dmin, dmax = adata.obs['distance_from_center'].min(), adata.obs['distance_from_center'].max()
adata.obs['radial_pseudotime'] = (adata.obs['distance_from_center'] - dmin) / (dmax - dmin)

fig, axes = plt.subplots(1,3,figsize=(18,6))
try:
    sc.pl.spatial(adata, color='radial_pseudotime', ax=axes[0], show=False, cmap='viridis', spot_size=50)
except: spatial_scatter(axes[0], coords, adata.obs['radial_pseudotime'].values, 'Radial Pseudotime', cmap='viridis')
sc.pl.umap(adata, color='radial_pseudotime', ax=axes[1], show=False, cmap='viridis')
for cl in list(adata.obs['leiden_neuronal'].unique())[:5]:
    axes[2].hist(adata.obs.loc[adata.obs['leiden_neuronal']==cl,'radial_pseudotime'], bins=30, alpha=0.5, label=cl)
axes[2].set(xlabel='Pseudotime', ylabel='Count'); axes[2].legend()
save_show(fig, R('radial_pseudotime.png')

pt_corr = []
for i in range(min(500, adata.shape[1])):
    e = get_exp(adata, adata.var_names[i])
    if e.max() > 0:
        c = np.corrcoef(e, adata.obs['radial_pseudotime'].values)[0,1]
        pt_corr.append({'gene': adata.var_names[i], 'correlation': c, 'abs_corr': abs(c)})
pt_df = pd.DataFrame(pt_corr).sort_values('abs_corr', ascending=False)
pt_df.to_csv(R('pseudotime_correlated_genes.csv'), index=False)
print("Top pseudotime-correlated genes:\n", pt_df.head(10))

# ── STEP 35: Statistical design with replicates ───────────────
step(35, "STATISTICAL DESIGN WITH REPLICATES")
pb_rows = []
for org in organoids:
    m = adata.obs['organoid'] == org
    cond = adata.obs.loc[m,'condition'].iloc[0] if 'condition' in adata.obs.columns else 'Unknown'
    ex = np.asarray(adata[m].X.mean(axis=0)).flatten()
    pb_rows.append({'organoid':org,'condition':cond,**dict(zip(adata.var_names, ex))})
pb_df = pd.DataFrame(pb_rows); pb_df.to_csv(R('pseudobulk_expression.csv'), index=False)
print(f"Pseudobulk: {len(pb_df)} organoids | Conditions: {pb_df['condition'].unique()}")

if top_motor_cluster and 'condition' in adata.obs.columns:
    mc_rows = []
    for org in organoids:
        om = adata.obs['organoid']==org; mm = om & (adata.obs['leiden_neuronal']==top_motor_cluster)
        cond = adata.obs.loc[om,'condition'].iloc[0]
        mc_rows.append({'organoid':org,'condition':cond,'total':om.sum(),
                        'motor':mm.sum(),'pct_motor':mm.sum()/om.sum()*100})
    mc_df = pd.DataFrame(mc_rows); mc_df.to_csv(R('motor_neuron_counts_by_organoid.csv'), index=False)
    print(mc_df)
    if len(mc_df['condition'].unique()) == 2:
        c1, c2 = mc_df['condition'].unique()
        g1 = mc_df[mc_df['condition']==c1]['pct_motor'].values
        g2 = mc_df[mc_df['condition']==c2]['pct_motor'].values
        if len(g1)>=2 and len(g2)>=2:
            t, p = ttest_ind(g1, g2)
            print(f"{c1}: {g1.mean():.2f}±{g1.std():.2f}  {c2}: {g2.mean():.2f}±{g2.std():.2f}")
            print(f"t={t:.3f}, p={p:.4f} → {'SIG' if p<.05 else 'NS'}")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "="*60)
print(f"Dataset: {adata.shape[0]} spots × {adata.shape[1]} genes")
print(f"Clusters: {len(groups)} (initial) | Motor: {top_motor_cluster} | Fibro: {top_fibro_cluster}")
print(f"Spatial coords: {'✓' if 'spatial' in adata.obsm else '✗'}")
print(f"Gene symbols: {'✓' if 'real_gene_name' in adata.var.columns else '✗'}")
print("="*60 + "\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
