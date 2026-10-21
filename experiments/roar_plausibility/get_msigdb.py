"""Extract Reactome and KEGG gene sets (gene symbols) from MSigDB 7.5.1 as bundled in the conda-forge package r-msigdbr 7.5.1
(R lazy-load database, LZMA-compressed objects). Writes REACTOME_msigdb751.gmt and KEGG_msigdb751.gmt. Needs pyreadr."""
import os, io, lzma, struct, tarfile, urllib.request, tempfile, pyreadr
URL = "https://conda.anaconda.org/conda-forge/noarch/r-msigdbr-7.5.1-r42hc72bb7e_1.tar.bz2"
tmp = tempfile.mkdtemp()
tb = os.path.join(tmp, "m.tar.bz2"); urllib.request.urlretrieve(URL, tb)
with tarfile.open(tb) as t:
    b = t.extractfile("lib/R/library/msigdbr/R/sysdata.rdb").read()
objs = []; off = 0
while off < len(b):
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
    d = dec.decompress(b[off + 5:]); used = len(b) - off - 5 - len(dec.unused_data)
    p = os.path.join(tmp, f"o{len(objs)}.rds"); open(p, "wb").write(d); objs.append(pyreadr.read_r(p)[None]); off += 5 + used
genes, gsg, gs = objs  # msigdbr_genes, msigdbr_geneset_genes, msigdbr_genesets
for sub in ["CP:REACTOME", "CP:KEGG"]:
    s = gs[gs.gs_subcat == sub]; m = gsg[gsg.gs_id.isin(s.gs_id)].merge(genes, on="gene_id"); name = s.set_index("gs_id")["gs_name"]
    with open(sub.split(":")[1] + "_msigdb751.gmt", "w") as f:
        for gid, grp in m.groupby("gs_id"):
            f.write(name[gid] + "\t" + gid + "\t" + "\t".join(sorted(set(grp.human_gene_symbol))) + "\n")
    print(sub, len(s), "gene sets")
