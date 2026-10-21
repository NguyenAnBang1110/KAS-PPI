"""Tai hien SEEK (Sousa23) cho PPI. Chay trong thu muc goc cua repo seek.
python seek_repro.py STAGE [options]
  kg                      : dung lai KG tu go.owl + GAF bang build_kg.run_save_graph, so voi KG.gpickle tac gia
  emb  retrain|authors    : authors = dung embedding lop cua tac gia; retrain = chay lai RDF2Vec (walker + Word2Vec)
  pairs EMB_FILE OUT      : tinh bieu dien cap bang generate_pair_representations.run_PairEmbeddings, so voi file tac gia
  ml   PAIR_FILE TAG [ALGS] : 10-fold RF/XGB/MLP (tham so mac dinh nhu train_ml.py), so voi Table 2
  expl PAIR_FILE EMB_FILE TAG ALG [MAX_PER_FOLD] : necessary/sufficient (Table 3, Table 4)"""
import sys, os, json, time, pickle, ast
import numpy as np
P = './PPI/'
DATA = P + 'STRING_pairs.txt'
PART = P + 'StratifiedPartitions/Indexes__crossvalidationTest__Run'
PAPER_T2 = {'XGB': (0.920, 0.910, 0.915), 'RF': (0.922, 0.892, 0.910), 'MLP': (0.908, 0.924, 0.917)}
PAPER_T3 = {'MLP': (-0.157, -0.137, -0.148, 0.932, 0.959, 0.950), 'XGB': (-0.109, -0.120, -0.113, 1.0, 1.0, 1.0),
            'RF': (-0.106, -0.153, -0.125, 0.973, 0.888, 0.945)}
PAPER_T4 = {'MLP': (5.6, 3.9, 0.4, 1.1), 'XGB': (6.2, 3.9, 0.4, 1.1), 'RF': (5.6, 3.7, 0.4, 1.3)}
OUT = P + 'Repro/'; os.makedirs(OUT, exist_ok=True)

def load_pairs(f):
    d = {}
    for l in open(f):
        t = l.rstrip('\n').split('\t'); d[(t[0].split('/')[-1], t[1].split('/')[-1])] = np.array(t[2:], float)
    return d
def dataset():
    ents, y = [], []
    for l in open(DATA):
        t = l.rstrip('\n').split('\t'); ents.append((t[0], t[1])); y.append(int(t[2]))
    return ents, np.array(y)
def folds():
    return {r: [int(l) for l in open(PART + str(r) + '.txt') if l.strip()] for r in range(1, 11)}
def make(alg):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    import xgboost as xgb
    return {'RF': RandomForestClassifier, 'MLP': MLPClassifier, 'XGB': xgb.XGBClassifier}[alg](random_state=0)
def load_emb(f):
    # file la dict Python voi nhay don; doi sang JSON de doc nhanh va it RAM hon eval/literal_eval
    return json.loads(open(f).read().replace("'", '"'))

st = sys.argv[1]; t0 = time.time()
if st == 'kg':
    from build_kg import run_save_graph
    run_save_graph(P + 'go.owl', P + 'go_annotations.gaf', OUT + 'KG_rebuilt.gpickle', OUT + 'Labelclasses_rebuilt.pkl', 'PPI')
    A = pickle.load(open(OUT + 'KG_rebuilt.gpickle', 'rb')); B = pickle.load(open(P + 'KG.gpickle', 'rb'))
    import rdflib
    named = lambda G: set((str(u), str(v)) for u, v in G.edges() if not isinstance(u, rdflib.term.BNode) and not isinstance(v, rdflib.term.BNode))
    nb = lambda G: sum(1 for u, v in G.edges() if isinstance(u, rdflib.term.BNode) or isinstance(v, rdflib.term.BNode))
    ea, eb = named(A), named(B)
    # blank node (BNode) cua OWL nhan ID ngau nhien moi lan parse, nen chi so cac canh giua node co ten
    r = {'nodes_rebuilt': A.number_of_nodes(), 'nodes_authors': B.number_of_nodes(), 'named_edges_rebuilt': len(ea), 'named_edges_authors': len(eb),
         'named_edges_only_rebuilt': len(ea - eb), 'named_edges_only_authors': len(eb - ea), 'bnode_edges_rebuilt': nb(A), 'bnode_edges_authors': nb(B)}
elif st == 'emb':
    if sys.argv[2] == 'authors':
        r = {'emb_file': P + 'Embeddings/Emb_classes_maxdepth4_nwalks100.txt'}
    else:
        from train_embedding_model import compute_embeddings_classes
        f = OUT + 'Emb_classes_retrained.txt'
        compute_embeddings_classes(P + 'go.owl', f, 100, 100, 4)
        r = {'emb_file': f, 'n_classes': len(load_emb(f))}
elif st == 'pairs':
    # mode 'repo': goi nguyen ham run_PairEmbeddings cua repo cho moi cap (cham, khoang 1-2 gio).
    # mode 'fast': cung dinh nghia DCA, chi luu cache nx.descendants (vai phut); kem kiem tra voi ham repo tren 500 cap dau.
    import networkx as nx, rdflib
    from generate_pair_representations import run_PairEmbeddings
    emb, out = sys.argv[2], sys.argv[3]; mode = sys.argv[4] if len(sys.argv) > 4 else 'fast'; N_CHECK = 500
    AUTH = P + 'Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt'
    if mode == 'repo':
        run_PairEmbeddings(emb, out, P + 'KG.gpickle', DATA, 100)
    else:
        G = pickle.load(open(P + 'KG.gpickle', 'rb')); E = load_emb(emb); ents, _ = dataset(); desc = {}
        def Dn(n):
            if n not in desc: desc[n] = frozenset(nx.descendants(G, n)) if G.has_node(n) else frozenset()
            return desc[n]
        with open(out, 'w') as fo:
            for a, b in ents:
                com = Dn(rdflib.term.URIRef(a)) & Dn(rdflib.term.URIRef(b))
                dca = [c for c in com if not any(o != c and c in Dn(o) for o in com) and str(c) in E]
                v = np.average(np.array([E[str(c)] for c in dca]), 0).tolist() if dca else [0] * 100
                fo.write(a + '\t' + b + ''.join('\t' + str(x) for x in v) + '\n')
        sub = OUT + 'pairs_subset.txt'; open(sub, 'w').writelines(open(DATA).readlines()[:N_CHECK])
        run_PairEmbeddings(emb, OUT + 'pairs_subset_repo.txt', P + 'KG.gpickle', sub, 100)
        R, F_ = load_pairs(OUT + 'pairs_subset_repo.txt'), load_pairs(out)
        dd = max(np.abs(R[k] - F_[k]).max() for k in R)
        print(f'fast vs ham repo tren {len(R)} cap dau: lech lon nhat {dd:.3g}')
    A, B = load_pairs(out), load_pairs(AUTH)
    ks = [k for k in A if k in B]; d = np.array([np.abs(A[k] - B[k]).max() for k in ks])
    r = {'mode': mode, 'pairs': len(A), 'pairs_matched_authors': len(ks), 'max_abs_diff_vs_authors': float(d.max()),
         'frac_pairs_identical_vs_authors': float((d < 1e-9).mean())}
    if mode != 'repo': r['max_abs_diff_fast_vs_repo_function_first500'] = float(dd)
elif st == 'ml':
    from sklearn.metrics import precision_score, recall_score, f1_score
    pf, tag = sys.argv[2], sys.argv[3]; algs = sys.argv[4].split(',') if len(sys.argv) > 4 else ['RF', 'XGB', 'MLP']
    R = load_pairs(pf); ents, y = dataset(); X = np.array([R[(a, b)] for a, b in ents]); F = folds(); r = {}
    for alg in algs:
        sc = []
        for k in range(1, 11):
            te = F[k]; m = np.ones(len(y), bool); m[te] = False
            clf = make(alg).fit(X[m], y[m]); p = clf.predict(X[te])
            sc.append((precision_score(y[te], p), recall_score(y[te], p), f1_score(y[te], p, average='weighted')))
            pickle.dump(clf, open(OUT + f'Model_{tag}_{alg}_{k}.pickle', 'wb'))
        med = np.median(np.array(sc), 0).round(4).tolist()
        r[alg] = {'median_Pr_Re_F1': med, 'paper_Pr_Re_F1': PAPER_T2[alg], 'folds_F1': [round(s[2], 4) for s in sc]}
        print(alg, r[alg], flush=True)
elif st == 'expl':
    import networkx as nx, rdflib
    from sklearn.metrics import precision_score, recall_score, f1_score
    pf, ef, tag, alg = sys.argv[2:6]; mx = int(sys.argv[6]) if len(sys.argv) > 6 else None
    G = pickle.load(open(P + 'KG.gpickle', 'rb')); E = load_emb(ef); ents, y = dataset(); F = folds()
    desc = {}
    def D(n):
        if n not in desc: desc[n] = frozenset(str(x) for x in nx.descendants(G, rdflib.term.URIRef(n))) if G.has_node(rdflib.term.URIRef(n)) else frozenset()
        return desc[n]
    def dcas(a, b):  # giong het getExplanations / run_PairEmbeddings
        com = D(a) & D(b)
        return [c for c in com if not any(o != c and c in D(o) for o in com) and c in E]
    avg = lambda V: np.mean(V, 0) if len(V) else np.zeros(100)
    rows, nec_len, suf_len, dca_len = [], [], [], []
    for k in range(1, 11):
        clf = pickle.load(open(OUT + f'Model_{tag}_{alg}_{k}.pickle', 'rb'))
        # partition cua repo xep cap duong truoc cap am, nen lay mau ngau nhien (seed = fold) thay vi N cap dau
        te = sorted(np.random.RandomState(k).choice(F[k], mx, replace=False)) if mx else F[k]
        cy, cg, cw, iy, ig, io = [], [], [], [], [], []
        for i in te:
            a, b = ents[i]; ds = dcas(a, b); V = [np.array(E[c]) for c in ds]; n = len(V)
            # gom moi bien the vao mot lan goi predict: [global, without_1..n, only_1..n]
            Xs = [avg(V)] + [avg(V[:j] + V[j + 1:]) for j in range(n)] + [V[j] for j in range(n)]
            pr = clf.predict(np.array(Xs)); g = pr[0]
            nec = [j for j in range(n) if pr[1 + j] != g]; suf = [j for j in range(n) if pr[1 + n + j] == g]
            p2 = clf.predict(np.array([avg([V[j] for j in range(n) if j not in nec]), avg([V[j] for j in suf])]))
            nec_len.append(len(nec)); suf_len.append(len(suf)); dca_len.append(n)
            if g == y[i]: cy.append(y[i]); cg.append(g); cw.append(p2[0])
            else: iy.append(y[i]); ig.append(g); io.append(p2[1])
        met = lambda t, p: (precision_score(t, p, zero_division=0), recall_score(t, p, zero_division=0), f1_score(t, p, average='weighted', zero_division=0))
        a1, a2, b1, b2 = met(cy, cg), met(cy, cw), met(iy, ig), met(iy, io)
        b3 = met(ig, io)  # only sufficient, cham theo du doan goc cua mo hinh thay vi nhan that
        rows.append([a2[j] - a1[j] for j in range(3)] + [b2[j] - b1[j] for j in range(3)] + list(b3))
        print('fold', k, np.round(rows[-1], 3).tolist(), round(time.time() - t0), 's', flush=True)
    med = np.median(rows, 0).round(3).tolist()
    r = {'table3_median_dPr_dRe_dF1_wo_necessary__dPr_dRe_dF1_only_sufficient': med[:6],
         'only_sufficient_vs_model_prediction_Pr_Re_F1': med[6:],
         'paper_table3_RDF2Vec': PAPER_T3[alg],
         'table4_suff_avg_std_nec_avg_std': [round(np.mean(suf_len), 2), round(np.std(suf_len), 2), round(np.mean(nec_len), 2), round(np.std(nec_len), 2)],
         'paper_table4_RDF2Vec': PAPER_T4[alg], 'shared_aspects_avg_std': [round(np.mean(dca_len), 2), round(np.std(dca_len), 2)], 'paper_shared': (9.1, 6.5)}
r['seconds'] = round(time.time() - t0)
print(json.dumps(r, indent=1))
name = {'ml': f'ml_{sys.argv[3] if len(sys.argv) > 3 else ""}', 'expl': f'expl_{sys.argv[4] if len(sys.argv) > 4 else ""}_{sys.argv[5] if len(sys.argv) > 5 else ""}'}.get(st, st)
json.dump(r, open(OUT + 'result_' + name + '.json', 'w'), indent=1, default=float)
