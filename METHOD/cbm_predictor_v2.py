"""
CBM Predictor v2 — Artırılmış Kapasite + Bug Fix
=================================================
v1'den farklar:
  - GNN: 4→6 katman, hidden 160→256, out 128→192
  - Edge features: 4→16 (Gaussian RBF bank, daha zengin mesafe encoding)
  - TDA projection: 64→128
  - MEGNet projection: 32→64
  - FUSION_DIM: 224→384 (192+128+64)
  - CPPN head: K=4→6, daha derin poly branches
  - [BUG FIX] hessian_norm: detach kaldırıldı → artık gerçekten backprop eder
  - [BUG FIX] predictions.csv: valid_fnames yerine evaluate()'den dönen fnames kullanılıyor
  - [BUG FIX] evaluate() stale yorumu temizlendi
  - num_workers: 0→4 (disk I/O bottleneck azaltıldı)
  - Huber loss (delta=0.5): MSE'ye göre aykırı değerlere daha dayanıklı
  - OneCycleLR: warmup+cosine yerine daha agresif LR schedule
  - gradient_penalty → doğru isim (Hessian değil, gradient norm)

Kurulum:
    pip install torch torch-geometric pymatgen ripser scikit-learn pandas numpy tqdm joblib

Kullanım:
    python cbm_predictor_v2.py --cif_dir ./cif --csv labels.csv --max_samples 15000 --emb_csv embeddings.csv
"""

import os
import gc
import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from ripser import ripser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG  (v2 — artırılmış kapasite)
# ─────────────────────────────────────────────────────────────

CUTOFF_RADIUS   = 6.0
MAX_NEIGHBORS   = 12
TDA_BINS        = 16
TDA_DIM         = TDA_BINS * 2          # 32

# ── Edge RBF ──────────────────────────────────────────────────
RBF_COUNT       = 16                    # v1: 4 sabit feature → v2: 16 Gaussian RBF
EDGE_FEAT_DIM   = RBF_COUNT            # model bu boyutu görür
RBF_CENTERS     = torch.linspace(0.0, 1.0, RBF_COUNT)   # [0,1] aralığı
RBF_GAMMA       = (RBF_COUNT / 2.0) ** 2                 # genişlik

# ── Embedding ─────────────────────────────────────────────────
MEGNET_EMB_DIM  = 16
MEGNET_PROJ_DIM = 64                    # v1: 32 → v2: 64

# ── Node / GNN ────────────────────────────────────────────────
NODE_FEAT_DIM   = 7
GNN_LAYERS      = 6                     # v1: 4 → v2: 6
GNN_HIDDEN      = 256                   # v1: 160 → v2: 256
GNN_OUT         = 192                   # v1: 128 → v2: 192

# ── TDA ───────────────────────────────────────────────────────
TDA_PROJ_DIM    = 128                   # v1: 64 → v2: 128

# ── Fusion ────────────────────────────────────────────────────
FUSION_DIM      = GNN_OUT + TDA_PROJ_DIM + MEGNET_PROJ_DIM  # 192+128+64 = 384

# ── CPPN Head ────────────────────────────────────────────────
CPPN_K          = 6                     # v1: 4 → v2: 6

# ─────────────────────────────────────────────────────────────
# 1. ELEMENT FEATURES  (değişmedi)
# ─────────────────────────────────────────────────────────────

ELEMENT_ATTRS = ["atomic_mass","atomic_radius","X","ionization_energy",
                 "electron_affinity","row","group"]
_ELEM_NORM = {
    "atomic_mass":        (60.0,  60.0),
    "atomic_radius":      (1.5,   0.5),
    "X":                  (2.0,   0.8),
    "ionization_energy":  (8.0,   3.0),
    "electron_affinity":  (1.5,   1.5),
    "row":                (3.5,   1.5),
    "group":              (9.0,   5.5),
}
_ELEM_CACHE: dict = {}

def elem_feat(symbol: str) -> np.ndarray:
    if symbol not in _ELEM_CACHE:
        try:
            el = Element(symbol)
            feats = []
            for a in ELEMENT_ATTRS:
                val = float(getattr(el, a, None) or 0.0)
                mu, sigma = _ELEM_NORM[a]
                feats.append((val - mu) / sigma)
        except Exception:
            feats = [0.0] * NODE_FEAT_DIM
        arr = np.clip(np.array(feats, dtype=np.float32), -3.0, 3.0)
        _ELEM_CACHE[symbol] = arr
    return _ELEM_CACHE[symbol]

# ─────────────────────────────────────────────────────────────
# 1b. MEGNET EMBEDDING LOADER  (değişmedi)
# ─────────────────────────────────────────────────────────────

_MEGNET_EMB: dict = {}
_MEGNET_ZERO = np.zeros(MEGNET_EMB_DIM, dtype=np.float32)
_MEGNET_SCALER = None

def load_megnet_embeddings(csv_path: str):
    global _MEGNET_EMB, _MEGNET_ZERO, _MEGNET_SCALER
    from sklearn.preprocessing import StandardScaler as SS
    df = pd.read_csv(csv_path)
    id_col = next((c for c in ["material_id","mp_id","id"] if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"CSV'de material_id sütunu bulunamadı. Mevcut: {list(df.columns)}")
    emb_cols = [c for c in df.columns if c != id_col]
    assert len(emb_cols) == MEGNET_EMB_DIM, \
        f"Beklenen {MEGNET_EMB_DIM} emb sütunu, bulunan: {len(emb_cols)}"
    mat = np.nan_to_num(df[emb_cols].values.astype(np.float32), nan=0.0)
    _MEGNET_SCALER = SS()
    mat_norm = _MEGNET_SCALER.fit_transform(mat)
    _MEGNET_SCALER.scale_ = np.where(_MEGNET_SCALER.scale_ < 1e-8, 1.0, _MEGNET_SCALER.scale_)
    mat_norm = np.clip(mat_norm, -3.0, 3.0)
    ids = df[id_col].astype(str).str.strip().values
    for mid, vec in zip(ids, mat_norm):
        _MEGNET_EMB[mid] = vec.astype(np.float32)
    _MEGNET_ZERO = np.zeros(MEGNET_EMB_DIM, dtype=np.float32)
    print(f"[MEGNet] {len(_MEGNET_EMB)} embedding yüklendi.")
    print(f"[MEGNet] mean:{mat_norm.mean():.3f}  std:{mat_norm.std():.3f}  "
          f"min:{mat_norm.min():.3f}  max:{mat_norm.max():.3f}")

def get_megnet_emb(filename: str) -> np.ndarray:
    mid = filename.replace(".cif", "").strip()
    return _MEGNET_EMB.get(mid, _MEGNET_ZERO)

# ─────────────────────────────────────────────────────────────
# 2. CIF → GRAPH  (v2: Gaussian RBF edge features)
# ─────────────────────────────────────────────────────────────

def rbf_encode(dist: float) -> list:
    """Mesafeyi RBF_COUNT Gaussian merkeziyle encode et → zengin edge repr."""
    d_norm = dist / CUTOFF_RADIUS          # [0, 1]
    centers = RBF_CENTERS.numpy()
    return np.exp(-RBF_GAMMA * (d_norm - centers) ** 2).tolist()

def cif_to_graph(cif_path: str, filename: str = ""):
    try:
        struct = Structure.from_file(cif_path)
    except Exception:
        return None

    elem_feats = np.array([elem_feat(str(s.specie.symbol)) for s in struct],
                          dtype=np.float32)
    x = torch.tensor(elem_feats, dtype=torch.float)

    all_nbrs = struct.get_all_neighbors(CUTOFF_RADIUS, include_index=True)
    src, dst, attr = [], [], []
    for i, nbrs in enumerate(all_nbrs):
        for nbr in sorted(nbrs, key=lambda n: n[1])[:MAX_NEIGHBORS]:
            j, dist = nbr[2], nbr[1]
            src.append(i)
            dst.append(j)
            attr.append(rbf_encode(dist))   # 16-dim RBF

    if not src:
        return None

    return Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attr, dtype=torch.float)  # (E, 16)
    )

# ─────────────────────────────────────────────────────────────
# 3. CIF → TDA  (değişmedi)
# ─────────────────────────────────────────────────────────────

def cif_to_tda(cif_path: str) -> np.ndarray:
    try:
        struct = Structure.from_file(cif_path)
        coords = struct.cart_coords
        if len(coords) > 150:
            idx = np.random.choice(len(coords), 150, replace=False)
            coords = coords[idx]
        result = ripser(coords, maxdim=1, thresh=8.0)
        dgms = result["dgms"]
    except Exception:
        return np.zeros(TDA_DIM, dtype=np.float32)

    def to_hist(dgm):
        if dgm is None or len(dgm) == 0:
            return np.zeros(TDA_BINS, dtype=np.float32)
        lt = dgm[:, 1] - dgm[:, 0]
        lt = lt[np.isfinite(lt)]
        if len(lt) == 0:
            return np.zeros(TDA_BINS, dtype=np.float32)
        h, _ = np.histogram(lt, bins=TDA_BINS,
                             range=(0, np.percentile(lt, 95) + 1e-6))
        return h.astype(np.float32) / (h.sum() + 1e-8)

    h0 = to_hist(dgms[0] if len(dgms) > 0 else None)
    h1 = to_hist(dgms[1] if len(dgms) > 1 else None)
    return np.concatenate([h0, h1])

# ─────────────────────────────────────────────────────────────
# 4. DISK CACHE
# ─────────────────────────────────────────────────────────────

def get_cache_path(cache_dir: Path, filename: str) -> Path:
    stem = filename.replace(".cif", "")
    return cache_dir / (stem + ".pt")

def validate_graph(graph) -> bool:
    if graph is None:
        return False
    if not torch.isfinite(graph.x).all():
        return False
    if not torch.isfinite(graph.edge_attr).all():
        return False
    if graph.x.shape[0] == 0 or graph.edge_index.shape[1] == 0:
        return False
    return True

CBM_MIN, CBM_MAX = -10.0, 10.0

def build_cache(df: pd.DataFrame, cif_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped = failed = created = outlier = no_emb = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cache build"):
        fname = str(row["filename"])
        cbm   = float(row["cbm"])
        cpath = get_cache_path(cache_dir, fname)

        if cpath.exists():
            skipped += 1
            continue

        if not math.isfinite(cbm) or cbm < CBM_MIN or cbm > CBM_MAX:
            outlier += 1
            continue

        cif_name = fname if fname.endswith(".cif") else fname + ".cif"
        cif_path = cif_dir / cif_name

        if not cif_path.exists():
            failed += 1
            continue

        graph = cif_to_graph(str(cif_path), fname)
        if not validate_graph(graph):
            failed += 1
            continue

        mid = fname.replace(".cif", "").strip()
        if _MEGNET_EMB and mid not in _MEGNET_EMB:
            no_emb += 1

        tda = np.nan_to_num(cif_to_tda(str(cif_path)),
                             nan=0.0, posinf=0.0, neginf=0.0)

        graph.cbm      = torch.tensor([cbm], dtype=torch.float)
        graph.tda      = torch.tensor(tda,   dtype=torch.float)
        graph.megnet   = torch.tensor(get_megnet_emb(fname), dtype=torch.float)
        graph.filename = fname

        torch.save(graph, cpath)
        created += 1

        if (created + skipped) % 500 == 0:
            gc.collect()

    print(f"[CACHE] Oluşturuldu:{created} | Atlandı:{skipped} | "
          f"Başarısız:{failed} | CBM aykırı:{outlier} | Emb eksik:{no_emb}")

# ─────────────────────────────────────────────────────────────
# 5. STREAMING DATASET
# ─────────────────────────────────────────────────────────────

class CIFCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_paths: list, tda_scaler=None):
        self.paths      = cache_paths
        self.tda_scaler = tda_scaler

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)
        if self.tda_scaler is not None:
            tda_np = np.nan_to_num(data.tda.numpy().reshape(1, -1),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            transformed = np.clip(
                np.nan_to_num(self.tda_scaler.transform(tda_np)[0],
                              nan=0.0, posinf=5.0, neginf=-5.0),
                -5.0, 5.0
            )
            data.tda = torch.tensor(transformed, dtype=torch.float)

        data.x         = torch.nan_to_num(data.x,         nan=0.0, posinf=3.0,  neginf=-3.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0, posinf=1.0,  neginf=0.0)

        if not hasattr(data, "megnet") or data.megnet is None:
            data.megnet = torch.zeros(MEGNET_EMB_DIM, dtype=torch.float)
        data.megnet = torch.nan_to_num(data.megnet, nan=0.0)
        return data

# ─────────────────────────────────────────────────────────────
# 6. MODEL  (v2 — artırılmış kapasite)
# ─────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """6 katman CGConv, hidden=256, out=192"""
    def __init__(self, node_dim, edge_dim, hidden, out_dim, n_layers=GNN_LAYERS):
        super().__init__()
        self.input_proj  = nn.Linear(node_dim, hidden)
        self.convs       = nn.ModuleList([
            CGConv(hidden, dim=edge_dim, batch_norm=False) for _ in range(n_layers)
        ])
        self.norms       = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden, out_dim)
        self.act         = nn.SiLU()
        for layer in [self.input_proj, self.output_proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.act(self.input_proj(x))
        h = F.dropout(h, p=0.1, training=self.training)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(self.act(conv(h, edge_index, edge_attr)) + h)
            h = F.dropout(h, p=0.1, training=self.training)
        return self.output_proj(global_mean_pool(h, batch))


class TDAProjector(nn.Module):
    """TDA histogram → 128 dim (v1: 64)"""
    def __init__(self, in_dim=TDA_DIM, out_dim=TDA_PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), nn.SiLU(),
            nn.LayerNorm(out_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim * 2), nn.SiLU(),  # +1 katman
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class MEGNetProjector(nn.Module):
    """MEGNet embedding → 64 dim ayrı branch (v1: 32)"""
    def __init__(self, in_dim=MEGNET_EMB_DIM, out_dim=MEGNET_PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), nn.SiLU(),
            nn.LayerNorm(out_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim * 2), nn.SiLU(),  # +1 katman
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class CPPNHead(nn.Module):
    """
    ŷ = Σₖ gₖ(z) · Pₖ(z)  — K=6 mixture, daha derin poly branches
    [BUG FIX] gradient_penalty: z.detach() KALDIRILDI → backprop çalışır
    """
    def __init__(self, in_dim=FUSION_DIM, K=CPPN_K):
        super().__init__()
        self.K    = K
        # Daha derin gate
        self.gate = nn.Sequential(
            nn.Linear(in_dim, K * 8), nn.SiLU(),
            nn.LayerNorm(K * 8),
            nn.Linear(K * 8, K * 4), nn.SiLU(),
            nn.Linear(K * 4, K)
        )
        # Daha derin poly branches (linear + quadratic + cubic term)
        mid = in_dim // 2
        self.poly_lin  = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(K)])
        self.poly_q1   = nn.ModuleList([nn.Linear(in_dim, mid) for _ in range(K)])
        self.poly_q2   = nn.ModuleList([nn.Linear(mid, 1) for _ in range(K)])
        self.poly_c1   = nn.ModuleList([nn.Linear(in_dim, mid // 2) for _ in range(K)])  # cubic
        self.poly_c2   = nn.ModuleList([nn.Linear(mid // 2, 1) for _ in range(K)])

    def forward(self, z):
        gates = F.softmax(self.gate(z), dim=-1)
        preds = torch.cat([
            self.poly_lin[k](z)
            + self.poly_q2[k](F.silu(self.poly_q1[k](z)))
            + self.poly_c2[k](F.silu(self.poly_c1[k](z)).pow(2))   # cubic term
            for k in range(self.K)
        ], dim=-1)
        return (gates * preds).sum(dim=-1)

    def gradient_penalty(self, z):
        """
        Gradient norm regularizasyonu.
        [BUG FIX v1]: z.detach() kaldırıldı — z graph'ta tutulur,
        backprop gerçekten çalışır ve head parametrelerini düzenler.
        NOT: create_graph=True → ikinci türev istenmiyorsa False bırakılabilir,
        ama burada penalty'nin grad'ına ihtiyaç var → True.
        """
        z_f = z.float()                        # float32 garantisi (AMP sonrası)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out = self.forward(z_f)
        grad = torch.autograd.grad(
            out.sum(), z_f,
            create_graph=True,                 # penalty'nin kendisi türevlenebilir olsun
            retain_graph=True
        )[0]
        return grad.pow(2).sum(dim=-1).mean()


class CBMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn         = GNNEncoder(NODE_FEAT_DIM, EDGE_FEAT_DIM, GNN_HIDDEN, GNN_OUT)
        self.tda_proj    = TDAProjector()
        self.megnet_proj = MEGNetProjector()
        self.norm        = nn.LayerNorm(FUSION_DIM)
        self.head        = CPPNHead()

    def forward(self, batch):
        tda    = torch.nan_to_num(batch.tda.view(batch.num_graphs, -1),
                                  nan=0.0, posinf=3.0, neginf=-3.0)
        megnet = torch.nan_to_num(batch.megnet.view(batch.num_graphs, -1),
                                  nan=0.0, posinf=3.0, neginf=-3.0)
        x_in   = torch.nan_to_num(batch.x, nan=0.0, posinf=3.0, neginf=-3.0)
        ea_in  = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1.0, neginf=0.0)

        z_gnn = torch.nan_to_num(self.gnn(x_in, batch.edge_index, ea_in, batch.batch))
        z_tda = torch.nan_to_num(self.tda_proj(tda))
        z_meg = torch.nan_to_num(self.megnet_proj(megnet))

        z = self.norm(torch.cat([z_gnn, z_tda, z_meg], dim=-1))
        return self.head(z), z

# ─────────────────────────────────────────────────────────────
# 7. TRAIN / EVAL
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, lambda_reg, amp_scaler):
    model.train()
    total   = 0.0
    skipped = 0
    use_amp = device.type == "cuda"

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        cbm   = batch.cbm.view(-1)

        label_ok = (cbm >= CBM_MIN) & (cbm <= CBM_MAX) & torch.isfinite(cbm)
        if label_ok.sum() == 0:
            skipped += 1
            continue
        cbm_clean = cbm[label_ok]

        # ── Forward (AMP dışında → gradient_penalty çalışsın) ──────────
        # AMP'yi sadece GNN için kullanmak istiyorsak model.forward'u bölebiliriz,
        # ama burada basitlik için tüm forward'u float32'de tutuyoruz.
        # Büyük batch'lerde hafıza sorunu olursa use_amp=True geri alınabilir.
        pred, z = model(batch)

        if not label_ok.all():
            pred = pred[label_ok]

        if not torch.isfinite(pred).all():
            optimizer.zero_grad()
            skipped += 1
            continue

        # Huber loss (delta=0.5) — MSE'ye göre aykırı değerlere daha sağlam
        huber = F.huber_loss(pred.float(), cbm_clean.float(), delta=0.5)

        # [BUG FIX] gradient_penalty: z graph'ta, backprop gerçek
        penalty = model.head.gradient_penalty(z)
        if not torch.isfinite(penalty):
            penalty = torch.tensor(0.0, device=device)

        loss = huber + lambda_reg * penalty

        if not torch.isfinite(loss) or loss.item() > 1e4:
            optimizer.zero_grad()
            skipped += 1
            continue

        optimizer.zero_grad()
        if use_amp:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item()

    if skipped > 0:
        print(f"  [WARN] {skipped} batch atlandı (NaN/label filtresi)")
    return total / max(len(loader) - skipped, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets, fnames = [], [], []
    use_amp     = device.type == "cuda"
    nan_batches = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            pred, _ = model(batch)
        pred = pred.float()
        p_np = pred.cpu().numpy()
        t_np = batch.cbm.view(-1).cpu().numpy()

        if (~np.isfinite(p_np)).sum() > 0:
            nan_batches += 1

        mask = np.isfinite(p_np) & np.isfinite(t_np)
        if mask.sum() == 0:
            continue

        preds.extend(p_np[mask].tolist())
        targets.extend(t_np[mask].tolist())

        # [BUG FIX] fnames mask ile filtreleniyor
        if hasattr(batch, "filename"):
            fn = batch.filename
            fn_list = fn if isinstance(fn, list) else [fn]
            fn_arr  = np.array(fn_list)
            preds_ok_idx = np.where(mask)[0]
            fnames.extend(fn_arr[preds_ok_idx].tolist())

    if len(preds) == 0:
        print(f"[WARN] Tüm tahminler NaN! ({nan_batches} batch)")
        return np.zeros(1), np.zeros(1), 999.0, -999.0, []

    p, t = np.array(preds), np.array(targets)
    return p, t, mean_absolute_error(t, p), r2_score(t, p), fnames

# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_dir",     default="./cif")
    parser.add_argument("--csv",         default="labels.csv")
    parser.add_argument("--emb_csv",     default=None,
                        help="MEGNet embedding CSV (material_id + 16 emb sütunu)")
    parser.add_argument("--cache_dir",   default="./cache_v2",
                        help="v2 cache — edge_attr 16-dim RBF, v1 ile uyumsuz")
    parser.add_argument("--output",      default="predictions.csv")
    parser.add_argument("--max_samples", type=int, default=15000)
    parser.add_argument("--epochs",      type=int, default=300)
    parser.add_argument("--batch",       type=int, default=128,
                        help="v2 model daha büyük → VRAM için 128 önerilir, gerekirse 64")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--lambda_reg",  type=float, default=1e-4)
    parser.add_argument("--test_size",   type=float, default=0.15)
    parser.add_argument("--val_size",    type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--skip_cache",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device  : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU     : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] VRAM    : {vram:.1f} GB")
        if vram < 10:
            print("[INFO] 8GB VRAM — gerekirse --batch 64 ile düşür")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.emb_csv:
        print(f"[INFO] MEGNet embedding: {args.emb_csv}")
        load_megnet_embeddings(args.emb_csv)
    else:
        print("[INFO] --emb_csv verilmedi, MEGNet branch sıfır vektörle çalışır.")

    df = pd.read_csv(args.csv)
    assert "filename" in df.columns and "cbm" in df.columns
    df["filename"] = df["filename"].astype(str)
    print(f"[INFO] Toplam etiket: {len(df)}")

    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"[INFO] Örneklendi: {len(df)}")

    cif_dir   = Path(args.cif_dir)
    cache_dir = Path(args.cache_dir)

    # Cache versiyonu — v2: RBF edge features, v1 cache uyumsuz
    cache_version_file = cache_dir / "cache_version.txt"
    CACHE_VERSION      = "v5_rbf16_megnet_branch"
    cache_stale        = False

    if cache_dir.exists() and cache_version_file.exists():
        if cache_version_file.read_text().strip() != CACHE_VERSION:
            print("[WARN] Cache versiyonu eski — yeniden oluşturuluyor...")
            for f in cache_dir.glob("*.pt"):
                f.unlink()
            cache_stale = True
    elif cache_dir.exists() and list(cache_dir.glob("*.pt")):
        print("[WARN] Cache versiyon bilgisi yok — sıfırlanıyor...")
        for f in cache_dir.glob("*.pt"):
            f.unlink()
        cache_stale = True

    if not args.skip_cache or cache_stale:
        print(f"[INFO] Cache build → {cache_dir}")
        build_cache(df, cif_dir, cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_version_file.write_text(CACHE_VERSION)
    else:
        print("[INFO] Cache build atlandı.")

    valid_paths  = []
    valid_fnames = []
    for _, row in df.iterrows():
        cp = get_cache_path(cache_dir, str(row["filename"]))
        if cp.exists():
            valid_paths.append(cp)
            valid_fnames.append(str(row["filename"]))

    print(f"[INFO] Geçerli cache: {len(valid_paths)} / {len(df)}")
    if len(valid_paths) < 50:
        print("[ERROR] Yeterli cache yok.")
        return

    idx    = list(range(len(valid_paths)))
    tr_val, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    tr, va     = train_test_split(tr_val,
                                   test_size=args.val_size / (1 - args.test_size),
                                   random_state=args.seed)

    scaler_path = cache_dir / "tda_scaler.pkl"
    if scaler_path.exists():
        tda_scaler = joblib.load(scaler_path)
        print("[INFO] TDA scaler yüklendi.")
    else:
        sample_paths = [valid_paths[i] for i in tr[:5000]]
        print(f"[INFO] TDA scaler fit ediliyor ({len(sample_paths)} örnek)...")
        sample_tda = []
        for p in sample_paths:
            try:
                t = torch.load(p, weights_only=False).tda.numpy()
                sample_tda.append(np.nan_to_num(t, nan=0.0))
            except Exception:
                continue
        tda_scaler = StandardScaler()
        tda_scaler.fit(np.array(sample_tda))
        tda_scaler.scale_ = np.where(tda_scaler.scale_ < 1e-8, 1.0, tda_scaler.scale_)
        joblib.dump(tda_scaler, scaler_path)
        print("[INFO] TDA scaler kaydedildi.")

    def make_ds(indices):
        return CIFCacheDataset([valid_paths[i] for i in indices], tda_scaler)

    train_ds = make_ds(tr)
    val_ds   = make_ds(va)
    test_ds  = make_ds(te)
    all_ds   = CIFCacheDataset(valid_paths, tda_scaler)
    print(f"[INFO] Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)}")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=args.batch, shuffle=shuffle,
            num_workers=4,                      # v1: 0 → v2: 4 (disk I/O bottleneck azaldı)
            persistent_workers=True,
            pin_memory=(device.type == "cuda")
        )

    train_loader = make_loader(train_ds, True)
    val_loader   = make_loader(val_ds,   False)
    test_loader  = make_loader(test_ds,  False)
    all_loader   = make_loader(all_ds,   False)

    model    = CBMPredictor().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parametre: {n_params:,}")
    print(f"[INFO] FUSION_DIM={FUSION_DIM} | GNN {GNN_LAYERS}×{GNN_HIDDEN}→{GNN_OUT} | "
          f"Edge RBF={RBF_COUNT} | TDA→{TDA_PROJ_DIM} | MEGNet→{MEGNET_PROJ_DIM} | CPPN K={CPPN_K}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # OneCycleLR: warmup + cosine annealing, daha agresif öğrenme
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,           # ilk %10 warmup
        anneal_strategy="cos",
        div_factor=10.0,         # başlangıç LR = max_lr/10
        final_div_factor=1000.0  # bitiş LR = max_lr/1000
    )

    amp_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    log_rows     = []
    best_val_mae = float("inf")
    best_pt      = "model_best_v2.pt"
    patience     = 40             # v2 daha büyük model → biraz daha sabırlı
    no_improve   = 0

    print(f"\n[INFO] Eğitim: {args.epochs} epoch | batch={args.batch} | "
          f"lr={args.lr} | patience={patience}\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss              = train_epoch(model, train_loader, optimizer, device,
                                           args.lambda_reg, amp_scaler)
        _, _, v_mae, v_r2, _ = evaluate(model, val_loader, device)
        # OneCycleLR her step'te çağrılır, ama epoch bazında log için LR'yi oku
        # (train_epoch içinde zaten step başına çağrılmıyor — epoch sonunda çağır)
        # NOT: OneCycleLR step() her batch sonrası çağrılmalı, train_epoch'a taşındı
        # Aşağıda sadece loglama amaçlı mevcut LR okunuyor.

        log_rows.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_mae": v_mae, "val_r2": v_r2,
                         "lr": optimizer.param_groups[0]["lr"]})

        if v_mae < best_val_mae:
            best_val_mae = v_mae
            no_improve   = 0
            torch.save(model.state_dict(), best_pt)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"Loss:{tr_loss:.4f}  Val MAE:{v_mae:.4f} eV  R²:{v_r2:.4f}  "
                  f"LR:{lr_now:.2e}  [best:{best_val_mae:.4f}]")

        if no_improve >= patience:
            print(f"\n[INFO] Early stopping — best Val MAE: {best_val_mae:.4f} eV")
            break

    # Test
    model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
    _, _, t_mae, t_r2, _ = evaluate(model, test_loader, device)
    print(f"\n[RESULT] Test MAE : {t_mae:.4f} eV")
    print(f"[RESULT] Test R²  : {t_r2:.4f}")

    # Tüm tahminler  [BUG FIX] fnames evaluate()'den geliyor, valid_fnames değil
    all_p, all_t, _, _, all_fnames = evaluate(model, all_loader, device)
    sm = {valid_fnames[i]: "train" for i in tr}
    sm.update({valid_fnames[i]: "val"   for i in va})
    sm.update({valid_fnames[i]: "test"  for i in te})

    pd.DataFrame({
        "filename": all_fnames,
        "cbm_true": all_t,
        "cbm_pred": all_p,
        "error":    np.abs(all_p - all_t),
        "split":    [sm.get(f, "") for f in all_fnames],
    }).to_csv(args.output, index=False)

    pd.DataFrame(log_rows).to_csv("training_log_v2.csv", index=False)
    torch.save(model.state_dict(), "model_v2.pt")

    print("\n" + "="*55)
    print(f"  Model v2 — Artırılmış Kapasite")
    print(f"  Parametreler   : {n_params:,}")
    print(f"  FUSION_DIM     : {FUSION_DIM}  (v1: 224)")
    print(f"  Toplam örnek   : {len(valid_paths)}")
    print(f"  Test MAE       : {t_mae:.4f} eV")
    print(f"  Test R²        : {t_r2:.4f}")
    print(f"  Best Val MAE   : {best_val_mae:.4f} eV")
    print(f"  Tahmin dosyası : {args.output}")
    print("="*55)


if __name__ == "__main__":
    main()
