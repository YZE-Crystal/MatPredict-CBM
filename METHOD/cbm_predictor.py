"""
CBM Predictor — Scalable CIF Pipeline (Disk Cache + Streaming)
===============================================================
Girdi  : ./cif/         → .cif dosyaları
         labels.csv     → sütunlar: filename, cbm
Çıktı  : predictions.csv, model.pt, training_log.csv

Özellikler:
  - Disk-based cache (graph + TDA → .pt dosyaları)
  - İkinci çalıştırmada preprocessing atlanır
  - RAM'de asla tüm veri tutulmaz
  - 8GB VRAM için optimize batch boyutu
  - --max_samples ile örnekleme

Kurulum:
    pip install torch torch-geometric pymatgen ripser scikit-learn pandas numpy tqdm joblib

Kullanım (ilk çalıştırma - 15K örnek):
    python cbm_predictor.py --cif_dir ./cif --csv labels.csv --max_samples 15000

Sonraki çalıştırma (cache var, hızlı):
    python cbm_predictor.py --cif_dir ./cif --csv labels.csv --max_samples 15000 --skip_cache
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
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.data import Data, Batch
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
# CONFIG
# ─────────────────────────────────────────────────────────────

CUTOFF_RADIUS = 6.0
MAX_NEIGHBORS = 12
TDA_BINS      = 16
TDA_DIM       = TDA_BINS * 2
NODE_FEAT_DIM = 7
EDGE_FEAT_DIM = 4
GNN_HIDDEN    = 128
GNN_OUT       = 128
TDA_PROJ_DIM  = 64
FUSION_DIM    = GNN_OUT + TDA_PROJ_DIM
CPPN_K        = 4

# ─────────────────────────────────────────────────────────────
# 1. ELEMENT FEATURES
# ─────────────────────────────────────────────────────────────

ELEMENT_ATTRS = ["atomic_mass","atomic_radius","X","ionization_energy",
                 "electron_affinity","row","group"]
# Approximate normalization constants for each attribute (mean, std)
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
        arr = np.array(feats, dtype=np.float32)
        # Clip to [-3, 3] to prevent outlier NaN
        arr = np.clip(arr, -3.0, 3.0)
        _ELEM_CACHE[symbol] = arr
    return _ELEM_CACHE[symbol]

# ─────────────────────────────────────────────────────────────
# 2. CIF → GRAPH
# ─────────────────────────────────────────────────────────────

def cif_to_graph(cif_path: str):
    try:
        struct = Structure.from_file(cif_path)
    except Exception:
        return None

    x = torch.tensor(
        np.array([elem_feat(str(s.specie.symbol)) for s in struct]),
        dtype=torch.float
    )

    all_nbrs = struct.get_all_neighbors(CUTOFF_RADIUS, include_index=True)
    src, dst, attr = [], [], []
    for i, nbrs in enumerate(all_nbrs):
        for nbr in sorted(nbrs, key=lambda n: n[1])[:MAX_NEIGHBORS]:
            j, dist = nbr[2], nbr[1]
            # Normalize: dist/CUTOFF, sin/cos encoding, Gaussian RBF-like
            d_norm = dist / CUTOFF_RADIUS          # [0,1]
            d_inv  = 1.0 / (dist + 1e-6)          # ~inverse distance
            d_sq   = (d_norm) ** 2
            d_exp  = math.exp(-dist / 2.0)        # Gaussian decay
            src.append(i); dst.append(j)
            attr.append([d_norm, d_inv, d_sq, d_exp])

    if not src:
        return None

    return Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attr, dtype=torch.float)
    )

# ─────────────────────────────────────────────────────────────
# 3. CIF → TDA
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
    """Graph tensörlerinde NaN/Inf/aşırı değer kontrolü."""
    if graph is None:
        return False
    if not torch.isfinite(graph.x).all():
        return False
    if not torch.isfinite(graph.edge_attr).all():
        return False
    if graph.x.shape[0] == 0 or graph.edge_index.shape[1] == 0:
        return False
    return True

CBM_MIN, CBM_MAX = -10.0, 10.0  # eV — fiziksel olarak makul aralık

def build_cache(df: pd.DataFrame, cif_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped = failed = created = outlier = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cache build"):
        fname = str(row["filename"])
        cbm   = float(row["cbm"])
        cpath = get_cache_path(cache_dir, fname)

        if cpath.exists():
            skipped += 1
            continue

        # CBM aykırı değer filtresi
        if not math.isfinite(cbm) or cbm < CBM_MIN or cbm > CBM_MAX:
            outlier += 1
            continue

        cif_name = fname if fname.endswith(".cif") else fname + ".cif"
        cif_path = cif_dir / cif_name

        if not cif_path.exists():
            failed += 1
            continue

        graph = cif_to_graph(str(cif_path))
        if not validate_graph(graph):
            failed += 1
            continue

        tda = cif_to_tda(str(cif_path))
        tda = np.nan_to_num(tda, nan=0.0, posinf=0.0, neginf=0.0)

        graph.cbm      = torch.tensor([cbm], dtype=torch.float)
        graph.tda      = torch.tensor(tda,   dtype=torch.float)
        graph.filename = fname

        torch.save(graph, cpath)
        created += 1

        if (created + skipped) % 500 == 0:
            gc.collect()

    print(f"[CACHE] Oluşturuldu: {created} | Atlandı (var): {skipped} | Başarısız: {failed} | CBM aykırı: {outlier}")

# ─────────────────────────────────────────────────────────────
# 5. STREAMING DATASET
# ─────────────────────────────────────────────────────────────

class CIFCacheDataset(torch.utils.data.Dataset):
    """Diskten tek tek yükler — RAM'de tüm veri tutulmaz."""
    def __init__(self, cache_paths: list, tda_scaler=None):
        self.paths      = cache_paths
        self.tda_scaler = tda_scaler

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)
        if self.tda_scaler is not None:
            tda_np = data.tda.numpy().reshape(1, -1)
            # NaN/Inf'i scaler'a vermeden önce temizle
            tda_np = np.nan_to_num(tda_np, nan=0.0, posinf=0.0, neginf=0.0)
            transformed = self.tda_scaler.transform(tda_np)[0]
            # Scaler çıktısını da temizle ve clip et
            transformed = np.nan_to_num(transformed, nan=0.0, posinf=5.0, neginf=-5.0)
            transformed = np.clip(transformed, -5.0, 5.0)
            data.tda = torch.tensor(transformed, dtype=torch.float)
        # Node/edge tensörlerini de temizle
        data.x         = torch.nan_to_num(data.x,         nan=0.0, posinf=3.0, neginf=-3.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
        return data

# ─────────────────────────────────────────────────────────────
# 6. MODEL
# ─────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, out_dim, n_layers=4):
        super().__init__()
        self.input_proj  = nn.Linear(node_dim, hidden)
        # batch_norm=False: eval modunda running stats henuz warm olmadan NaN uretir
        # LayerNorm ile stabilite saglariz
        self.convs       = nn.ModuleList([
            CGConv(hidden, dim=edge_dim, batch_norm=False) for _ in range(n_layers)
        ])
        self.norms       = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden, out_dim)
        self.act         = nn.SiLU()
        # Xavier init
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), nn.SiLU(),
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim),
        )
    def forward(self, z): return self.net(z)


class CPPNHead(nn.Module):
    """ŷ = Σₖ gₖ(z) · Pₖ(z)  — smooth polinom mixture head"""
    def __init__(self, in_dim, K=4):
        super().__init__()
        self.K    = K
        self.gate = nn.Sequential(
            nn.Linear(in_dim, K * 4), nn.SiLU(), nn.Linear(K * 4, K)
        )
        self.poly_lin = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(K)])
        self.poly_q1  = nn.ModuleList([nn.Linear(in_dim, in_dim // 2) for _ in range(K)])
        self.poly_q2  = nn.ModuleList([nn.Linear(in_dim // 2, 1) for _ in range(K)])

    def forward(self, z):
        gates = F.softmax(self.gate(z), dim=-1)
        preds = torch.cat([
            self.poly_lin[k](z) + self.poly_q2[k](F.silu(self.poly_q1[k](z)))
            for k in range(self.K)
        ], dim=-1)
        return (gates * preds).sum(dim=-1)

    def hessian_norm(self, z):
        # float32'de hesapla — AMP float16 ile NaN patlamasi onlenir
        z = z.detach().float().requires_grad_(True)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out = self.forward(z)
        grad = torch.autograd.grad(out.sum(), z, create_graph=False)[0]
        return grad.pow(2).sum(dim=-1).mean()


class CBMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn      = GNNEncoder(NODE_FEAT_DIM, EDGE_FEAT_DIM, GNN_HIDDEN, GNN_OUT)
        self.tda_proj = TDAProjector(TDA_DIM, TDA_PROJ_DIM)
        self.norm     = nn.LayerNorm(FUSION_DIM)
        self.head     = CPPNHead(FUSION_DIM, K=CPPN_K)

    def forward(self, batch):
        tda   = batch.tda.view(batch.num_graphs, -1)
        # NaN guard on inputs
        tda   = torch.nan_to_num(tda,   nan=0.0, posinf=3.0, neginf=-3.0)
        x_in  = torch.nan_to_num(batch.x, nan=0.0, posinf=3.0, neginf=-3.0)
        ea_in = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
        z_gnn = self.gnn(x_in, batch.edge_index, ea_in, batch.batch)
        z_tda = self.tda_proj(tda)
        z_gnn = torch.nan_to_num(z_gnn, nan=0.0)
        z_tda = torch.nan_to_num(z_tda, nan=0.0)
        z     = self.norm(torch.cat([z_gnn, z_tda], dim=-1))
        return self.head(z), z

# ─────────────────────────────────────────────────────────────
# 7. TRAIN / EVAL
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, lambda_reg, scaler):
    model.train()
    total = 0.0
    skipped = 0
    use_amp = device.type == "cuda"
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        cbm   = batch.cbm.view(-1)

        # Label sanity: [-10, 10] eV dışı değerleri filtrele
        label_ok = (cbm >= -10.0) & (cbm <= 10.0) & torch.isfinite(cbm)
        if not label_ok.all():
            # Batch'teki sadece geçerli örnekleri al — basit: tüm batch'i atla
            if label_ok.sum() == 0:
                skipped += 1
                continue
            cbm = cbm[label_ok]

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            pred, z = model(batch)

        # Sadece geçerli label'lara karşılık gelen tahminleri al
        if not label_ok.all():
            pred = pred[label_ok]

        if not torch.isfinite(pred).all():
            optimizer.zero_grad()
            skipped += 1
            continue

        mse = F.mse_loss(pred.float(), cbm.float())

        # hessian_norm autocast DISINDA float32 olarak hesaplanir
        hess = model.head.hessian_norm(z)
        if not torch.isfinite(hess):
            hess = torch.tensor(0.0, device=device)

        loss = mse + lambda_reg * hess

        # Loss patlaması — büyük değerlerde NaN zincirini kır
        if not torch.isfinite(loss) or loss.item() > 1e4:
            optimizer.zero_grad()
            skipped += 1
            continue

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()

    if skipped > 0:
        print(f"  [WARN] {skipped} batch NaN nedeniyle atlandi")
    return total / max(len(loader) - skipped, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    # model.eval() yerine: dropout kapali, BN train modunda (running stats warm olmayabilir)
    model.eval()
    preds, targets, fnames = [], [], []
    use_amp = device.type == "cuda"
    nan_batches = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            pred, _ = model(batch)
        pred = pred.float()
        p_np = pred.cpu().numpy()
        t_np = batch.cbm.view(-1).cpu().numpy()
        # NaN debug
        n_nan = (~np.isfinite(p_np)).sum()
        if n_nan > 0:
            nan_batches += 1
        # NaN filtre
        mask = np.isfinite(p_np) & np.isfinite(t_np)
        if mask.sum() == 0:
            continue
        preds.extend(p_np[mask].tolist())
        targets.extend(t_np[mask].tolist())
        if hasattr(batch, "filename"):
            fn = batch.filename
            fnames.extend(fn if isinstance(fn, list) else [fn])
    if len(preds) == 0:
        print(f"[WARN] Tum tahminler NaN! ({nan_batches} batch etkilendi)")
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
    parser.add_argument("--cache_dir",   default="./cache",
                        help="Graph+TDA cache klasörü (disk)")
    parser.add_argument("--output",      default="predictions.csv")
    parser.add_argument("--max_samples", type=int, default=15000)
    parser.add_argument("--epochs",      type=int, default=200)
    parser.add_argument("--batch",       type=int, default=256,
                        help="8GB VRAM için 256 önerilir")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lambda_reg",  type=float, default=1e-4)
    parser.add_argument("--test_size",   type=float, default=0.15)
    parser.add_argument("--val_size",    type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--skip_cache",  action="store_true",
                        help="Cache build adımını atla (zaten varsa)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device  : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU     : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] VRAM    : {vram:.1f} GB")
        if vram < 10:
            print(f"[INFO] 8GB VRAM algılandı → batch={args.batch} (gerekirse --batch 32 ile düşür)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # CSV
    print(f"\n[INFO] CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    assert "filename" in df.columns and "cbm" in df.columns, \
        "CSV'de 'filename' ve 'cbm' sütunları olmalı"
    df["filename"] = df["filename"].astype(str)
    print(f"[INFO] Toplam etiket: {len(df)}")

    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"[INFO] Örneklendi  : {len(df)}")

    cif_dir   = Path(args.cif_dir)
    cache_dir = Path(args.cache_dir)

    # Cache build
    # ÖNEMLI: edge_attr formatı değişti (v2: normalize edilmiş), eski cache silinmeli!
    cache_version_file = cache_dir / "cache_version.txt"
    CACHE_VERSION = "v2_normalized_edges"
    cache_stale = False
    if cache_dir.exists() and cache_version_file.exists():
        existing_ver = cache_version_file.read_text().strip()
        if existing_ver != CACHE_VERSION:
            print(f"[WARN] Cache versiyonu eski ({existing_ver} != {CACHE_VERSION})!")
            print("[WARN] Eski cache siliniyor ve yeniden oluşturuluyor...")
            import shutil
            for f in cache_dir.glob("*.pt"):
                f.unlink()
            cache_stale = True
    elif cache_dir.exists() and list(cache_dir.glob("*.pt")):
        # Versiyon dosyası yok ama .pt var = eski format
        print("[WARN] Cache versiyon bilgisi yok — eski format varsayildi, sifirlanıyor...")
        for f in cache_dir.glob("*.pt"):
            f.unlink()
        cache_stale = True

    if not args.skip_cache or cache_stale:
        print(f"\n[INFO] Cache build → {cache_dir}")
        print("[WARN] İlk seferinde uzun sürebilir (TDA hesaplama).")
        print("[INFO] Kesintisiz devam eder — yarıda bıraksan bile kaldığı yerden devam eder.\n")
        build_cache(df, cif_dir, cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_version_file.write_text(CACHE_VERSION)
    else:
        print("[INFO] Cache build atlandı.")

    # Geçerli cache path'leri topla
    valid_paths  = []
    valid_fnames = []
    for _, row in df.iterrows():
        cp = get_cache_path(cache_dir, str(row["filename"]))
        if cp.exists():
            valid_paths.append(cp)
            valid_fnames.append(str(row["filename"]))

    print(f"\n[INFO] Geçerli cache: {len(valid_paths)} / {len(df)}")
    if len(valid_paths) < 50:
        print("[ERROR] Yeterli cache yok.")
        return

    # Split — önce yap, scaler train split üzerinde fit edilecek
    idx = list(range(len(valid_paths)))
    tr_val, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    tr, va     = train_test_split(tr_val,
                                   test_size=args.val_size / (1 - args.test_size),
                                   random_state=args.seed)

    # TDA Scaler — train split üzerinde fit et (data leakage olmaz), max 5000 örnek
    scaler_path = cache_dir / "tda_scaler.pkl"
    if scaler_path.exists():
        tda_scaler = joblib.load(scaler_path)
        print("[INFO] TDA scaler yüklendi.")
    else:
        scaler_sample_paths = [valid_paths[i] for i in tr[:5000]]
        print(f"[INFO] TDA scaler fit ediliyor ({len(scaler_sample_paths)} train örnek)...")
        sample_tda = []
        for p in scaler_sample_paths:
            try:
                t = torch.load(p, weights_only=False).tda.numpy()
                t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                sample_tda.append(t)
            except Exception:
                continue
        tda_arr = np.array(sample_tda)
        tda_scaler = StandardScaler()
        tda_scaler.fit(tda_arr)
        # Sıfır varyans sütunlarda NaN üretir — std'yi 1'e zorla
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
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader = make_loader(train_ds, True)
    val_loader   = make_loader(val_ds,   False)
    test_loader  = make_loader(test_ds,  False)
    all_loader   = make_loader(all_ds,   False)

    # Model
    model    = CBMPredictor().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parametre: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Warmup (ilk 10 epoch) + Cosine decay
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler (GPU kullanimi ~%80-90'a cikar)
    amp_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # torch.compile PyG global_mean_pool ile uyumsuz - devre disi
    print("[INFO] torch.compile: devre disi (PyG uyumsuzlugu), AMP aktif")

    # Training loop
    log_rows      = []
    best_val_mae  = float("inf")
    best_pt       = "model_best.pt"
    patience      = 30
    no_improve    = 0

    print(f"\n[INFO] Eğitim: {args.epochs} epoch | batch={args.batch} | early_stop patience={patience}\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss              = train_epoch(model, train_loader, optimizer, device, args.lambda_reg, amp_scaler)
        _, _, v_mae, v_r2, _ = evaluate(model, val_loader, device)
        scheduler.step()

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
            print(f"\n[INFO] Early stopping — {patience} epoch iyileşme yok. "
                  f"En iyi Val MAE: {best_val_mae:.4f} eV")
            break

    # Test
    model.load_state_dict(torch.load(best_pt, map_location=device))
    _, _, t_mae, t_r2, _ = evaluate(model, test_loader, device)
    print(f"\n[RESULT] Test MAE : {t_mae:.4f} eV")
    print(f"[RESULT] Test R²  : {t_r2:.4f}")

    # Tüm tahminler
    all_p, all_t, _, _, _ = evaluate(model, all_loader, device)
    sm = {i: "train" for i in tr}
    sm.update({i: "val" for i in va})
    sm.update({i: "test" for i in te})

    pd.DataFrame({
        "filename": valid_fnames,
        "cbm_true": all_t,
        "cbm_pred": all_p,
        "error":    np.abs(all_p - all_t),
        "split":    [sm.get(i, "") for i in range(len(valid_paths))],
    }).to_csv(args.output, index=False)

    pd.DataFrame(log_rows).to_csv("training_log.csv", index=False)
    torch.save(model.state_dict(), "model.pt")

    print("\n" + "="*50)
    print(f"  Toplam örnek   : {len(valid_paths)}")
    print(f"  Test MAE       : {t_mae:.4f} eV")
    print(f"  Test R²        : {t_r2:.4f}")
    print(f"  Best Val MAE   : {best_val_mae:.4f} eV")
    print(f"  Tahmin dosyası : {args.output}")
    print("="*50)


if __name__ == "__main__":
    main()
