"""
net2_data.py -- v2 task construction for the degradation-warning study.

Changes from v1 (after the 18 Oct 2026 review):
  * no backward filling: RSRQ/SINR are forward-filled within a segment only,
    with missing-value indicators; values still missing at a segment start get
    a fixed neutral constant (RSRQ -12 dB, SINR 0 dB);
  * a row with missing RSRP or CQI ENDS the segment (it is not silently
    removed and the observations either side are not joined);
  * scoring eligibility is CAUSAL: a sample is eligible unless the link is
    observably inside a confirmed episode at that instant (this and the
    previous MIN_RUN-1 samples all degraded); retrospective onsets remain the
    ground truth;
  * sample durations are the actual time to the next sample in the segment
    (the last sample of a segment has duration 0);
  * every eligible sample keeps a label status: 1 (onset in (t, t+H]),
    0 (segment observed through t+H with no onset) or censored.

Parameters (sensitivity analyses vary them through Cfg):
  RSRP_THR=-100 dBm, CQI_THR=7, MIN_RUN=2, MIN_CLEAR=3, GAP_S=30 s, W=20.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

CHANNELS = ["RSRP", "RSRQ", "SINR", "CQI"]
NEUTRAL = {"RSRQ": -12.0, "SINR": 0.0}


@dataclass(frozen=True)
class Cfg:
    rsrp_thr: float = -100.0
    cqi_thr: float = 7.0
    min_run: int = 2
    min_clear: int = 3
    gap_s: float = 30.0
    W: int = 20
    event: str = "rc"          # "rc": RSRP < rsrp_thr and CQI < cqi_thr (main definition)
                               # "tput_floor": DL_bitrate < dl_floor (kbit/s)
                               # "tput_drop": DL_bitrate < drop_ratio x median of the previous drop_ref samples
    dl_floor: float = 1000.0
    drop_ratio: float = 0.5
    drop_ref: int = 10

    def tag(self):
        base = f"r{-self.rsrp_thr:g}_c{self.cqi_thr:g}_m{self.min_run}_w{self.W}"
        if self.event == "rc":
            return base
        if self.event == "tput_floor":
            return base + f"_tfloor{self.dl_floor:g}"
        if self.event == "tput_drop":
            return base + f"_tdrop{self.drop_ratio:g}x{self.drop_ref}"
        raise ValueError(self.event)


def load(path, cfg=Cfg()):
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["link_id", "timestamp"], kind="stable").reset_index(drop=True)
    df = df[~df.duplicated(["link_id", "timestamp"])].reset_index(drop=True)
    for c in CHANNELS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # unit-safe seconds
    df["t"] = ((df["timestamp"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy(float)
    valid = df["RSRP"].notna().to_numpy() & df["CQI"].notna().to_numpy()
    t = df["t"].to_numpy()
    link = df["link_id"].to_numpy()
    new_link = np.r_[True, link[1:] != link[:-1]]
    dt = np.r_[np.inf, np.diff(t)]
    prev_invalid = np.r_[True, ~valid[:-1]]
    start = valid & (new_link | (dt > cfg.gap_s) | prev_invalid)
    seg = np.cumsum(start) - 1
    df["seg"] = np.where(valid, seg, -1)
    n_rows_total, n_invalid = len(df), int((~valid).sum())
    df = df[valid].reset_index(drop=True)
    s = df["seg"].to_numpy()
    tt = df["t"].to_numpy()
    same_next = np.r_[s[1:] == s[:-1], False]
    df["dur"] = np.where(same_next, np.r_[np.diff(tt), 0.0], 0.0)        # actual seconds to next sample
    df["dt_prev"] = np.r_[0.0, np.diff(tt)] * np.r_[False, s[1:] == s[:-1]]
    for c in ("RSRQ", "SINR"):
        df[f"{c}_miss"] = df[c].isna().astype(np.float32)
        df[c] = df.groupby("seg")[c].ffill().fillna(NEUTRAL[c])           # forward only
    df.attrs.update(rows_total=n_rows_total, rows_missing_rsrp_cqi=n_invalid)
    return df


def degraded(df, cfg=Cfg()):
    """Per-measurement degradation indicator for the configured event definition.
    The throughput events use only delivered downlink throughput (independent of
    RSRP and CQI); a missing throughput value is not degraded."""
    if cfg.event == "rc":
        return ((df["RSRP"] < cfg.rsrp_thr) & (df["CQI"] < cfg.cqi_thr)).to_numpy()
    if "DL_bitrate" not in df.columns:
        raise ValueError("throughput event needs a DL_bitrate column (prep_cork2.py keeps it)")
    dl = pd.to_numeric(df["DL_bitrate"], errors="coerce")
    if cfg.event == "tput_floor":
        return (dl < cfg.dl_floor).fillna(False).to_numpy()
    if cfg.event == "tput_drop":
        ref = dl.groupby(df["seg"]).transform(
            lambda v: v.shift(1).rolling(cfg.drop_ref, min_periods=cfg.drop_ref).median())
        return ((dl < cfg.drop_ratio * ref) & (ref > 0)).fillna(False).to_numpy()
    raise ValueError(cfg.event)


def episodes(df, cfg=Cfg()):
    """Retrospective ground truth: in_ep (episode membership) and onset. Also
    the CAUSAL state conf_ep: the last min_run samples (up to and including t)
    are all degraded, i.e. an episode is observably under way at t."""
    deg = degraded(df, cfg)
    seg = df["seg"].to_numpy()
    in_ep = np.zeros(len(df), bool); onset = np.zeros(len(df), bool); conf = np.zeros(len(df), bool)
    bounds = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(df)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = deg[a:b]; n = b - a
        run = 0
        for k in range(n):                                   # causal confirmation
            run = run + 1 if c[k] else 0
            conf[a + k] = run >= cfg.min_run
        i, clear = 0, 0                                      # retrospective truth
        while i < n:
            if c[i]:
                j = i
                while j < n and c[j]:
                    j += 1
                if j - i >= cfg.min_run:
                    in_ep[a + i:a + j] = True
                    if clear >= cfg.min_clear:
                        onset[a + i] = True
                    clear = 0
                else:
                    clear += j - i
                i = j
            else:
                clear += 1; i += 1
    df["degraded"], df["in_ep"], df["onset"], df["conf_ep"] = deg, in_ep, onset, conf
    return df


def build(df, H, cfg=Cfg()):
    """Eligible samples (causal) with >= W samples of history in the segment.
    Returns arrays for ALL eligible samples; y = 1 / 0 / -1 (censored)."""
    t = df["t"].to_numpy(); seg = df["seg"].to_numpy(); onset = df["onset"].to_numpy()
    conf = df["conf_ep"].to_numpy()
    feats = df[CHANNELS + ["RSRQ_miss", "SINR_miss"]].to_numpy(np.float32)
    dtp = df["dt_prev"].to_numpy(np.float32)
    bounds = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(df)]
    rows, ys, nxt = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ts = t[a:b]; on_t = ts[onset[a:b]]; end = ts[-1]
        for k in range(cfg.W - 1, b - a):
            i = a + k
            if conf[i]:
                continue
            j = np.searchsorted(on_t, ts[k], side="right")
            nx = on_t[j] if j < len(on_t) else np.inf
            y = 1 if nx <= ts[k] + H else (0 if end >= ts[k] + H else -1)
            rows.append(i); ys.append(y); nxt.append(nx)
    rows = np.asarray(rows, np.int64)
    idx = rows[:, None] - cfg.W + 1 + np.arange(cfg.W)[None, :]
    return dict(rows=rows, y=np.asarray(ys, np.int8), t=t[rows], seg=seg[rows],
                link=df["link_id"].to_numpy()[rows], dur=df["dur"].to_numpy()[rows],
                next_onset=np.asarray(nxt), win=feats[idx], dwin=dtp[idx], cfg=cfg, H=H)


def onsets_table(df, d):
    """All onsets, with whether each is warnable at H (some eligible, uncensored
    sample in [t_o - H, t_o) in the same segment)."""
    o = df.loc[df["onset"], ["t", "seg", "link_id"]].reset_index(drop=True)
    H = d["H"]; ok = d["y"] >= 0
    key_s = d["seg"][ok] * 1e9 + (d["t"][ok] - 1.5e9)
    order = np.argsort(key_s); ks = key_s[order]
    ko = o["seg"].to_numpy() * 1e9 + (o["t"].to_numpy() - 1.5e9)
    lo = np.searchsorted(ks, ko - H, side="left"); hi = np.searchsorted(ks, ko, side="left")
    o["warnable"] = hi > lo
    return o


def engineered(d):
    """Tabular features from the window ending at t (past only)."""
    win, dwin = d["win"], d["dwin"]
    tt = np.cumsum(dwin, axis=1); tt = tt - tt[:, -1:]
    feats = []
    for c in range(4):
        v = win[:, :, c]
        vm = v.mean(1, keepdims=True); tm = tt.mean(1, keepdims=True)
        slope = ((tt - tm) * (v - vm)).sum(1) / np.maximum(((tt - tm) ** 2).sum(1), 1e-6)
        feats += [v[:, -1], v.mean(1), v.std(1), v.min(1), v.max(1), v[:, -1] - v[:, 0], v[:, -1] - v[:, -5], slope]
    cfg = d["cfg"]
    feats += [dwin[:, 1:].mean(1), (win[:, :, 0] < cfg.rsrp_thr).mean(1), (win[:, :, 3] < cfg.cqi_thr).mean(1),
              win[:, :, 4].mean(1), win[:, :, 5].mean(1)]
    F = np.stack(feats, 1).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def summary(df, H_list=(10, 30, 60), cfg=Cfg()):
    """Event and exposure accounting reported in the paper (per H)."""
    out = dict(rows_total=df.attrs.get("rows_total"), rows_missing_rsrp_cqi=df.attrs.get("rows_missing_rsrp_cqi"),
               rows_used=len(df), links=df["link_id"].nunique(), segments=df["seg"].nunique(),
               observed_h=df["dur"].sum() / 3600, degraded_share=df["in_ep"].mean(),
               episodes=int((df["in_ep"] & ~np.r_[False, df["in_ep"].to_numpy()[:-1] & (np.diff(df["seg"].to_numpy()) == 0)]).sum()),
               onsets=int(df["onset"].sum()), links_with_onset=df.loc[df["onset"], "link_id"].nunique())
    rows = []
    for H in H_list:
        d = build(df, H, cfg); o = onsets_table(df, d); lab = d["y"] >= 0
        rows.append(dict(H=H, eligible=len(d["y"]), labelled=int(lab.sum()), censored=int((~lab).sum()),
                         eligible_h=d["dur"].sum() / 3600, exposure_h=d["dur"][lab].sum() / 3600,
                         positive_rate=float(d["y"][lab].mean()), warnable_onsets=int(o["warnable"].sum()),
                         all_onsets=len(o)))
    return out, pd.DataFrame(rows)


def folds(link, k=5, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(link); rng.shuffle(u)
    f = {l: i % k for i, l in enumerate(u)}
    return np.array([f[l] for l in link])
