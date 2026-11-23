"""Steps 3-4 - Preprocessing and feature engineering.

For every directed pair the raw rows are aligned to a 30-minute grid (slot x path matrices).
A *decision* is made at slot t using only information up to and including slot t-1:

  incumbent  = best-RTT path of slot t-1 (the persistence choice)
  candidates = paths probed at slot t, including probes with 100 % loss ("dead" paths:
               no RTT, loss 100 %, always a failure, regret = DEAD_REGRET_MS)
  best(t)    = lowest RTT at t among live candidates
  regret_p   = RTT_p(t) - best(t) >= 0;  fail_p = regret_p > 20 ms or loss_p(t) > 5 %
  (the oracle picks the lowest-RTT candidate with loss <= 5 %, see core.baseline)

Missing measurements are never imputed: windows are masked, and the number of
observations and the time since the last observation are carried as features.
Skipped probes (paths that previously timed out) are encoded as unavailability events.

Outputs data/processed/candidates.parquet (one row per decision x candidate path).
"""
import numpy as np
import pandas as pd

from config import DEAD_REGRET_MS, FAIL_LOSS, FAIL_MS, FAIL_MS_STRICT, PROC, VANTAGE, WINDOWS

JUMP_MS = 20.0  # threshold for an RTT spike / change point between consecutive observations


def make_labels(df, ms=FAIL_MS, loss=FAIL_LOSS):
    """Failure label and graded ranking relevance for a (regret margin, loss) threshold.
    Dead paths (100 % loss) fail under every threshold."""
    fail = (df.regret > ms) | (df.loss_t > loss)
    rel = np.select([fail, df.regret > 10, df.regret > 2], [0, 1, 2], 3)
    return fail, rel


def rolling(a: pd.DataFrame, k, fn, minp=1):
    return getattr(a.rolling(k, min_periods=minp), fn)()


def pair_panel(pg: pd.DataFrame, sg: pd.DataFrame):
    slots = pd.date_range(pg.slot.min(), pg.slot.max(), freq="30min")
    fps = sorted(pg.fp.unique())
    S, P = len(slots), len(fps)
    idx = pd.Index(slots)
    ok = pg[pg.status == "ok"]   # includes 100 %-loss probes (no RTT)

    def mat(col, src=ok):
        return src.pivot_table(index="slot", columns="fp", values=col, aggfunc="first").reindex(index=idx, columns=fps)

    rtt, loss, mdev = mat("rtt_avg"), mat("loss"), mat("rtt_mdev")
    obs = rtt.notna()          # live observation (RTT measured)
    probed = loss.notna()      # probed at all (live or dead)
    sk = pg[pg.status == "skipped"].assign(v=1.0)
    skipped = sk.pivot_table(index="slot", columns="fp", values="v", aggfunc="first").reindex(index=idx, columns=fps).notna()
    # path-set presence (snapshot exists -> listed or not; missing snapshot -> carry forward)
    snap_slots = set(sg.slot.unique())
    listed = sg.dropna(subset=["fp"]).assign(v=1.0)
    pres = listed.pivot_table(index="slot", columns="fp", values="v", aggfunc="first").reindex(index=idx, columns=fps).notna()
    tout = listed[listed.status == "timeout"].pivot_table(index="slot", columns="fp", values="v", aggfunc="first").reindex(index=idx, columns=fps).notna()
    has_snap = pd.Series([s in snap_slots for s in slots], index=idx)
    pres = pres.where(has_snap, np.nan).ffill().fillna(False).astype(bool) | probed  # probed => present

    # ---------- run structure (continuous presence in the path set) ----------
    P_np = pres.to_numpy()
    runlen = np.zeros((S, P)); nruns = np.zeros((S, P)); obs_run = np.zeros((S, P))
    O_np = obs.to_numpy()
    for s in range(S):
        prev_len = runlen[s - 1] if s else np.zeros(P)
        prev_runs = nruns[s - 1] if s else np.zeros(P)
        started = P_np[s] & (prev_len == 0)
        runlen[s] = np.where(P_np[s], prev_len + 1, 0)
        nruns[s] = prev_runs + started
        prev_or = obs_run[s - 1] if s else np.zeros(P)
        obs_run[s] = np.where(P_np[s], np.where(started, 0, prev_or) + O_np[s], 0)

    # ---------- per-path history features, value at row s = window ending at s ----------
    F = {}
    last_rtt = rtt.ffill()
    last_idx = pd.DataFrame(np.where(O_np, np.arange(S)[:, None], np.nan), index=idx, columns=fps).ffill()
    F["since_obs"] = pd.DataFrame(np.arange(S, dtype=float)[:, None].repeat(P, 1), index=idx, columns=fps) - last_idx
    F["last_rtt"] = last_rtt
    F["last_loss"] = loss.ffill()
    F["last_jit"] = mdev.ffill()
    prev_obs_rtt = rtt.shift(1).ffill()            # previous observation before row s
    dif = rtt - prev_obs_rtt                        # change at observed rows
    F["delta1"] = dif.ffill()
    spike = (dif > JUMP_MS).astype(float).where(obs, 0.0)
    cpt = (dif.abs() > JUMP_MS).astype(float).where(obs, 0.0)
    lossev = (loss > 0).astype(float)
    xs = pd.DataFrame(np.arange(S, dtype=float)[:, None].repeat(P, 1), index=idx, columns=fps).where(obs)
    for k in WINDOWS:
        n = rolling(obs.astype(float), k, "sum")
        F[f"nobs_w{k}"] = n
        F[f"rtt_mean_w{k}"] = rolling(rtt, k, "mean")
        F[f"loss_mean_w{k}"] = rolling(loss, k, "mean")
        F[f"jit_mean_w{k}"] = rolling(mdev, k, "mean")
        if k > 1:
            F[f"rtt_std_w{k}"] = rolling(rtt, k, "std", 2)
            F[f"rtt_min_w{k}"] = rolling(rtt, k, "min")
            F[f"rtt_max_w{k}"] = rolling(rtt, k, "max")
            F[f"spikes_w{k}"] = rolling(spike, k, "sum")
            F[f"cpts_w{k}"] = rolling(cpt, k, "sum")
            F[f"lossev_w{k}"] = rolling(lossev, k, "mean")
            if k >= 3:  # least-squares slope over the observed points of the window (ms / interval)
                y0 = rtt.fillna(0); x0 = xs.fillna(0)
                sx, sy = rolling(x0, k, "sum"), rolling(y0, k, "sum")
                sxx, sxy = rolling(x0 * x0, k, "sum"), rolling(x0 * y0, k, "sum")
                den = n * sxx - sx * sx
                F[f"rtt_slope_w{k}"] = ((n * sxy - sx * sy) / den.where(den > 0)).where(n >= 3)
    F["rtt_ema6"] = rtt.ewm(span=6, ignore_na=True).mean().where(last_rtt.notna())
    F["cummean_rtt"] = rtt.expanding().mean()      # baseline-only (epsilon-greedy value estimate)
    F["skipped_w12"] = rolling(skipped.astype(float), 12, "sum")
    F["timeout_now"] = tout.astype(float)
    F["present"] = pres.astype(float)
    F["runage_h"] = pd.DataFrame(runlen * 0.5, index=idx, columns=fps)
    F["n_appear"] = pd.DataFrame(nruns, index=idx, columns=fps)
    F["obs_in_run"] = pd.DataFrame(obs_run, index=idx, columns=fps)

    # ---------- relative features among recently observed paths ----------
    recent = (F["since_obs"] <= 12) & last_rtt.notna()
    lr = last_rtt.where(recent)
    ema = F["rtt_ema6"].where(recent)
    l_np, e_np = lr.to_numpy(), ema.to_numpy()
    gap_l = np.full((S, P), np.nan); gap_e = np.full((S, P), np.nan); nwin = np.zeros((S, P)); rnk = np.full((S, P), np.nan)
    for s in range(S):
        v = l_np[s]; m = ~np.isnan(v)
        if m.sum() >= 2:
            for j in np.where(m)[0]:
                others = np.delete(v, j); others = others[~np.isnan(others)]
                gap_l[s, j] = v[j] - others.min()
                nwin[s, j] = (others <= v[j] + 10).sum()
            r = pd.Series(v[m]).rank().to_numpy()
            rnk[s, m] = (r - 1) / max(m.sum() - 1, 1)
        ve = e_np[s]; me = ~np.isnan(ve)
        if me.sum() >= 2:
            for j in np.where(me)[0]:
                others = np.delete(ve, j); others = others[~np.isnan(others)]
                gap_e[s, j] = ve[j] - others.min()
    F["gap_last"] = pd.DataFrame(gap_l, index=idx, columns=fps)
    F["gap_ema"] = pd.DataFrame(gap_e, index=idx, columns=fps)
    F["n_within10"] = pd.DataFrame(nwin, index=idx, columns=fps)
    F["rank_last"] = pd.DataFrame(rnk, index=idx, columns=fps)

    # pair-level context (row s)
    rt = rtt.to_numpy()
    srt = np.sort(np.where(np.isnan(rt), np.inf, rt), axis=1)
    # pair-level volatility (added in Phase 3 after error analysis): how the previous
    # persistence decision turned out, and how often the best path changed recently
    has = O_np.any(1)
    PR_np, L_np = probed.to_numpy(), loss.to_numpy()
    bidx = np.where(has, np.argmin(np.where(O_np, rt, np.inf), 1), -1)
    prev_reg = np.full(S, np.nan); prev_fail = np.zeros(S); bchg = np.zeros(S)
    for s in range(1, S):
        if has[s] and bidx[s - 1] >= 0:
            b = bidx[s - 1]
            bchg[s] = float(bidx[s] != b)
            if PR_np[s, b]:
                prev_reg[s] = (rt[s, b] - np.nanmin(np.where(O_np[s], rt[s], np.nan))) if O_np[s, b] else DEAD_REGRET_MS
                prev_fail[s] = float(prev_reg[s] > FAIL_MS or L_np[s, b] > FAIL_LOSS)
    ctx = pd.DataFrame({
        "n_probed": PR_np.sum(1), "n_present": P_np.sum(1),
        "margin2": np.where(np.isfinite(srt[:, 1]), srt[:, 1] - srt[:, 0], np.nan) if P > 1 else np.nan,
        "prev_regret": prev_reg, "prev_fail": prev_fail,
    }, index=idx)
    for k in (6, 12):
        ctx[f"bestchg_w{k}"] = pd.Series(bchg, index=idx).rolling(k, min_periods=1).sum()
        ctx[f"fail_w{k}"] = pd.Series(prev_fail, index=idx).rolling(k, min_periods=1).sum()
    return slots, fps, rtt, loss, obs, probed, F, ctx, runlen


def build():
    pings = pd.read_parquet(PROC / "pings.parquet")
    sets = pd.read_parquet(PROC / "pathsets.parquet")
    struct = pings.groupby(["src", "dst", "fp"])[["n_as", "n_isd", "hop_count"]].first()
    out = []
    for (src, dst), pg in pings.groupby(["src", "dst"]):
        sg = sets[(sets.src == src) & (sets.dst == dst)]
        slots, fps, rtt, loss, obs, probed, F, ctx, runlen = pair_panel(pg, sg)
        R, L, O, PR = rtt.to_numpy(), loss.to_numpy(), obs.to_numpy(), probed.to_numpy()
        Fn = {k: v.to_numpy() for k, v in F.items()}
        C = ctx.to_numpy(); ccols = list(ctx.columns)
        nas = struct.loc[(src, dst)].reindex(fps)
        for t in range(1, len(slots)):
            if not O[t - 1].any() or not O[t].any():
                continue
            inc = int(np.nanargmin(np.where(O[t - 1], R[t - 1], np.nan)))
            cand = np.where(PR[t])[0]          # every probed path, dead ones included
            best = np.nanmin(R[t, O[t]])       # regret reference: lowest live RTT (regret >= 0)
            for j in cand:
                r = dict(src=src, dst=dst, slot=slots[t], t=t, fp=fps[j],
                         is_inc=(j == inc), inc_fp=fps[inc], inc_probed=bool(PR[t, inc]),
                         n_cand=len(cand), rtt_t=R[t, j], loss_t=L[t, j], best_t=best, dead=not O[t, j],
                         runage_now_h=runlen[t, j] * 0.5,
                         n_as=nas.n_as.iloc[j], n_isd=nas.n_isd.iloc[j], hop_count=nas.hop_count.iloc[j])
                for k, a in Fn.items():
                    r[k] = a[t - 1, j]
                for ci, c in enumerate(ccols):
                    r["ctx_" + c] = C[t - 1, ci]
                out.append(r)
        print(f"{VANTAGE[src]}->{VANTAGE[dst]}: {len(slots)} slots, {len(fps)} paths")
    df = pd.DataFrame(out)
    df["pair"] = df.src.map(VANTAGE) + "->" + df.dst.map(VANTAGE)
    df["regret"] = (df.rtt_t - df.best_t).where(~df.dead, DEAD_REGRET_MS)
    # graded relevance for learning-to-rank (higher = better next-interval outcome)
    df["fail"], df["rel"] = make_labels(df)
    df["fail50"] = make_labels(df, ms=FAIL_MS_STRICT)[0]
    hr = df.slot.dt.hour + df.slot.dt.minute / 60
    df["hod_sin"], df["hod_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    dw = df.slot.dt.dayofweek
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
    df["decision_id"] = df.groupby(["pair", "t"]).ngroup()
    df.to_parquet(PROC / "candidates.parquet", index=False)
    d = df.groupby("decision_id").first()
    print(f"candidates: {len(df)} rows ({int(df.dead.sum())} dead-path rows), decisions: {len(d)}, "
          f"evaluable (incumbent probed): {d.inc_probed.sum()}")


if __name__ == "__main__":
    build()
