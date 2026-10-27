"""
================================================================================
COST-BENEFIT TASK — ANALYSIS
================================================================================
Handles all three log types produced by the task. Point it at whichever files
exist for a subject; anything left as None is skipped.

  A. TRAINING LOG      {name}_simplified_training_log.csv
       overall summary, phase progression, per-session table, per-cue
       performance, FIVE-CRITERION breaking point, cost evaluation, error and
       quit-point breakdown, bouts, timing, learning curves, integrity checks

  B. PROBE LOG         {name}_probe_log.csv
       per-N completion and abandonment, whether a breaking point was found,
       per-touch motor readout, engagement, recommended override line

  C. COLLECTION FILES  {name}_collection_trials.csv
                       {name}_collection_touches.csv
                       {name}_collection_sessions.csv
       per-session table, 2x2 cost x reward cells, per-condition comparison,
       PLATEAU CHECK, satiety / time-on-task, sequential effects, touch
       kinematics, video-sync reference

USAGE (VS Code, no terminal needed)
  Set the paths below, run the file. Nothing is ever modified — read only.
================================================================================
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ── SETTINGS — EDIT THESE ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SUBJECT_FOLDER = r"/Volumes/yerkes2/TWLab/share/Zabdi/touchscreen_test/Vino"
SUBJECT_NAME   = "Vino"

# Leave as None to auto-build from SUBJECT_FOLDER + SUBJECT_NAME.
# Set explicitly to analyse a file living somewhere else.
TRAINING_LOG    = None      # e.g. r"/path/Yoshi_simplified_training_log.csv"
PROBE_LOG       = None      # e.g. r"/path/Yoshi_probe_log.csv"
COLLECTION_TRIALS   = None
COLLECTION_TOUCHES  = None
COLLECTION_SESSIONS = None

OUTPUT_DIR       = None     # folder for plots + summary CSVs; None to skip saving
MAKE_PLOTS       = True
SHOW_PLOT_WINDOW = True     # False if running headless

# A gap longer than this (seconds) between trials starts a new "bout"
BOUT_GAP_SECONDS = 120

# ── Breaking-point criteria — MUST match the task script ─────────────────────
ABANDON_RATE_THRESHOLD      = 0.35    # C1  (was 0.60 in earlier versions)
ENGAGEMENT_DROP_THRESHOLD   = 0.60    # C2
ENGAGEMENT_FLOOR_FRACTION   = 0.50    # C3
CUMULATIVE_DROP_THRESHOLD   = 0.60    # C4
STALL_SESSION_LIMIT         = 4       # C5
PHASE4_TRIALS_PER_SESSION   = 50
PHASE4_N_VALUES             = [2, 3, 4, 5, 6, 8, 10, 12]
# High-cost N is set AT the breaking point, not one below it.

# ── Plateau rule for data collection ─────────────────────────────────────────
# Pre-specify this BEFORE looking at the data. Using the most recent
# PLATEAU_WINDOW sessions, plateau is reached when for every cue:
#   (a) completion rate does not rise monotonically across the whole window, and
#   (b) mean of the last half is within PLATEAU_TOLERANCE_PTS of the first half.
PLATEAU_WINDOW        = 4
PLATEAU_TOLERANCE_PTS = 10.0

REAL_CUES = ["HCHR", "HCLR", "LCLR", "LCHR"]
CUE_INFO = {
    "HCHR":  {"cost": "High", "reward": "High", "pellets": 3},
    "HCLR":  {"cost": "High", "reward": "Low",  "pellets": 1},
    "LCLR":  {"cost": "Low",  "reward": "Low",  "pellets": 1},
    "LCHR":  {"cost": "Low",  "reward": "High", "pellets": 3},
    "PLAIN": {"cost": "None", "reward": "Low",  "pellets": 1},
}
CUE_COLORS = {"HCHR": "#CCA15D", "HCLR": "#808080", "LCLR": "#FFC7D7",
              "LCHR": "#A785E8", "PLAIN": "#AAAAAA"}

# ══════════════════════════════════════════════════════════════════════════════
# ── helpers ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

W = 88

def hdr(t, ch="="):
    print("\n" + ch * W); print(t); print(ch * W)

def sub(t):
    print(f"\n--- {t} " + "-" * max(0, W - len(t) - 5))

def show(df, max_rows=80):
    if df is None or len(df) == 0:
        print("  (no data)"); return
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 60,
                           "display.width", 220):
        print(df.to_string(index=False))

def pct(a, b):
    return float("nan") if not b else round(100.0 * a / b, 2)

def safe_read_csv(path, label=""):
    """
    Read a CSV, surviving header drift.

    If a log was created before columns were added to the task script, its
    header row is narrower than its newer data rows and pandas raises
    ParserError. Rather than crash the whole analysis, fall back to the stdlib
    csv reader (which has no expectation about row width), pad every row to the
    widest, and say clearly what happened so the file can be repaired.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError as e:
        print(f"\n  !! {label or Path(path).name}: HEADER DRIFT — {e}")
        import csv as _csv
        with open(path, newline="") as f:
            rows = list(_csv.reader(f))
        if not rows:
            return pd.DataFrame()
        hdr = rows[0]
        width = max(len(r) for r in rows)
        if width > len(hdr):
            extra = width - len(hdr)
            hdr = hdr + [f"UNLABELLED_{i+1}" for i in range(extra)]
            print(f"     header has {width-extra} columns, rows carry up to {width}; "
                  f"{extra} surplus column(s) named UNLABELLED_*.")
        data = [(r + [""] * width)[:width] for r in rows[1:]]
        df = pd.DataFrame(data, columns=hdr[:width])
        # coerce numeric-looking columns; errors="ignore" was removed in pandas 3.0
        for c in df.columns:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= max(1, int(0.8 * df[c].astype(str).str.strip().ne("").sum())):
                df[c] = conv
        print("     Recovered, but OLD rows may be misaligned against the new columns.")
        print("     Fix properly with repair_log_headers.py, or just relaunch the task —")
        print("     _ensure_csv now migrates the header automatically on next run.\n")
        return df


def resolve(explicit, suffix):
    if explicit:
        return Path(explicit)
    if SUBJECT_FOLDER and SUBJECT_NAME:
        return Path(SUBJECT_FOLDER) / f"{SUBJECT_NAME}{suffix}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ══ A.  TRAINING LOG ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def load_training(path):
    df = safe_read_csv(path, 'training log')
    df.columns = [c.strip() for c in df.columns]
    for c in ["Phase #", "Substage #", "Session #", "Trial #",
              "Circle Touch Latency (ms)", "Cue Touches Required",
              "Cue Touches Completed", "Cue Interaction Duration (ms)",
              "First Cue Touch RT (ms)", "Correct Trial?", "Current % Correct",
              "Pellets Dispensed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dt"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time Stim On"].astype(str).str.strip(),
        format="%m-%d-%Y %H:%M:%S", errors="coerce")
    df["is_marker"] = df["Trial Type"].astype(str).str.upper().str.contains("MARKER", na=False)
    df["skey"] = (df["Phase #"].astype("Int64").astype(str) + "." +
                  df["Substage #"].astype("Int64").astype(str) + "." +
                  df["Session #"].astype("Int64").astype(str))
    return df


def training_overall(df, path):
    hdr(f"A1. TRAINING — OVERALL  ({path.name})")
    real = df[~df.is_marker]
    print(f"  Rows:        {len(df)}  (trials {len(real)}, markers {int(df.is_marker.sum())})")
    if real["dt"].notna().any():
        a, b = real["dt"].min(), real["dt"].max()
        print(f"  Dates:       {a:%Y-%m-%d} → {b:%Y-%m-%d}  ({(b-a).days} days)")
    print(f"  Phases:      {sorted(real['Phase #'].dropna().unique().astype(int))}")
    print(f"  Sessions:    {real['skey'].nunique()}")
    print(f"  Cues seen:   {[c for c in REAL_CUES if (real['Trial Type']==c).any()]}")
    last = real.iloc[-1]
    print(f"  Position:    Phase {int(last['Phase #'])}, sub {int(last['Substage #'])}, "
          f"session {int(last['Session #'])}, trial {int(last['Trial #'])}")
    print(f"  Lifetime:    {len(real)} trials, {int(real['Correct Trial?'].sum())} correct "
          f"({pct(int(real['Correct Trial?'].sum()), len(real))}%)")


def training_phases(df):
    hdr("A2. PHASE / SUBSTAGE PROGRESSION")
    real = df[~df.is_marker]; rows = []
    for (ph, sb), g in real.groupby(["Phase #", "Substage #"], dropna=True):
        rows.append({"Phase": int(ph), "Sub": int(sb),
                     "Sessions": g["Session #"].nunique(), "Trials": len(g),
                     "Correct": int(g["Correct Trial?"].sum()),
                     "% Correct": pct(int(g["Correct Trial?"].sum()), len(g)),
                     "N req": (int(g["Cue Touches Required"].max())
                               if g["Cue Touches Required"].notna().any() else None),
                     "Cues": ",".join(c for c in REAL_CUES if (g["Trial Type"] == c).any()),
                     "First": g["dt"].min().strftime("%Y-%m-%d") if g["dt"].notna().any() else "",
                     "Last": g["dt"].max().strftime("%Y-%m-%d") if g["dt"].notna().any() else ""})
    out = pd.DataFrame(rows).sort_values(["Phase", "Sub"]); show(out); return out


def training_sessions(df):
    hdr("A3. PER-SESSION SUMMARY")
    real = df[~df.is_marker]; rows = []
    for skey, g in real.groupby("skey", sort=False):
        g = g.sort_values("Trial #"); ph, sb, ss = [int(x) for x in skey.split(".")]
        dur = ((g["dt"].max() - g["dt"].min()).total_seconds() / 60.0
               if g["dt"].notna().sum() > 1 else np.nan)
        r = {"Phase": ph, "Sub": sb, "Sess": ss,
             "Date": g["dt"].min().strftime("%Y-%m-%d") if g["dt"].notna().any() else "",
             "Trials": len(g), "Correct": int(g["Correct Trial?"].sum()),
             "%Corr": pct(int(g["Correct Trial?"].sum()), len(g)),
             "Dur(min)": round(dur, 1) if dur == dur else np.nan,
             "Pellets": int(g["Pellets Dispensed"].sum(skipna=True))}
        for c in REAL_CUES + ["PLAIN"]:
            sg = g[g["Trial Type"] == c]
            if len(sg):
                r[f"{c} n"] = len(sg); r[f"{c} %"] = pct(int(sg["Correct Trial?"].sum()), len(sg))
        rows.append(r)
    out = pd.DataFrame(rows); show(out); return out


def training_cues(df):
    hdr("A4. PER-CUE PERFORMANCE")
    real = df[~df.is_marker]; rows = []
    for c in REAL_CUES + ["PLAIN"]:
        g = real[real["Trial Type"] == c]
        if not len(g):
            continue
        i = CUE_INFO[c]
        rows.append({"Cue": c, "Cost": i["cost"], "Reward": i["reward"], "Pellets": i["pellets"],
                     "Trials": len(g), "Correct": int(g["Correct Trial?"].sum()),
                     "% Complete": pct(int(g["Correct Trial?"].sum()), len(g)),
                     "Med RT": round(g["First Cue Touch RT (ms)"].median(), 1),
                     "Med dur": round(g["Cue Interaction Duration (ms)"].median(), 1),
                     "Med init lat": round(g["Circle Touch Latency (ms)"].median(), 1)})
    show(pd.DataFrame(rows))
    sub("Completion by phase (%)")
    piv = (real[real["Trial Type"].isin(REAL_CUES + ["PLAIN"])]
           .pivot_table(index="Phase #", columns="Trial Type",
                        values="Correct Trial?", aggfunc="mean") * 100).round(1)
    show(piv.reset_index())


# ── the five-criterion breaking point ────────────────────────────────────────
def phase4_stats(df):
    ph4 = df[(df["Phase #"] == 4) & (df["Trial Type"] == "HCHR")].copy()
    if ph4.empty:
        return {}
    out = {}
    for n in PHASE4_N_VALUES:
        g = ph4[ph4["Cue Touches Required"] == n]
        if not len(g):
            continue
        per = g.groupby(["Substage #", "Session #"]).size().tolist()
        out[n] = {"sessions": per, "n_sessions": len(per), "attempted": len(g),
                  "abandoned": int((g["Correct Trial?"] == 0).sum())}
    return out


def breaking_point(df, trials_per_session=PHASE4_TRIALS_PER_SESSION):
    """Mirror of analyze_breaking_point() in the task script."""
    st = phase4_stats(df)
    if not st:
        return {"status": "NO_DATA", "high_cost_n": None, "breaking_point": None,
                "criterion": None, "detail": "No Phase 4 HCHR data.", "per_n": {}}
    ns = sorted(st); floor = ENGAGEMENT_FLOOR_FRACTION * trials_per_session
    prev = None; best = 0.0; current_n = max(ns)
    for n in ns:
        s = st[n]; avg = sum(s["sessions"]) / len(s["sessions"])
        ab = s["abandoned"] / s["attempted"] if s["attempted"] else 0.0
        def fire(crit, msg):
            return {"status": "MEASURED", "high_cost_n": n, "breaking_point": n,
                    "criterion": crit, "per_n": st,
                    "detail": f"Breaking point at N={n} via {crit}. {msg} High-cost N = {n}."}
        if ab > ABANDON_RATE_THRESHOLD:
            return fire("abandonment", f"{ab:.1%} of {s['attempted']} trials abandoned.")
        if prev and prev > 0 and (prev - avg) / prev >= ENGAGEMENT_DROP_THRESHOLD:
            return fire("engagement drop vs previous N",
                        f"{avg:.1f} vs {prev:.1f} trials/session.")
        if n == current_n and avg < floor:
            return fire("absolute engagement floor",
                        f"{avg:.1f} trials/session, floor is {floor:.0f}.")
        if best > 0 and (best - avg) / best >= CUMULATIVE_DROP_THRESHOLD:
            return fire("cumulative drop from best", f"{avg:.1f} vs best {best:.1f}.")
        if n == current_n and s["n_sessions"] >= STALL_SESSION_LIMIT:
            return fire("stall", f"{s['n_sessions']} sessions at this N without passing.")
        prev = avg; best = max(best, avg)
    return {"status": "CEILING_NOT_MEASURED", "high_cost_n": max(ns), "breaking_point": None,
            "criterion": None, "per_n": st,
            "detail": f"No criterion fired across N={ns}. N={max(ns)} is a FLOOR, not an estimate."}


def training_breaking_point(df, subject=""):
    hdr("A5. BREAKING POINT (Phase 4)")
    real = df[~df.is_marker]
    res = breaking_point(real)
    if res["per_n"]:
        rows = []
        prev = None
        for n in sorted(res["per_n"]):
            s = res["per_n"][n]; avg = sum(s["sessions"]) / len(s["sessions"])
            ab = 100 * s["abandoned"] / s["attempted"]
            drop = "—" if prev is None else f"{100*(prev-avg)/prev:+.1f}%"
            rows.append({"N": n, "Sessions": s["n_sessions"], "Attempted": s["attempted"],
                         "Completed": s["attempted"] - s["abandoned"],
                         "Abandon %": round(ab, 1), "Trials/sess": round(avg, 1),
                         "Δ vs prev": drop,
                         "C1 >35%?": "YES" if ab > 100 * ABANDON_RATE_THRESHOLD else "no"})
            prev = avg
        show(pd.DataFrame(rows))
    print()
    print(f"  status      : {res['status']}")
    print(f"  high-cost N : {res['high_cost_n']}")
    if res["criterion"]:
        print(f"  criterion   : {res['criterion']}")
    print(f"  {res['detail']}")
    if res["status"] == "CEILING_NOT_MEASURED":
        print("\n  " + "!" * 62)
        print("  !!  BREAKING POINT NOT MEASURED — value above is a lower bound.")
        print("  !!  Run PROBE mode, then set HIGH_COST_N_OVERRIDES.")
        print("  " + "!" * 62)
    elif res["status"] == "MEASURED":
        print(f'\n  Optional pin:  HIGH_COST_N_OVERRIDES = {{"{subject}": {res["high_cost_n"]}}}')
    return res


def training_cost(df):
    hdr("A6. COST EVALUATION")
    real = df[~df.is_marker]
    sub("Completion vs touch requirement (multi-touch cues)")
    mt = real[real["Trial Type"].isin(["HCHR", "HCLR"])]
    if len(mt):
        g = (mt.groupby("Cue Touches Required")
               .agg(Trials=("Correct Trial?", "size"), Correct=("Correct Trial?", "sum")).reset_index())
        g["% Complete"] = (100 * g.Correct / g.Trials).round(1)
        g["Cue Touches Required"] = g["Cue Touches Required"].astype(int)
        show(g)
    sub("2x2 cost x reward (completion %)")
    cc = real[real["Trial Type"].isin(REAL_CUES)].copy()
    if len(cc):
        cc["Cost"] = cc["Trial Type"].map(lambda c: CUE_INFO[c]["cost"])
        cc["Reward"] = cc["Trial Type"].map(lambda c: CUE_INFO[c]["reward"])
        show((cc.pivot_table(index="Cost", columns="Reward", values="Correct Trial?",
                             aggfunc="mean") * 100).round(1).reset_index())
    sub("Effort effect (matched reward)")
    for lbl, hi, lo in [("Low reward", "HCLR", "LCLR"), ("High reward", "HCHR", "LCHR")]:
        a, b = real[real["Trial Type"] == hi], real[real["Trial Type"] == lo]
        if len(a) and len(b):
            print(f"  {lbl:<12}: {hi} {100*a['Correct Trial?'].mean():5.1f}%  vs  "
                  f"{lo} {100*b['Correct Trial?'].mean():5.1f}%   "
                  f"→ cost = {100*(b['Correct Trial?'].mean()-a['Correct Trial?'].mean()):+.1f} pts")
    sub("Reward effect (matched cost)")
    for lbl, hi, lo in [("High cost", "HCHR", "HCLR"), ("Low cost", "LCHR", "LCLR")]:
        a, b = real[real["Trial Type"] == hi], real[real["Trial Type"] == lo]
        if len(a) and len(b):
            print(f"  {lbl:<12}: {hi}(3p) {100*a['Correct Trial?'].mean():5.1f}%  vs  "
                  f"{lo}(1p) {100*b['Correct Trial?'].mean():5.1f}%   "
                  f"→ reward = {100*(a['Correct Trial?'].mean()-b['Correct Trial?'].mean()):+.1f} pts")


def training_errors(df):
    hdr("A7. ERRORS AND QUIT POINTS")
    real = df[~df.is_marker]
    err = real[real["Error Type"].notna() & (real["Error Type"].astype(str) != "None")]
    if len(err):
        show(err.groupby(["Trial Type", "Error Type"]).size().reset_index(name="Count"))
        sub("Refusal vs abandonment (multi-touch)")
        for cue in ["HCHR", "HCLR"]:
            g = real[real["Trial Type"] == cue]
            if not len(g):
                continue
            s = int((g["Error Type"].astype(str) == f"{cue.lower()}_start_timeout").sum())
            i = int((g["Error Type"].astype(str) == f"{cue.lower()}_incomplete_timeout").sum())
            print(f"  {cue}: completed {int(g['Correct Trial?'].sum())}, "
                  f"never-started {s}, started-but-quit {i}  (of {len(g)})")
    else:
        print("  No errors logged.")
    sub("Quit-point distribution (failed multi-touch trials)")
    print("  0 touches = refusal to engage. 1..N-1 = abandoned mid-sequence.\n")
    for cue in ["HCHR", "HCLR"]:
        g = real[(real["Trial Type"] == cue) & (real["Correct Trial?"] == 0)]
        if not len(g):
            continue
        vc = g["Cue Touches Completed"].value_counts().sort_index()
        print(f"  {cue} — {len(g)} failed")
        for t, c in vc.items():
            if t == t:
                print(f"    after {int(t):>2} touches: {c:>4}  {'#'*int(round(38*c/vc.max()))}")


def training_bouts(df):
    hdr(f"A8. BOUT STRUCTURE (gap > {BOUT_GAP_SECONDS}s = new bout)")
    real = df[~df.is_marker].copy()
    if real["dt"].isna().all():
        print("  No timestamps."); return None
    rows = []
    for skey, g in real.groupby("skey", sort=False):
        g = g.sort_values("Trial #").copy()
        g["gap"] = g["dt"].diff().dt.total_seconds()
        g["bout"] = (g["gap"] > BOUT_GAP_SECONDS).fillna(False).cumsum()
        ph, sb, ss = [int(x) for x in skey.split(".")]
        for b, bg in g.groupby("bout"):
            d = (bg["dt"].max() - bg["dt"].min()).total_seconds() / 60.0
            rows.append({"Phase": ph, "Sub": sb, "Sess": ss, "Bout": int(b) + 1,
                         "Trials": len(bg), "Dur(min)": round(d, 1),
                         "Trials/min": round(len(bg)/d, 2) if d > 0 else np.nan,
                         "%Corr": pct(int(bg["Correct Trial?"].sum()), len(bg))})
    bt = pd.DataFrame(rows); show(bt, 40)
    sub("Per session")
    show(bt.groupby(["Phase", "Sub", "Sess"])
           .agg(Bouts=("Bout", "count"), MeanTrials=("Trials", "mean"),
                Longest=("Trials", "max")).round(1).reset_index(), 40)
    return bt


def training_timing(df):
    hdr("A9. TIMING")
    print("  WHAT EACH MEASURE CAPTURES — the trial runs:")
    print("     circle appears -> [Circle Touch Latency] -> CUE REVEALED")
    print("        -> [First Cue Touch RT] -> touches -> [Cue Interaction Duration]")
    print()
    print("  Circle Touch Latency is measured BEFORE the cue is revealed, so it")
    print("  cannot differ by trial type — the circle looks identical on every")
    print("  trial. Break it down by CONDITION or session, never by cue.")
    print("  First Cue Touch RT is the decision-relevant one: measured after the")
    print("  animal can see the cost and reward. PLAIN has no composite to touch,")
    print("  so it has no cue RT by design.")
    real = df[~df.is_marker]
    for col, lab in [("Circle Touch Latency (ms)", "Circle touch latency — PRE-CUE (do not compare across cues)"),
                     ("First Cue Touch RT (ms)", "First cue touch RT — POST-CUE (decision-relevant)"),
                     ("Cue Interaction Duration (ms)", "Cue interaction duration")]:
        sub(f"{lab}")
        show((real[real["Trial Type"].isin(REAL_CUES + ["PLAIN"])]
              .groupby("Trial Type")[col]
              .agg(n="count", median="median", q25=lambda s: s.quantile(.25),
                   q75=lambda s: s.quantile(.75)).round(1).reset_index()))
    sub("Per-touch speed on COMPLETED multi-touch trials (motor readout)")
    mt = real[(real["Trial Type"].isin(["HCHR", "HCLR"])) & (real["Correct Trial?"] == 1)].copy()
    if len(mt):
        mt["ms_per_touch"] = mt["Cue Interaction Duration (ms)"] / mt["Cue Touches Completed"]
        show(mt.groupby(["Phase #", "Trial Type"])["ms_per_touch"]
               .agg(n="count", median="median").round(1).reset_index())
        print("\n  Speed on trials the animal CHOSE to complete — uncontaminated by")
        print("  the decision to engage. This is the cleanest motor measure available.")


def training_learning(df):
    hdr("A10. LEARNING CURVES — first exposure per cue")
    real = df[~df.is_marker]
    for cue in REAL_CUES:
        g = real[real["Trial Type"] == cue]
        if not len(g):
            continue
        f = g.iloc[0]
        sub(f"{cue} — first seen Phase {int(f['Phase #'])}, session {int(f['Session #'])}, "
            f"trial {int(f['Trial #'])}")
        p = (g.groupby(["Phase #", "Session #"])
               .agg(Trials=("Correct Trial?", "size"), Correct=("Correct Trial?", "sum")).reset_index())
        p["% Complete"] = (100 * p.Correct / p.Trials).round(1)
        show(p, 30)
        fs = g[(g["Phase #"] == f["Phase #"]) & (g["Session #"] == f["Session #"])]
        if len(fs) >= 3:
            print("    first session : " + " ".join("O" if v == 1 else "." for v in fs["Correct Trial?"]))
            print("    touches done  : " + " ".join(
                str(int(t)) if t == t else "?" for t in fs["Cue Touches Completed"]))


def training_integrity(df):
    hdr("A11. PELLETS AND INTEGRITY")
    real = df[~df.is_marker]
    d = real[real["Pellets Dispensed"].fillna(0) > 0]
    if len(d):
        tot = int(d["Pellets Dispensed"].sum())
        yes = int(d.loc[d["Pellet Delivered?"].astype(str) == "Yes", "Pellets Dispensed"].sum())
        print(f"  Commanded {tot}, confirmed {yes} ({pct(yes,tot)}%), "
              f"UNCONFIRMED {tot-yes} ({pct(tot-yes,tot)}%)")
    issues = 0
    mk = df[df.is_marker]
    if len(mk):
        print(f"\n  {len(mk)} migration marker row(s) present (excluded from analysis)"); issues += 1
    for skey, g in real.groupby("skey", sort=False):
        tn = g["Trial #"].dropna().astype(int).tolist()
        miss = sorted(set(range(min(tn), max(tn)+1)) - set(tn)) if tn else []
        if miss:
            print(f"  GAP in {skey}: missing trials {miss[:15]}{'...' if len(miss)>15 else ''}"); issues += 1
        dup = g["Trial #"][g["Trial #"].duplicated()].tolist()
        if dup:
            print(f"  DUPLICATE trials in {skey}: {sorted(set(dup))[:15]}"); issues += 1
    if not issues:
        print("\n  No integrity problems found.")


# ══════════════════════════════════════════════════════════════════════════════
# ══ B.  PROBE LOG ═════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def analyze_probe(path, subject=""):
    hdr(f"B. BREAKING-POINT PROBE  ({path.name})")
    d = safe_read_csv(path, 'probe log')
    d.columns = [c.strip() for c in d.columns]
    for c in ["N Level", "Trial #", "Cue Touches Required", "Cue Touches Completed",
              "Cue Interaction Duration (ms)", "First Cue Touch RT (ms)",
              "Circle Touch Latency (ms)", "Correct Trial?", "Pellets Dispensed"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["dt"] = pd.to_datetime(d["Date"].astype(str) + " " + d["Time Stim On"].astype(str),
                             format="%m-%d-%Y %H:%M:%S", errors="coerce")

    print(f"  Sessions: {d['SessionUID'].nunique()}   rows: {len(d)}")
    for uid, g in d.groupby("SessionUID"):
        span = (g["dt"].max() - g["dt"].min())
        print(f"    {uid}  {g['dt'].min():%Y-%m-%d %H:%M} → {g['dt'].max():%H:%M}  ({span})")

    h = d[d["Trial Type"] == "HCHR"]
    if h.empty:
        print("  No HCHR probe trials."); return None

    sub("Per N level")
    rows = []; prev = None
    for n, g in h.groupby("N Level"):
        att = len(g); ok = int(g["Correct Trial?"].sum()); ab = att - ok
        c = g[g["Correct Trial?"] == 1]
        spt = (c["Cue Interaction Duration (ms)"] / c["Cue Touches Completed"]).median() if len(c) else np.nan
        rows.append({"N": int(n), "Attempted": att, "Completed": ok, "Abandoned": ab,
                     "Abandon %": round(100*ab/att, 1),
                     "C1 fires (>35%)": "YES" if ab/att > ABANDON_RATE_THRESHOLD else "no",
                     "Med dur (ms)": round(g["Cue Interaction Duration (ms)"].median(), 0),
                     "Med ms/touch": round(spt, 1) if spt == spt else np.nan,
                     "Med init lat": round(g["Circle Touch Latency (ms)"].median(), 0)})
        prev = n
    tab = pd.DataFrame(rows); show(tab)

    fired = tab[tab["C1 fires (>35%)"] == "YES"]
    print()
    if len(fired):
        n = int(fired.iloc[0]["N"])
        print(f"  BREAKING POINT FOUND at N={n}  (abandonment {fired.iloc[0]['Abandon %']}%)")
        print(f"  High-cost N = {n}\n")
        print(f'  Add to the task script:  HIGH_COST_N_OVERRIDES = {{"{subject}": {n}}}')
    else:
        top = int(tab["N"].max()); mx = tab["Abandon %"].max()
        print("  " + "!" * 62)
        print("  !!  NO BREAKING POINT FOUND.")
        print(f"  !!  Highest abandonment was {mx}% — below the {100*ABANDON_RATE_THRESHOLD:.0f}% bar.")
        print(f"  !!  Highest N tested: {top}.")
        print("  !!  This subject has NOT been measured. Options:")
        print("  !!    (a) run another probe session extending PROBE_N_LEVELS beyond")
        print(f"  !!        {top}, or")
        print("  !!    (b) accept a chosen N and set HIGH_COST_N_OVERRIDES manually,")
        print("  !!        documenting that it is a floor rather than a measurement.")
        print("  " + "!" * 62)

    # was any level cut short?
    sub("Level completeness")
    expected = h.groupby("N Level").size().max()
    for n, g in h.groupby("N Level"):
        flag = "" if len(g) >= expected else "  <-- CUT SHORT (session ended early)"
        print(f"  N={int(n):>2}: {len(g)} HCHR trials{flag}")

    sub("Motor readout — per-touch speed on completed trials")
    c = h[h["Correct Trial?"] == 1].copy()
    if len(c):
        c["ms_per_touch"] = c["Cue Interaction Duration (ms)"] / c["Cue Touches Completed"]
        m = c.groupby("N Level")["ms_per_touch"].agg(n="count", median="median").round(1)
        show(m.reset_index())
        v = m["median"].tolist()
        if len(v) > 1:
            print(f"\n  Change from N={int(m.index[0])} to N={int(m.index[-1])}: "
                  f"{v[0]:.0f} → {v[-1]:.0f} ms/touch  ({100*(v[-1]-v[0])/v[0]:+.0f}%)")
            print("  A flat or modest rise means the animal is not fatiguing —")
            print("  failures (if any) reflect choice, not motor capacity.")

    sub("Quit points on failed trials")
    f = h[h["Correct Trial?"] == 0]
    if len(f):
        vc = f["Cue Touches Completed"].value_counts().sort_index()
        for t, cnt in vc.items():
            print(f"  after {int(t):>2} touches: {cnt}")
    else:
        print("  No failed HCHR trials.")
    return tab, h


# ══════════════════════════════════════════════════════════════════════════════
# ══ C.  DATA COLLECTION ═══════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def load_collection(trials_path):
    d = safe_read_csv(trials_path, 'collection trials')
    d.columns = [c.strip() for c in d.columns]
    for c in ["Trial #", "Cue Touches Required", "Cue Touches Completed",
              "Cue Interaction Duration (ms)", "First Cue Touch RT (ms)",
              "Circle Touch Latency (ms)", "Correct Trial?", "Pellets Dispensed",
              "CumulativeTrials", "CumulativePellets", "MinutesIntoSession",
              "HighCostN", "ParkinsonianScore", "DaysSinceInfusion", "DaysSinceMPTP",
              "Mono_TrialStart", "Mono_CueOnset", "Mono_Outcome"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def collection_overview(d, path):
    hdr(f"C1. DATA COLLECTION — OVERVIEW  ({path.name})")
    print(f"  Trials:     {len(d)}")
    print(f"  Sessions:   {d['SessionUID'].nunique()}")
    print(f"  Conditions: {sorted(d['Condition'].dropna().unique())}")
    print(f"  High-cost N used: {sorted(d['HighCostN'].dropna().unique().astype(int))}")
    sub("Trials per condition")
    show(d.groupby("Condition").agg(Sessions=("SessionUID", "nunique"),
                                     Trials=("Trial #", "size"),
                                     Correct=("Correct Trial?", "sum")).reset_index())


def collection_sessions(d, sessions_path=None):
    hdr("C2. PER-SESSION SUMMARY")
    rows = []
    for uid, g in d.groupby("SessionUID", sort=False):
        r = {"SessionUID": uid, "Date": g["Date"].iloc[0],
             "Condition": g["Condition"].iloc[0],
             "Inf#": g["InfusionNumber"].iloc[0] if "InfusionNumber" in g else "",
             "DaysPost": g["DaysSinceInfusion"].iloc[0] if "DaysSinceInfusion" in g else "",
             "PDscore": g["ParkinsonianScore"].iloc[0] if "ParkinsonianScore" in g else "",
             "N": int(g["HighCostN"].iloc[0]) if g["HighCostN"].notna().any() else "",
             "Trials": len(g),
             "Dur(min)": round(g["MinutesIntoSession"].max(), 1),
             "Pellets": int(g["CumulativePellets"].max())}
        for c in REAL_CUES + ["PLAIN"]:
            sg = g[g["Trial Type"] == c]
            if len(sg):
                r[f"{c} n"] = len(sg)
                r[f"{c} %"] = pct(int(sg["Correct Trial?"].sum()), len(sg))
        rows.append(r)
    out = pd.DataFrame(rows); show(out)
    return out


def collection_cells(d):
    hdr("C3. COST x REWARD CELLS")
    cc = d[d["Trial Type"].isin(REAL_CUES)].copy()
    if not len(cc):
        print("  no cue trials"); return
    cc["Cost"] = cc["Trial Type"].map(lambda c: CUE_INFO[c]["cost"])
    cc["Reward"] = cc["Trial Type"].map(lambda c: CUE_INFO[c]["reward"])
    for cond, g in cc.groupby("Condition"):
        sub(f"{cond}  (n={len(g)} cue trials)")
        piv = (g.pivot_table(index="Cost", columns="Reward",
                             values="Correct Trial?", aggfunc="mean") * 100).round(1)
        show(piv.reset_index())
        cells = {c: 100 * g.loc[g["Trial Type"] == c, "Correct Trial?"].mean() for c in REAL_CUES}
        if all(v == v for v in cells.values()):
            eff_hi = cells["LCHR"] - cells["HCHR"]
            eff_lo = cells["LCLR"] - cells["HCLR"]
            rew_hi = cells["HCHR"] - cells["HCLR"]
            rew_lo = cells["LCHR"] - cells["LCLR"]
            print(f"\n    effort effect  : high-reward {eff_hi:+.1f} pts | low-reward {eff_lo:+.1f} pts")
            print(f"    reward effect  : high-cost {rew_hi:+.1f} pts | low-cost {rew_lo:+.1f} pts")


def collection_by_condition(d):
    hdr("C4. PER-CUE COMPLETION BY CONDITION")
    piv = (d[d["Trial Type"].isin(REAL_CUES + ["PLAIN"])]
           .pivot_table(index="Condition", columns="Trial Type",
                        values="Correct Trial?", aggfunc="mean") * 100).round(1)
    show(piv.reset_index())
    sub("Trial counts (check the design is balanced)")
    show(d[d["Trial Type"].isin(REAL_CUES + ["PLAIN"])]
         .pivot_table(index="Condition", columns="Trial Type",
                      values="Trial #", aggfunc="size").reset_index())


def plateau_check(sess_tab):
    hdr("C5. PLATEAU CHECK")
    print(f"  Rule (pre-specified): using the last {PLATEAU_WINDOW} sessions, plateau is")
    print(f"  reached when for EVERY cue (a) completion does not rise monotonically")
    print(f"  across the window, and (b) the last-half mean is within")
    print(f"  {PLATEAU_TOLERANCE_PTS:.0f} points of the first-half mean.\n")

    if len(sess_tab) < PLATEAU_WINDOW:
        print(f"  Only {len(sess_tab)} session(s) — need {PLATEAU_WINDOW}. NOT YET ASSESSABLE.")
        return False

    w = sess_tab.tail(PLATEAU_WINDOW)
    half = PLATEAU_WINDOW // 2
    ok_all = True
    for cue in REAL_CUES:
        col = f"{cue} %"
        if col not in w.columns or w[col].isna().any():
            print(f"  {cue}: missing data in window — cannot assess"); ok_all = False; continue
        v = w[col].astype(float).tolist()
        rising = all(v[i] < v[i+1] for i in range(len(v)-1))
        first, last = np.mean(v[:half]), np.mean(v[half:])
        stable = abs(last - first) <= PLATEAU_TOLERANCE_PTS
        ok = (not rising) and stable
        ok_all &= ok
        print(f"  {cue}: {' → '.join(f'{x:.0f}' for x in v)}   "
              f"first-half {first:.1f} vs last-half {last:.1f} (Δ{last-first:+.1f})")
        print(f"        monotonic-rise? {'yes' if rising else 'no':<4} "
              f"stable? {'yes' if stable else 'no':<4} -> {'OK' if ok else 'NOT STABLE'}")
    print()
    if ok_all:
        first_uid = w.iloc[0]["SessionUID"]
        print("  PLATEAU REACHED.")
        print(f"  Baseline session 1 = {first_uid}")
        print("  Earlier sessions are pre-plateau: keep them, label them as an")
        print("  acquisition curve, and exclude them from the baseline condition.")
    else:
        print("  NOT YET — performance is still changing. Keep collecting.")
        print("  Do not start counting baseline sessions until the rule is satisfied.")
    return ok_all


def collection_latency(d):
    hdr("C6b. LATENCY MEASURES")
    print("  Circle Touch Latency is PRE-CUE (animal cannot yet see the trial type).")
    print("  Use it pooled, split by condition — it indexes willingness to engage.")
    sub("Pre-cue circle touch latency by condition (ms, all cues pooled)")
    v = d.copy()
    v["Circle Touch Latency (ms)"] = pd.to_numeric(v["Circle Touch Latency (ms)"], errors="coerce")
    show(v.groupby("Condition")["Circle Touch Latency (ms)"]
          .agg(n="count", median="median", q25=lambda s: s.quantile(.25),
               q75=lambda s: s.quantile(.75)).round(0).reset_index())

    print("\n  First Cue Touch RT is POST-CUE — the animal now knows cost and reward.")
    print("  This is the one that can legitimately differ between cue types.")
    sub("Post-cue RT by cue x condition (ms)")
    v["First Cue Touch RT (ms)"] = pd.to_numeric(v["First Cue Touch RT (ms)"], errors="coerce")
    t = (v[v["Trial Type"].isin(REAL_CUES)]
         .pivot_table(index="Condition", columns="Trial Type",
                      values="First Cue Touch RT (ms)", aggfunc="median").round(0))
    show(t.reset_index())


def collection_satiety(d):
    hdr("C6. SATIETY / TIME ON TASK")
    print("  Completion rate by position within session — the covariate for the GLMM.\n")
    d = d.copy()
    d["quartile"] = d.groupby("SessionUID")["Trial #"].transform(
        lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"))
    piv = (d[d["Trial Type"].isin(REAL_CUES)]
           .pivot_table(index="quartile", columns="Trial Type",
                        values="Correct Trial?", aggfunc="mean") * 100).round(1)
    show(piv.reset_index())
    sub("Trials completed per session (persistence measure)")
    t = d.groupby(["Condition", "SessionUID"]).size().reset_index(name="Trials")
    show(t.groupby("Condition")["Trials"].agg(n="count", median="median",
                                               min="min", max="max").round(1).reset_index())


def collection_sequential(d):
    hdr("C7. SEQUENTIAL EFFECTS")
    print("  Does the previous trial's outcome change the next decision?\n")
    if "PrevTrialOutcome" not in d.columns:
        print("  PrevTrialOutcome not in file."); return
    g = d[d["Trial Type"].isin(REAL_CUES) & d["PrevTrialOutcome"].isin(["correct", "incorrect"])]
    if not len(g):
        print("  no data"); return
    piv = (g.pivot_table(index="Trial Type", columns="PrevTrialOutcome",
                         values="Correct Trial?", aggfunc="mean") * 100).round(1)
    show(piv.reset_index())
    print("\n  A large drop after 'incorrect' on high-cost cues suggests the animal")
    print("  is updating its cost estimate after a failure.")


def collection_touches(touch_path, trials):
    hdr(f"C8. TOUCH KINEMATICS  ({touch_path.name})")
    t = safe_read_csv(touch_path, 'collection touches')
    t.columns = [c.strip() for c in t.columns]
    for c in ["Trial #", "Touch Index", "Mono_Touch", "X", "Y"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    print(f"  Touch rows: {len(t)}   sessions: {t['SessionUID'].nunique()}")
    if not len(t):
        print("  (empty — nothing to analyse)"); return

    t = t.sort_values(["SessionUID", "Trial #", "Touch Index"])
    t["iti_s"] = t.groupby(["SessionUID", "Trial #"])["Mono_Touch"].diff()

    sub("Inter-touch interval by cue (seconds, multi-touch cues)")
    mt = t[t["Trial Type"].isin(["HCHR", "HCLR"]) & t["iti_s"].notna()]
    if len(mt):
        show(mt.groupby("Trial Type")["iti_s"]
               .agg(n="count", median="median", q25=lambda s: s.quantile(.25),
                    q75=lambda s: s.quantile(.75)).round(3).reset_index())

    sub("Progressive slowing within a touch sequence")
    print("  Median interval by touch index — rising values indicate fatigue;")
    print("  flat values indicate the animal is not motor-limited.\n")
    if len(mt):
        show(mt.groupby(["Trial Type", "Touch Index"])["iti_s"]
               .agg(n="count", median="median").round(3).reset_index(), 40)

    sub("Touch scatter (is the animal aiming consistently?)")
    show(t.groupby("Trial Type")[["X", "Y"]].agg(["median", "std"]).round(1).reset_index())

    if "Condition" in trials.columns:
        sub("Inter-touch interval by condition")
        key = trials[["SessionUID", "Condition"]].drop_duplicates()
        m = mt.merge(key, on="SessionUID", how="left")
        if len(m):
            show(m.groupby(["Condition", "Trial Type"])["iti_s"]
                   .agg(n="count", median="median").round(3).reset_index())


def collection_sync(sessions_path):
    hdr(f"C9. VIDEO SYNC REFERENCE  ({sessions_path.name})")
    s = safe_read_csv(sessions_path, 'collection sessions')
    s.columns = [c.strip() for c in s.columns]
    cols = [c for c in ["SessionUID", "Date", "Condition", "Start Wallclock",
                        "Duration (min)", "CameraEnabled", "SyncFlash_Mono",
                        "VideoFilename", "PrimingPellet_Mono",
                        "PrimingPellet_Delivered", "Trials", "% Correct",
                        "Pellets Dispensed", "Pellets Confirmed", "Exit Reason"]
            if c in s.columns]
    show(s[cols])
    if "PrimingPellet_Mono" in s.columns:
        print("\n  PrimingPellet_Mono is seconds from session start to the free pellet.")
        print("  Use it to locate the clean reach-and-retrieve movement in the video.")
    if "SyncFlash_Mono" in s.columns:
        print("  SyncFlash_Mono lists flash onsets (semicolon separated), same clock.")


# ══════════════════════════════════════════════════════════════════════════════
# ── PLOTS ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _new_fig(n_panels, ncol, title, show_window):
    import matplotlib
    if not show_window:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.3 * ncol, 4.6 * nrow), squeeze=False)
    ax = [a for row in axes for a in row]
    for a in ax[n_panels:]:
        a.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return plt, fig, ax


def _panel(fn, axis, name):
    """Draw one panel; a failure marks that panel only, never the whole figure."""
    try:
        fn(axis)
    except Exception as e:
        axis.text(.5, .5, f"{name}\nfailed:\n{e}", ha="center", va="center",
                  transform=axis.transAxes, fontsize=8, color="crimson", wrap=True)
        axis.set_xticks([]); axis.set_yticks([])
        print(f"  (panel '{name}' failed: {e})")


def _boxplot(a, data, labels, colors=None):
    """Version-safe boxplot: matplotlib renamed labels -> tick_labels in 3.9."""
    try:
        bp = a.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    except TypeError:
        bp = a.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    if colors:
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(.75)
    return bp


def _save(plt, fig, outdir, fname, show_window):
    if outdir:
        p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
        f = p / fname
        plt.savefig(f, dpi=140, bbox_inches="tight")
        print(f"  Saved plot: {f}")
    if show_window:
        plt.show()
    return fig


# ── TRAINING FIGURE (original six panels) ────────────────────────────────────
def plot_training(train, outdir=None, show_window=True, subject=""):
    real = train[~train.is_marker]
    cues = [c for c in REAL_CUES + ["PLAIN"] if (real["Trial Type"] == c).any()]
    plt, fig, ax = _new_fig(6, 3, f"Training — {subject}", show_window)

    def p1(a):   # completion by cue
        v = [100 * real.loc[real["Trial Type"] == c, "Correct Trial?"].mean() for c in cues]
        a.bar(cues, v, color=[CUE_COLORS[c] for c in cues], edgecolor="k")
        a.set_ylim(0, 108); a.set_ylabel("% completed"); a.set_title("Completion by cue")
        for k, x in enumerate(v):
            a.text(k, x + 2, f"{x:.1f}", ha="center", fontsize=9)
    _panel(p1, ax[0], "completion by cue")

    def p2(a):   # phase 4 ramp
        p4 = real[(real["Phase #"] == 4) & (real["Trial Type"] == "HCHR")]
        if not len(p4):
            a.text(.5, .5, "no Phase 4 data", ha="center", transform=a.transAxes); return
        g = p4.groupby("Cue Touches Required")["Correct Trial?"].mean() * 100
        a.plot(g.index, g.values, "o-", lw=2, ms=7, color=CUE_COLORS["HCHR"])
        a.axhline(100 - 100 * ABANDON_RATE_THRESHOLD, ls="--", c="r", alpha=.6,
                  label=f"{100*ABANDON_RATE_THRESHOLD:.0f}% abandon bar")
        a.set_xlabel("N touches required"); a.set_ylabel("% completed")
        a.set_title("Phase 4 ramp (breaking-point probe)")
        a.legend(fontsize=8); a.set_ylim(0, 108)
        a.set_xticks(sorted(g.index.astype(int)))
    _panel(p2, ax[1], "phase 4 ramp")

    def p3(a):   # trials per session
        ss = (real.groupby("skey", sort=False)
                  .agg(n=("Trial #", "size"), ph=("Phase #", "first")).reset_index())
        a.plot(range(len(ss)), ss["n"], "-", color="lightgrey", zorder=1)
        for ph, g in ss.groupby("ph"):
            a.scatter(g.index, g["n"], s=32, label=f"P{int(ph)}", zorder=2)
        a.set_xlabel("session (chronological)"); a.set_ylabel("trials completed")
        a.set_title("Trials per session"); a.legend(fontsize=7, ncol=4)
    _panel(p3, ax[2], "trials per session")

    def p4f(a):  # % correct across sessions
        sc = (real.groupby("skey", sort=False)
                  .agg(p=("Correct Trial?", "mean"), ph=("Phase #", "first")).reset_index())
        sc["p"] *= 100
        a.plot(range(len(sc)), sc["p"], "-", color="lightgrey", zorder=1)
        for ph, g in sc.groupby("ph"):
            a.scatter(g.index, g["p"], s=32, label=f"P{int(ph)}", zorder=2)
        a.set_xlabel("session (chronological)"); a.set_ylabel("% correct")
        a.set_title("% correct across sessions")
        a.legend(fontsize=7, ncol=4); a.set_ylim(0, 108)
    _panel(p4f, ax[3], "% correct across sessions")

    def p5(a):   # quit points
        any_data = False
        for i, cue in enumerate(["HCHR", "HCLR"]):
            g = real[(real["Trial Type"] == cue) & (real["Correct Trial?"] == 0)]
            if not len(g):
                continue
            vc = g["Cue Touches Completed"].dropna().value_counts().sort_index()
            if not len(vc):
                continue
            any_data = True
            a.bar(vc.index + (0.2 if i else -0.2), vc.values, width=.4,
                  label=f"{cue} (n={len(g)})", color=CUE_COLORS[cue], edgecolor="k")
        if not any_data:
            a.text(.5, .5, "no failed multi-touch trials", ha="center", va="center",
                   transform=a.transAxes)
        else:
            a.legend(fontsize=8)
        a.set_xlabel("touches completed before quitting")
        a.set_ylabel("failed trials"); a.set_title("Quit points")
    _panel(p5, ax[4], "quit points")

    def p6(a):   # POST-CUE reaction time (the decision-relevant latency)
        # Circle Touch Latency is measured BEFORE the cue is revealed, so it
        # cannot differ by trial type — splitting it by cue would plot noise.
        # First Cue Touch RT is measured AFTER the cue appears, once the animal
        # knows the cost and reward. PLAIN has no composite to touch, so it is
        # correctly absent here.
        data, labs, cols = [], [], []
        for c in REAL_CUES:
            v = pd.to_numeric(real.loc[real["Trial Type"] == c,
                                       "First Cue Touch RT (ms)"], errors="coerce").dropna()
            if len(v) > 4:
                v = v[v < v.quantile(.95)]
            if len(v):
                data.append(v.values); labs.append(c); cols.append(CUE_COLORS[c])
        if not data:
            a.text(.5, .5, "no cue RT data", ha="center", va="center",
                   transform=a.transAxes); return
        _boxplot(a, data, labs, cols)
        a.set_ylabel("first cue touch RT (ms)")
        a.set_title("Post-cue RT — after cost is known (95th pct trimmed)")
    _panel(p6, ax[5], "post-cue RT")

    plt.tight_layout()
    return _save(plt, fig, outdir, f"{subject}_training_plots.png", show_window)


# ── PROBE FIGURE ─────────────────────────────────────────────────────────────
def plot_probe(tab, raw, outdir=None, show_window=True, subject=""):
    plt, fig, ax = _new_fig(6, 3, f"Breaking-point probe — {subject}", show_window)
    Ns = tab["N"].tolist()
    comp = raw[raw["Correct Trial?"] == 1].copy()
    if len(comp):
        comp["ms_per_touch"] = comp["Cue Interaction Duration (ms)"] / comp["Cue Touches Completed"]

    def p1(a):   # abandonment vs N — the decision measure
        a.plot(Ns, tab["Abandon %"], "o-", lw=2, ms=9, color="#B03A2E")
        a.axhline(100 * ABANDON_RATE_THRESHOLD, ls="--", c="r",
                  label=f"{100*ABANDON_RATE_THRESHOLD:.0f}% breaking-point bar")
        for x, y in zip(Ns, tab["Abandon %"]):
            a.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8)
        a.set_xlabel("N touches required"); a.set_ylabel("% abandoned")
        a.set_title("Abandonment vs effort"); a.set_xticks(Ns)
        a.set_ylim(-3, max(45, tab["Abandon %"].max() * 1.4)); a.legend(fontsize=8)
    _panel(p1, ax[0], "abandonment")

    def p2(a):   # completion + trials attempted
        a.bar(Ns, tab["Completed"], width=1.1, color="#5B9E6B",
              edgecolor="k", label="completed")
        a.bar(Ns, tab["Abandoned"], width=1.1, bottom=tab["Completed"],
              color="#D98880", edgecolor="k", label="abandoned")
        a.set_xlabel("N touches required"); a.set_ylabel("HCHR trials")
        a.set_title("Trials attempted per level"); a.set_xticks(Ns); a.legend(fontsize=8)
    _panel(p2, ax[1], "trials per level")

    def p3(a):   # motor readout
        a.plot(Ns, tab["Med ms/touch"], "o-", lw=2, ms=9, color="#2E86C1")
        a.set_xlabel("N touches required"); a.set_ylabel("median ms per touch")
        a.set_title("Motor speed (completed trials only)"); a.set_xticks(Ns)
        v = tab["Med ms/touch"].tolist()
        if len(v) > 1 and v[0]:
            a.annotate(f"{100*(v[-1]-v[0])/v[0]:+.0f}% across range",
                       xy=(.5, .06), xycoords="axes fraction", ha="center", fontsize=9)
    _panel(p3, ax[2], "motor speed")

    def p4(a):   # total time on cue
        a.plot(Ns, tab["Med dur (ms)"] / 1000, "o-", lw=2, ms=9, color="#7D3C98")
        a.set_xlabel("N touches required"); a.set_ylabel("median cue duration (s)")
        a.set_title("Time spent per completed trial"); a.set_xticks(Ns)
    _panel(p4, ax[3], "cue duration")

    def p5(a):   # willingness to start
        data, labs = [], []
        for n in Ns:
            v = raw.loc[raw["N Level"] == n, "Circle Touch Latency (ms)"].dropna()
            if len(v) > 4:
                v = v[v < v.quantile(.95)]
            if len(v):
                data.append(v.values); labs.append(str(int(n)))
        if not data:
            a.text(.5, .5, "no latency data", ha="center", transform=a.transAxes); return
        _boxplot(a, data, labs, ["#F5B041"] * len(data))
        a.set_xlabel("N touches required"); a.set_ylabel("initiation latency (ms)")
        a.set_title("Willingness to start (95th pct trimmed)")
    _panel(p5, ax[4], "initiation latency by N")

    def p6(a):   # where failures happen
        f = raw[raw["Correct Trial?"] == 0]
        if not len(f):
            a.text(.5, .5, "no failed trials", ha="center", va="center",
                   transform=a.transAxes)
            a.set_title("Quit points"); return
        for n in sorted(f["N Level"].unique()):
            g = f[f["N Level"] == n]
            a.scatter(g["Cue Touches Completed"], [n] * len(g), s=70,
                      alpha=.75, edgecolor="k", label=f"N={int(n)}")
        a.set_xlabel("touches completed before quitting")
        a.set_ylabel("N level"); a.set_yticks(Ns)
        a.set_title("Quit points by level"); a.legend(fontsize=8)
    _panel(p6, ax[5], "quit points")

    plt.tight_layout()
    return _save(plt, fig, outdir, f"{subject}_probe_plots.png", show_window)


# ── COLLECTION FIGURE ────────────────────────────────────────────────────────
def plot_collection(d, touches=None, outdir=None, show_window=True, subject=""):
    conds = sorted(d["Condition"].dropna().unique())
    cues = [c for c in REAL_CUES if (d["Trial Type"] == c).any()]
    n_panels = 9 if touches is not None and len(touches) else 8
    plt, fig, ax = _new_fig(n_panels, 3, f"Data collection — {subject}", show_window)

    def p1(a):   # 2x2 cells
        cc = d[d["Trial Type"].isin(REAL_CUES)]
        v = {c: 100 * cc.loc[cc["Trial Type"] == c, "Correct Trial?"].mean() for c in REAL_CUES}
        x = np.arange(2); w = .36
        a.bar(x - w/2, [v["HCHR"], v["LCHR"]], w, label="High reward (3p)",
              color="#4A7BA7", edgecolor="k")
        a.bar(x + w/2, [v["HCLR"], v["LCLR"]], w, label="Low reward (1p)",
              color="#C9A227", edgecolor="k")
        a.set_xticks(x); a.set_xticklabels(["High cost", "Low cost"])
        a.set_ylim(0, 108); a.set_ylabel("% completed")
        a.set_title("Cost x reward cells"); a.legend(fontsize=8)
        for k, pair in enumerate([("HCHR", "HCLR"), ("LCHR", "LCLR")]):
            a.text(k - w/2, v[pair[0]] + 2, f"{v[pair[0]]:.0f}", ha="center", fontsize=8)
            a.text(k + w/2, v[pair[1]] + 2, f"{v[pair[1]]:.0f}", ha="center", fontsize=8)
    _panel(p1, ax[0], "cost x reward")

    def p2(a):   # cue x condition
        w = .8 / max(len(conds), 1)
        for k, cond in enumerate(conds):
            g = d[d["Condition"] == cond]
            v = [100 * g.loc[g["Trial Type"] == c, "Correct Trial?"].mean() for c in cues]
            a.bar(np.arange(len(cues)) + k*w - .4 + w/2, v, width=w,
                  label=cond, edgecolor="k")
        a.set_xticks(range(len(cues))); a.set_xticklabels(cues)
        a.set_ylim(0, 108); a.set_ylabel("% completed")
        a.set_title("Cue x condition"); a.legend(fontsize=8)
    _panel(p2, ax[1], "cue x condition")

    def p3(a):   # per-cue across sessions (plateau trace)
        for c in cues:
            s = d[d["Trial Type"] == c].groupby("SessionUID", sort=False)["Correct Trial?"].mean() * 100
            if len(s):
                a.plot(range(len(s)), s.values, "o-", label=c, color=CUE_COLORS[c], ms=6)
        a.set_ylim(0, 108); a.set_xlabel("collection session")
        a.set_ylabel("% completed"); a.set_title("Per-cue across sessions (plateau trace)")
        a.legend(fontsize=8)
    _panel(p3, ax[2], "per-cue across sessions")

    def p4(a):   # POST-CUE reaction time by cue — decision-relevant
        # See note in plot_training: Circle Touch Latency is pre-cue and cannot
        # vary by trial type. This is First Cue Touch RT, measured after the cue
        # is revealed. PLAIN has no composite, so it is absent by design.
        data, labs, cols = [], [], []
        for c in cues:
            v = pd.to_numeric(d.loc[d["Trial Type"] == c,
                                    "First Cue Touch RT (ms)"], errors="coerce").dropna()
            if len(v) > 4:
                v = v[v < v.quantile(.95)]
            if len(v):
                data.append(v.values); labs.append(c); cols.append(CUE_COLORS[c])
        if not data:
            a.text(.5, .5, "no cue RT data", ha="center", transform=a.transAxes); return
        _boxplot(a, data, labs, cols)
        a.set_ylabel("first cue touch RT (ms)")
        a.set_title("Post-cue RT by cue — after cost is known")
    _panel(p4, ax[3], "post-cue RT")

    def p5(a):   # satiety
        dd = d.copy()
        dd["q"] = dd.groupby("SessionUID")["Trial #"].transform(
            lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"))
        for c in cues:
            g = dd[dd["Trial Type"] == c].groupby("q")["Correct Trial?"].mean() * 100
            if len(g):
                a.plot(range(len(g)), g.values, "o-", label=c, color=CUE_COLORS[c], ms=6)
        a.set_xticks(range(4)); a.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        a.set_ylim(0, 108); a.set_xlabel("quartile within session")
        a.set_ylabel("% completed"); a.set_title("Satiety / time on task"); a.legend(fontsize=8)
    _panel(p5, ax[4], "satiety")

    def p6(a):   # trials per session, chronological with condition colouring
        t = (d.groupby("SessionUID", sort=False)
               .agg(n=("Trial #", "size"), cond=("Condition", "first"),
                    mins=("MinutesIntoSession", "max")).reset_index())
        t["idx"] = range(len(t))
        a.plot(t["idx"], t["n"], "-", color="lightgrey", zorder=1)
        for cond in conds:
            g = t[t["cond"] == cond]
            a.scatter(g["idx"], g["n"], s=60, label=cond, edgecolor="k", zorder=2)
        a.set_xlabel("session (chronological)"); a.set_ylabel("trials completed")
        a.set_title("Persistence: trials per session"); a.legend(fontsize=8)
        a.set_xticks(t["idx"].tolist())
    _panel(p6, ax[5], "trials per session")

    def p7(a):   # quit points
        any_d = False
        for i, cue in enumerate(["HCHR", "HCLR"]):
            g = d[(d["Trial Type"] == cue) & (d["Correct Trial?"] == 0)]
            if not len(g):
                continue
            vc = g["Cue Touches Completed"].dropna().value_counts().sort_index()
            if not len(vc):
                continue
            any_d = True
            a.bar(vc.index + (0.2 if i else -0.2), vc.values, width=.4,
                  label=f"{cue} (n={len(g)})", color=CUE_COLORS[cue], edgecolor="k")
        if not any_d:
            a.text(.5, .5, "no failed multi-touch trials", ha="center", va="center",
                   transform=a.transAxes)
        else:
            a.legend(fontsize=8)
        a.set_xlabel("touches completed before quitting")
        a.set_ylabel("failed trials"); a.set_title("Quit points")
    _panel(p7, ax[6], "quit points")

    def p_precue(a):
        # Circle Touch Latency IS meaningful — just not by cue type. Pooled
        # across cues it indexes willingness to start a trial at all, which is
        # where bradykinesia and apathy would show up. Split by CONDITION.
        data, labs = [], []
        for cond in conds:
            v = pd.to_numeric(d.loc[d["Condition"] == cond,
                                    "Circle Touch Latency (ms)"], errors="coerce").dropna()
            if len(v) > 4:
                v = v[v < v.quantile(.95)]
            if len(v):
                data.append(v.values); labs.append(cond)
        if not data:
            a.text(.5, .5, "no latency data", ha="center", transform=a.transAxes); return
        _boxplot(a, data, labs, ["#8FBF9F"] * len(data))
        a.set_ylabel("circle touch latency (ms)")
        a.set_title("Pre-cue engagement by condition (all cues pooled)")
    _panel(p_precue, ax[7], "pre-cue engagement")

    if n_panels == 9:
        def p8(a):   # progressive slowing within a touch sequence
            t = touches.sort_values(["SessionUID", "Trial #", "Touch Index"]).copy()
            t["iti"] = t.groupby(["SessionUID", "Trial #"])["Mono_Touch"].diff()
            mt = t[t["Trial Type"].isin(["HCHR", "HCLR"]) & t["iti"].notna()]
            for cue in ["HCHR", "HCLR"]:
                g = mt[mt["Trial Type"] == cue].groupby("Touch Index")["iti"].median()
                if len(g):
                    a.plot(g.index, g.values, "o-", label=cue, color=CUE_COLORS[cue], ms=6)
            a.set_xlabel("touch index within sequence")
            a.set_ylabel("median inter-touch interval (s)")
            a.set_title("Progressive slowing (fatigue check)"); a.legend(fontsize=8)
        _panel(p8, ax[8], "progressive slowing")

    plt.tight_layout()
    return _save(plt, fig, outdir, f"{subject}_collection_plots.png", show_window)


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    tp = resolve(TRAINING_LOG, "_simplified_training_log.csv")
    pp = resolve(PROBE_LOG, "_probe_log.csv")
    ct = resolve(COLLECTION_TRIALS, "_collection_trials.csv")
    cu = resolve(COLLECTION_TOUCHES, "_collection_touches.csv")
    cs = resolve(COLLECTION_SESSIONS, "_collection_sessions.csv")

    hdr(f"SUBJECT: {SUBJECT_NAME}", "#")
    for lab, p in [("training", tp), ("probe", pp), ("collection trials", ct),
                   ("collection touches", cu), ("collection sessions", cs)]:
        print(f"  {lab:<20} {'FOUND    ' if p and p.exists() else 'not found'}  {p}")

    train_df = probe_tab = probe_raw = coll_df = None
    sess_tab = None

    if tp and tp.exists():
        train_df = load_training(tp)
        training_overall(train_df, tp)
        prog = training_phases(train_df)
        training_sessions(train_df)
        training_cues(train_df)
        training_breaking_point(train_df, SUBJECT_NAME)
        training_cost(train_df)
        training_errors(train_df)
        training_bouts(train_df)
        training_timing(train_df)
        training_learning(train_df)
        training_integrity(train_df)

    if pp and pp.exists():
        probe_tab, probe_raw = analyze_probe(pp, SUBJECT_NAME)

    if ct and ct.exists():
        coll_df = load_collection(ct)
        collection_overview(coll_df, ct)
        sess_tab = collection_sessions(coll_df)
        collection_cells(coll_df)
        collection_by_condition(coll_df)
        plateau_check(sess_tab)
        collection_latency(coll_df)
        collection_satiety(coll_df)
        collection_sequential(coll_df)
        if cu and cu.exists():
            collection_touches(cu, coll_df)
        if cs and cs.exists():
            collection_sync(cs)

    if MAKE_PLOTS:
        hdr("PLOTS")
        if train_df is not None:
            try:
                plot_training(train_df, OUTPUT_DIR, SHOW_PLOT_WINDOW, SUBJECT_NAME)
            except Exception as e:
                print(f"  training figure failed: {type(e).__name__}: {e}")
        if probe_tab is not None:
            try:
                plot_probe(probe_tab, probe_raw, OUTPUT_DIR, SHOW_PLOT_WINDOW, SUBJECT_NAME)
            except Exception as e:
                print(f"  probe figure failed: {type(e).__name__}: {e}")
        if coll_df is not None:
            try:
                tdf = None
                if cu and cu.exists():
                    tdf = safe_read_csv(cu, 'collection touches')
                    tdf.columns = [c.strip() for c in tdf.columns]
                    for c in ["Trial #", "Touch Index", "Mono_Touch", "X", "Y"]:
                        if c in tdf.columns:
                            tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
                plot_collection(coll_df, tdf, OUTPUT_DIR, SHOW_PLOT_WINDOW, SUBJECT_NAME)
            except Exception as e:
                print(f"  collection figure failed: {type(e).__name__}: {e}")

    if OUTPUT_DIR:
        o = Path(OUTPUT_DIR); o.mkdir(parents=True, exist_ok=True)
        for name, d in [("probe_summary", probe_tab), ("collection_sessions", sess_tab)]:
            if d is not None and len(d):
                f = o / f"{SUBJECT_NAME}_{name}.csv"; d.to_csv(f, index=False)
                print(f"  Saved: {f}")

    hdr("DONE", "#")


if __name__ == "__main__":
    main()
