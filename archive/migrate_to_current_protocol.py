"""
Migration helper: bringing a monkey already mid-protocol under ANY previous
version of this training script forward to the CURRENT 8-phase protocol.

CURRENT PROTOCOL SUMMARY (what this script migrates subjects INTO):
  Phase 1-4:  unchanged from previous versions
  Phase 5:    Cue 2 intro    (single substage)
  Phase 6:    Cue 3 intro    (single substage)
  Phase 7:    Cue 4 intro — LCHR star, 40s timeout, 2 sessions, >=70% correct
  Phase 8:    Full task — all 4 cues, independent per-cue rolling windows,
              20s timeout. HCHR>=65%, HCLR>=35%, LCLR>=75%, LCHR>=80%

KEY CHANGES FROM PREVIOUS VERSIONS:
  • HCLR shape: pentagon (old) → scalloped flower PNG (new). No data impact.
  • LCLR shape: triangle (same). No change.
  • LCHR: brand-new 4th cue (star PNG, 1 touch, 3 pellets). New Phase 7.
  • Old Phase 7 (full 3-cue task) → renumbered to Phase 8 and gains LCHR.
  • Cue order now supports 4 positions: HCHR always first; positions 2-4 configurable.
  • LCLR Phase 8 threshold: 80% (old) → 75% (new).

HOW THIS SCRIPT DECIDES WHAT TO DO
-----------------------------------
The migration is driven by WHICH CUES the subject has actually been exposed to
(read from the 'Trial Type' column) — NOT by the last phase number alone. This
matters because different protocol versions numbered phases differently, so two
subjects can both show "last row = Phase 6" yet have completely different cue
histories. Deciding from actual cue exposure is the only reliable approach.

The script:
  1. Reads every 'Trial Type' value to find which of HCHR/HCLR/LCLR/LCHR the
     subject has seen, and in what order they were first introduced.
  2. Builds the recommended CUE_ORDER: HCHR first, then already-learned cues in
     their introduction order, then not-yet-seen cues (LCHR last by default).
  3. Identifies the next cue to introduce (first not-yet-seen cue) and maps it to
     the correct current-protocol phase:
        position 2 → Phase 5,  position 3 → Phase 6,  position 4 → Phase 7.
     (If all four cues have been seen → Phase 8, the full task.)
  4. Determines whether a marker row is needed:
        • If the subject's log was made by an OLDER protocol (Phase 5 has more
          than one substage), OR
        • the destination phase already has rows in the log (old rows that mean
          something different under the current code), OR
        • the last row is already at/after the destination phase,
      then a marker row is appended (with --apply) to steer resume into a CLEAN
      new session of the destination phase, using a high session number so old
      and new rows never blend in the pass check.
        • Otherwise, no row changes are needed and the subject resumes naturally;
          only the CUE_ORDER_OVERRIDES entry must be set.

In ALL cases, existing rows are preserved untouched — the script only ever
APPENDS a single marker row, never edits or deletes history.

USAGE
-----
  python migrate_to_current_protocol.py /path/to/Monkey_training_log.csv
      Diagnostic only — backs up, prints the recommendation.

  python migrate_to_current_protocol.py /path/to/Monkey_training_log.csv --apply
      Appends the marker row (only if one is needed).

EXAMPLE — a monkey (e.g. Vino) who trained under the ORIGINAL 2-cue protocol and
only ever saw HCHR + LCLR (never HCLR or LCHR):
  → recommended order: ["HCHR", "LCLR", "HCLR", "LCHR"]
  → next cue = HCLR → destination = Phase 6 (Cue 3 intro = HCLR)
  → marker needed (old two-substage Phase 5 present) → run with --apply
  → then set CUE_ORDER_OVERRIDES["Vino"] = ["HCHR", "LCLR", "HCLR", "LCHR"]

AFTER RUNNING
-------------
Always set CUE_ORDER_OVERRIDES in the training script as the script prints:

  # New monkey (no prior history) — no entry needed, default is:
  DEFAULT_CUE_ORDER = ["HCHR", "HCLR", "LCLR", "LCHR"]

  # Monkey who learned HCHR then LCLR under the old 2-cue protocol:
  CUE_ORDER_OVERRIDES = {
      "MonkeyName": ["HCHR", "LCLR", "HCLR", "LCHR"],
  }

  # Monkey who learned HCHR, HCLR, LCLR (standard old order):
  CUE_ORDER_OVERRIDES = {
      "MonkeyName": ["HCHR", "HCLR", "LCLR", "LCHR"],   # same as default — no entry needed
  }
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_PHASE8_TRIALS_PER_SESSION = 100   # must match Phase 8 config in training script


def backup_csv(csv_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = csv_path.with_name(f"{csv_path.stem}_BACKUP_{ts}{csv_path.suffix}")
    shutil.copy2(csv_path, backup)
    return backup


# Trials per session for each *current* phase, used to build a "session complete"
# marker so get_resume_state advances cleanly into the target phase.
CURRENT_PHASE_TRIALS = {5: 50, 6: 50, 7: 50, 8: 100}

# Cue types that count as "real" cue exposure (PLAIN is not a discriminative cue).
REAL_CUES = ["HCHR", "HCLR", "LCLR", "LCHR"]


def diagnose(df: pd.DataFrame):
    """
    Decide what to do based on WHICH CUES the subject has actually been exposed
    to (from the Trial Type column) — not merely on the last phase number.

    This is the key fix over the previous version: two subjects can both have a
    'last row = Phase 6' but have completely different cue histories depending on
    which protocol version they trained under. The correct migration depends on
    what they've actually seen, so we inspect that directly.
    """
    if df.empty:
        raise ValueError("Log file is empty.")

    last       = df.iloc[-1]
    last_phase = int(last["Phase #"])
    last_sub   = int(last["Substage #"])
    last_sess  = int(last["Session #"])
    last_trial = int(last["Trial #"])

    counts = df.groupby(["Phase #", "Substage #"]).size().to_dict()

    # Which real cues has this subject ever seen?
    seen = [c for c in REAL_CUES if (df["Trial Type"] == c).any()]
    seen_set = set(seen)

    # Old-protocol structural signature: Phase 5 with more than one substage.
    # Current protocol has single-substage Phase 5/6/7, so >1 Phase-5 substage
    # means these rows were generated by an older version.
    ph5_subs = sorted(df[df["Phase #"] == 5]["Substage #"].unique().tolist())
    old_structure = len(ph5_subs) > 1

    print("=== Log summary ===")
    print(f"Last row: Phase {last_phase}, Substage {last_sub}, "
          f"Session {last_sess}, Trial {last_trial}")
    print(f"Total rows: {len(df)}")
    print("Row counts by (Phase, Substage):")
    for (p, s), n in sorted(counts.items()):
        print(f"  Phase {p}, Substage {s}: {n} rows")
    print(f"\nReal cues ever seen by this subject: {seen if seen else '(none)'}")
    if old_structure:
        print(f"NOTE: Phase 5 has multiple substages {ph5_subs} — this log was generated")
        print(f"      by an OLDER protocol version (current Phase 5 is single-substage).")
    print()

    # ----------------------------------------------------------------
    # Decide the correct cue order and next phase from cue exposure.
    # HCHR is always first. The remaining cues are ordered: those the subject
    # has already learned come next (in the order they were introduced), then
    # the not-yet-seen cues, with LCHR last by default.
    # ----------------------------------------------------------------
    # Determine the order in which non-HCHR cues were FIRST seen.
    intro_order = []
    for _, row in df.iterrows():
        tt = row["Trial Type"]
        if tt in ("HCLR", "LCLR", "LCHR") and tt not in intro_order:
            intro_order.append(tt)

    not_seen = [c for c in ["HCLR", "LCLR", "LCHR"] if c not in intro_order]
    # Keep LCHR last among the not-seen cues if present.
    if "LCHR" in not_seen:
        not_seen = [c for c in not_seen if c != "LCHR"] + ["LCHR"]

    recommended_order = ["HCHR"] + intro_order + not_seen

    # The next cue to introduce = first not-yet-seen cue (in recommended order).
    # Its phase = 5 + (index of that cue in positions 2..4).
    # Position 2 -> Phase 5, position 3 -> Phase 6, position 4 -> Phase 7.
    next_cue = not_seen[0] if not_seen else None

    if next_cue is None:
        print("Subject has already seen all four cues (HCHR, HCLR, LCLR, LCHR).")
        print("They belong in the new Phase 8 (full task).")
        print(f"\nRecommended CUE_ORDER: {recommended_order}")
        return "GOTO_PHASE", {
            "target_phase": 8, "recommended_order": recommended_order,
            "next_cue": None, "last_session": last_sess, "old_structure": old_structure,
        }

    position = recommended_order.index(next_cue)      # 1..3
    target_phase = 5 + (position - 1)                 # pos1->P5, pos2->P6, pos3->P7

    print(f"Cues already learned (after HCHR): {intro_order if intro_order else '(none)'}")
    print(f"Next cue to introduce: {next_cue}")
    print(f"→ Destination phase under current protocol: Phase {target_phase} "
          f"(Cue {position+1} intro = {next_cue})")
    print(f"\nRecommended CUE_ORDER for this subject: {recommended_order}")

    # If the subject's log already has rows at the target phase (from the old
    # protocol meaning something different), we must steer past them so old and
    # new rows don't blend in the pass check.
    rows_at_target = int((df["Phase #"] == target_phase).sum())
    needs_marker = old_structure or rows_at_target > 0 or last_phase >= target_phase

    return "GOTO_PHASE", {
        "target_phase": target_phase,
        "recommended_order": recommended_order,
        "next_cue": next_cue,
        "last_session": last_sess,
        "rows_at_target": rows_at_target,
        "needs_marker": needs_marker,
        "old_structure": old_structure,
    }


def apply_phase_reset(csv_path: Path, headers: list, last_row: dict,
                       target_phase: int, recommended_order: list):
    """
    Append a marker row that steers get_resume_state into a clean new session of
    `target_phase`, Substage 0, without touching any existing rows.

    Mechanism: get_resume_state reads the LAST row. If Trial # >= that phase's
    trials_per_session, it treats the session as complete and starts the next
    session fresh (Session +1, Trial 0). We give the marker a session number far
    above any real one (last_session + 1000) so old and new rows stay visually
    and numerically distinct, and so the new sessions form their own clean
    consecutive block for the pass check.

    The marker's Trial Type is MIGRATION_MARKER (ignored by every per-cue metric),
    Correct Trial? = 0, but Current % Correct = 100 so it can never *block* a
    pass check if it were ever scanned.
    """
    tps = CURRENT_PHASE_TRIALS.get(target_phase, 50)
    last_session = int(last_row.get("Session #", 0))
    marker_session = last_session + 1000

    new_row = {h: last_row.get(h, "") for h in headers}
    new_row.update({
        "Phase #":               target_phase,
        "Substage #":            0,
        "Session #":             marker_session,
        "Trial #":               tps,   # >= trials_per_session -> "session complete"
        "Trial Type":            "MIGRATION_MARKER",
        "Cue Touches Required":  "N/A",
        "Cue Touches Completed": "N/A",
        "Cue Interaction Duration (ms)": "N/A",
        "First Cue Touch RT (ms)": "N/A",
        "Correct Trial?":        0,
        "Current % Correct":     100,
        "Pellets Dispensed":     0,
        "Pellet Delivered?":     "N/A",
        "Error Type":            f"migration_marker_goto_phase{target_phase}",
        "Date":                  datetime.today().strftime("%m-%d-%Y"),
    })

    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writerow(new_row)

    print(f"\nAppended marker: Phase {target_phase}, Substage 0, Session {marker_session}, "
          f"Trial {tps} (marked complete).")
    print(f"On next launch, get_resume_state starts Phase {target_phase}, "
          f"Session {marker_session + 1}, Trial 1.")
    print(f"All {len(headers) and 'existing'} rows are preserved as history; only this")
    print(f"marker was appended. The session-number gap ({marker_session}) keeps old and")
    print(f"new rows separate for the pass check.")
    print(f"\nREQUIRED: set this subject's cue order in the training script:")
    print(f'    CUE_ORDER_OVERRIDES = {{')
    print(f'        "<MonkeyName>": {recommended_order},')
    print(f'    }}')
    print(f"Do this BEFORE the next session so the destination phase introduces the")
    print(f"correct cue.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a subject's training log to the current 8-phase protocol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_path", type=Path,
                        help="Subject's *_simplified_training_log.csv")
    parser.add_argument("--apply", action="store_true",
                        help="Append the marker row that steers resume into the "
                             "recommended destination phase. Without this flag, "
                             "diagnostic only.")
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: file not found: {args.csv_path}")
        sys.exit(1)

    backup = backup_csv(args.csv_path)
    print(f"Backup: {backup}\n")

    df = pd.read_csv(args.csv_path)
    headers = list(df.columns)

    try:
        verdict, info = diagnose(df)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print()

    target_phase = info["target_phase"]
    order        = info["recommended_order"]
    needs_marker = info.get("needs_marker", True)

    if not needs_marker:
        # Subject can resume naturally; only the cue order needs setting.
        print("NO ROW CHANGES NEEDED — the subject can resume naturally into the")
        print(f"correct phase (Phase {target_phase}).")
        print("\nREQUIRED before the next session: set this subject's cue order:")
        print(f'    CUE_ORDER_OVERRIDES = {{')
        print(f'        "<MonkeyName>": {order},')
        print(f'    }}')
        return

    # A marker is needed to steer resume past old-protocol rows.
    if not args.apply:
        print("A marker row IS needed for this subject (old-protocol rows present, or")
        print(f"existing rows in the destination phase). Re-run with --apply to append")
        print(f"a marker steering resume into a clean new Phase {target_phase}.")
        print("\nWhat --apply will do:")
        print(f"  • Append ONE marker row (no existing rows touched)")
        print(f"  • Next launch resumes at Phase {target_phase}, a fresh high-numbered session")
        print(f"  • You must also set CUE_ORDER_OVERRIDES:")
        print(f'        "<MonkeyName>": {order},')
        return

    apply_phase_reset(args.csv_path, headers, df.iloc[-1].to_dict(),
                      target_phase=target_phase, recommended_order=order)


if __name__ == "__main__":
    main()
