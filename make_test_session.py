#!/usr/bin/env python3
"""
Generate a complete, self-consistent session of trial rows for testing phase
advancement. Produces rows identical in shape to what the real program writes,
with a correct running 'Current % Correct' on every row.

Usage examples:
  # A 50-trial Phase 5 session that ends at 80% correct, session 2:
  python make_test_session.py --phase 5 --sub 0 --session 2 --trials 50 --pct 80

  # Append directly to a monkey's log:
  python make_test_session.py --phase 5 --sub 0 --session 2 --trials 50 --pct 80 \
      >> MonkeyName/MonkeyName_simplified_training_log.csv

Notes:
  * The pass check reads only the LAST row's 'Current % Correct' per session,
    but this writes a fully consistent running percentage on every row so the
    file also looks realistic and resume logic behaves correctly.
  * Correct trials are spread across the session; the final row lands exactly
    on the requested percentage (rounded to whole correct trials).
"""
import argparse
from datetime import datetime

CUE_DEFAULTS = {          # trial_type: (touches_required, pellets)
    "PLAIN": (0, 1), "HCHR": (6, 3), "HCLR": (6, 1), "LCLR": (1, 1), "LCHR": (1, 3),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--sub", type=int, default=0)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--pct", type=float, required=True, help="target final %% correct")
    ap.add_argument("--type", default="HCHR", choices=list(CUE_DEFAULTS),
                    help="trial type to fill the session with (default HCHR)")
    ap.add_argument("--date", default=datetime.today().strftime("%m-%d-%Y"))
    args = ap.parse_args()

    n = args.trials
    n_correct = round(args.pct / 100.0 * n)
    touches, pellets = CUE_DEFAULTS[args.type]

    # Decide which trials are correct: spread the correct ones evenly.
    # Put incorrect trials at evenly spaced positions.
    n_incorrect = n - n_correct
    incorrect_positions = set()
    if n_incorrect > 0:
        step = n / n_incorrect
        incorrect_positions = {int(i * step) for i in range(n_incorrect)}

    cum_correct = 0
    for i in range(n):
        trial_num = i + 1
        is_correct = 0 if i in incorrect_positions else 1
        cum_correct += is_correct
        pct = cum_correct / trial_num * 100

        if is_correct:
            t_done = touches
            npel = pellets
            deliv = "Yes"
            err = "None"
        else:
            t_done = 0
            npel = 0
            deliv = "N/A"
            err = f"{args.type.lower()}_incomplete_timeout" if args.type in ("HCHR","HCLR") else f"{args.type.lower()}_timeout"

        row = [
            args.date, args.phase, args.sub, args.session, trial_num,
            "09:00:00", "09:00:01", "500.0",
            args.type, touches, t_done, "2000", "600",
            is_correct, f"{pct:.2f}", npel, deliv, err,
        ]
        print(",".join(str(x) for x in row))

if __name__ == "__main__":
    main()
