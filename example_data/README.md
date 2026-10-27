# Example files

**None of this is real data.** Every row in this folder was made up. "ExampleMonkey"
is not an animal, the dates are invented, and the numbers were generated to look
plausible rather than to mean anything.

They are here for one reason: so that anyone reading the code can see what each
results file actually looks like — what the columns are called, what order they come
in, and what sort of values go in them — without needing access to real animal
records.

**Please don't analyze these.** Any pattern you find in them is an accident of how
they were generated.

## What's here

| File | Shows the layout of |
| --- | --- |
| `ExampleMonkey_simplified_training_log.csv` | The training record, one row per trial |
| `ExampleMonkey_session_log.csv` | The training summary, one row per session |
| `ExampleMonkey_collection_trials.csv` | Experimental trials |
| `ExampleMonkey_collection_touches.csv` | Individual taps, for lining up with video |
| `ExampleMonkey_collection_sessions.csv` | Experimental sessions |
| `ExampleMonkey_probe_log.csv` | An effort test |

The column names and their order match what the task program actually writes, so
these are a reliable reference for the file format even though the contents are not
real.

## Real data

Real session and trial files are deliberately kept out of this repository. The
`.gitignore` file in the main folder blocks them by name, so they stay on the lab
computer and are never uploaded.
