# Cost-Benefit Touchscreen Task

A script to automatically train and collect data from non-human primates on a
cost-benefit task, which can be personalized for each monkey.

An animal sits at a touchscreen. A shape appears, and the shape tells the animal what
this trial will cost and what it will pay — a few taps or many, a small reward or a
large one. The program presents the trials, counts the taps, runs the pellet
dispenser, decides whether the animal succeeded, and writes everything to
spreadsheets. By comparing how often an animal takes each deal, you can measure how
it weighs effort against reward.

One file does three jobs: it trains a naive animal from scratch through eight stages,
it runs balanced experimental sessions once training is finished, and it measures an
individual animal's effort limit. A second file reads the results afterward and
produces the summaries and figures. It runs on a Raspberry Pi in the testing booth,
or on any Linux computer.

---

## Easy to run

**You operate it by touching the screen.** Starting the program puts it fullscreen
and asks two things as large on-screen buttons: which animal, and which mode.
Anywhere a number is needed, you get a big on-screen keypad. There is no terminal to
work in beyond the one command that starts it, and nothing to edit before a routine
session.

**It remembers where every animal is.** The program reads each animal's own records
to work out which stage they reached and how far into the current session they got,
then carries on from exactly there. Stop halfway through and come back tomorrow and
it picks up mid-session with the right counts. Nothing to write down, nothing to set.

**Each animal gets its own setup.** How much effort counts as "a lot" is worked out
per animal, and the order the shapes are introduced can be set per animal too. Both
are single lines near the top of the file, and an animal with nothing set simply gets
the defaults.

**It advances animals on its own.** During training it checks performance at the end
of every session and moves an animal up a stage once they meet the bar. It tells you
on screen what it decided.

**It never changes modes behind your back.** The menu shows a reminder of where the
animal currently stands, but the choice is always yours, every single session.

**It looks after its own files.** If the program gains a new column in a later
version, it updates your existing spreadsheets to match and saves a dated backup
first. There is no file maintenance for you to do.

---

## What's in here

| File | What it does |
| --- | --- |
| `FULL_Cost_reward_task_training.py` | **The task itself.** This is what runs in the booth. It handles training, data collection, and effort testing. |
| `Analyze_task.py` | **The results reader.** Point it at an animal's files and it produces summaries and figures. It never changes your data. |
| `make_test_session.py` | A small tool for generating fake sessions, so you can check the program behaves correctly without using an animal. |
| `example_data/` | Made-up files that show what each results file looks like. Not real data — see the note in that folder. |
| `archive/` | A tool that has finished its job, kept as a record. Nothing in here needs running. |

---

## The four shapes

Every trial starts the same way: a cyan circle appears on a black screen. When the
animal touches it, a colored shape appears behind the circle. That shape is the cue
for what this trial will cost and what it will pay.

| Shape | Color | Taps needed | Pellets | In plain terms |
| --- | --- | --- | --- | --- |
| Diamond | Gold | Many | 3 | Hard work, big payoff |
| Flower | Gray | Many | 1 | Hard work, small payoff |
| Triangle | Pink | One | 1 | Easy, small payoff |
| Star | Purple | One | 3 | Easy, big payoff |
| No shape | — | One | 1 | A freebie, to check the animal is still engaged |

The two "hard work" shapes always ask for the **same** number of taps, so the only
thing separating them is the size of the reward. That is the point of the design: it
isolates how much the animal cares about reward size when the effort is held
constant.

How many taps counts as "many" is worked out separately for each animal — see
**Effort level** below.

---

## Getting it running

You need a Raspberry Pi (or any computer running Linux), a touchscreen, and an Ontrak
ADU200 pellet dispenser plugged in by USB.

**1. Install the three things the program needs.** Open a terminal and paste:

```
pip install pygame pandas hidapi
```

**2. Put the files somewhere sensible** — a folder called `cost-benefit-task` in your
home directory works fine.

**3. Plug in the pellet dispenser before you start.** The program checks for it as
soon as it opens and will stop with an error if it isn't there.

**4. Start the task:**

```
python FULL_Cost_reward_task_training.py
```

From here on it's all touch. Press **Shift + S** to begin a session, **Esc** at any
point to stop, and **Shift + E** to close the program afterward.

---

## The three modes

**Training (press T).** Takes a naive animal from "touch the big circle" all the way
to the full four-shape task, in eight stages. It keeps track of where each animal is
and picks up exactly where it left off, and moves them up a stage once the
performance bar is met.

**Data collection (press D).** For animals that have finished training. All four
shapes appear equally often and nothing changes between sessions — no stages, no
advancement. This is the mode that produces your experimental data. It asks a few
questions about the session first, then runs until you press Esc. The list of
conditions it offers is a setting you control in the file.

**Effort test (press P).** A standalone check of how much tapping an animal will
tolerate before giving up. Only the gold diamond appears, and the number of taps
climbs until the animal starts refusing. It writes to its own separate file and
doesn't touch the animal's training records, so it's safe to run on an animal that
has already finished training.

---

## Effort level

"Many taps" isn't a fixed number — it's worked out per animal, because what's
demanding for one animal is easy for another.

During training, the program gradually increases the number of taps required and
watches for the point where the animal starts giving up. It uses five signs of having
hit a limit: refusing too many trials, working much more slowly than it did at an
easier level, working very slowly in absolute terms, dropping well below its own best
pace, or getting stuck at the same level for several sessions without improving.

If you'd rather set the number yourself — because you measured it another way, or
because the animal never reached a limit — you can. Near the top of the task file
there's a short list:

```python
HIGH_COST_N_OVERRIDES = {
    "Yoshi": 8,
}
```

Add a line like that and the program stops working it out for that animal and simply
uses your number everywhere: training, data collection, and all.

> **A note on measuring versus choosing.** The program tells you which of these
> happened. If it says `MEASURED`, the animal genuinely reached a limit. If it says
> `CEILING_NOT_MEASURED`, the animal finished the whole range without ever giving up,
> so the number is only a *floor* — the real limit is somewhere above it, and you
> haven't found it yet. The program prints a loud warning in that case, and it's worth
> taking seriously: an effort level the animal never struggles with leaves no room for
> your manipulation to show an effect.

---

## Settings you might want to change

All of these live near the top of the task file as capitalized names. Change the
value, save the file, and it takes effect next time you start the program.

| Setting | What it controls |
| --- | --- |
| `HIGH_COST_N_OVERRIDES` | Sets the number of taps by hand for a particular animal |
| `CUE_ORDER_OVERRIDES` | Changes the order the shapes are introduced during training |
| `CONDITIONS` | The list of session types offered in data collection |
| `RESPONSE_TIMEOUT_FULL_MS` | How long the animal has per tap in the final training stage |
| `COLLECTION_TIMEOUT_MS` | How long per tap during data collection |
| `NORMAL_ITI_MS` / `PENALTY_ITI_MS` | The pause between trials, after a success and after a failure |
| `PELLET_PULSE_MS` | How long the dispenser runs per pellet — adjust if it jams or double-drops |
| `CAMERA_RECORDING_ENABLED` | Turn on if you're filming, so the video can be lined up with the data |
| `INTER_TOUCH_LOCKOUT_ENABLED` | The brief dark gap between taps (explained below) |

**The gap between taps.** On the many-tap trials, the screen goes dark for a random
tenth to a third of a second after each tap, and taps during that gap don't count.
This stops an animal from drumming on the screen and getting credit for it, so the
taps that do count reflect a real decision to keep going. The response clock pauses
during the gap, so the animal is never penalized for it.

---

## Where your data goes

The program makes a folder for each animal, named after them, right next to the
program file. Inside you'll find:

| File | One row is |
| --- | --- |
| `..._simplified_training_log.csv` | One training trial |
| `..._session_log.csv` | One training session, start to finish |
| `..._collection_trials.csv` | One experimental trial |
| `..._collection_touches.csv` | One individual tap, for lining up with video |
| `..._collection_sessions.csv` | One data-collection session |
| `..._probe_log.csv` | One trial from an effort test |

These are plain spreadsheet files — you can open any of them in Excel.

**You don't need to maintain these files.** If the program is updated and gains a new
column, it quietly updates your old files to match the next time you run it, and
saves a dated backup alongside first. You don't have to fix anything by hand, and you
shouldn't try — editing them yourself can confuse the updater.

---

## Reading the results

Open `Analyze_task.py`, put your file paths at the top, and run it. It produces a
written summary and a set of figures, in three groups depending on what you give it:
training progress, effort-test results, or experimental data. Anything you leave
blank is skipped.

It only ever reads. It will not change or overwrite your data.

---

## Good to know

**One "session" is a count of trials, not a sitting.** The program counts a session
as finished when the set number of trials is done. If you stop early and come back
tomorrow, it picks up the same session where it left off. So a single session in the
files can span several days, and you shouldn't read the dates in a session as one
visit.

**The freebie trials count toward the percentages.** They're automatically correct, so
a session's overall "percent correct" is flattered by however many of them that stage
includes. The final-stage check looks at each shape separately and isn't affected.

**There's no time limit on starting a trial.** The animal can take as long as it likes
to touch the first circle. The clock only starts once the shape appears.

---

## If something goes wrong

**It won't start, and mentions the device or HID.** The pellet dispenser isn't plugged
in, isn't powered, or another program is holding on to it. Unplug it, plug it back in,
and try again.

**The first trial of a session doesn't deliver a pellet.** This was a real fault and
it's fixed. The program now sends a harmless warm-up signal to the dispenser when it
opens, and delivers one free pellet before the first trial. If you're still seeing it,
the dispenser itself likely needs adjusting — try raising `PELLET_PULSE_MS`.

**Everything feels sluggish on the Pi.** Check whether you're watching through remote
screen sharing — that's usually the screen sharing, not the task. If it's slow on the
Pi's own monitor, make sure `RELAY_VERBOSE` is set to `False`.

**A spreadsheet won't open, or your analysis complains about the number of columns.**
Just run the task program once for that animal. It repairs the file automatically and
keeps a backup. Don't edit it by hand first.

---

## Reusing this

The code is released under the MIT license (see `LICENSE`) — you're welcome to use and
adapt it, as long as the original credit stays attached.

If you use it in published work, please cite it. And if you adapt it for your own lab,
I'd be glad to hear about it.
