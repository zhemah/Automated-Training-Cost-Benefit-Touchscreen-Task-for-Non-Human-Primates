"""
Simplified Cost-Benefit Training Task for NHP Touchscreen
Python 3.11 | Pygame | ADU200 Relay

TASK STRUCTURE (5 trial types, randomized):
  PLAIN   — touch cyan circle → instant green screen + 1 pellet (low probability)
            NOTE: PLAIN trials are always auto-correct (no mechanic to fail);
            they count toward "Current % Correct" / session pass thresholds.
  HCHR    — touch circle → diamond composite appears (gold/tan) →
             touch composite N times → green screen + 3 pellets
             FAIL: timeout or incomplete → red screen + penalized ITI, no pellet
  HCLR    — touch circle → scalloped-flower composite appears (grey) →
             touch composite N times (same N as HCHR, derived from Phase 4
             breaking point) → green screen + 1 pellet
             FAIL: timeout or incomplete → red screen + penalized ITI, no pellet
  LCLR    — touch circle → triangle composite appears (pink) →
             touch composite once → green screen + 1 pellet
             FAIL: timeout → red screen + penalized ITI, no pellet
  LCHR    — touch circle → star composite appears (purple) →
             touch composite once → green screen + 3 pellets
             FAIL: timeout → red screen + penalized ITI, no pellet

TRAINING PHASES (automated, no operator input between sessions):
  Phase 1  — touch full-screen circle
  Phase 2  — touch variable-size circle
  Phase 3  — introduce HCHR cue, N=1 touch only, 50/50 with plain
  Phase 4  — ramp HCHR required touches N=2→3→4→5→6→8→10→12 (PHASE4_N_VALUES);
             breaking-point analysis determines the N used by HCHR and HCLR
             from Phase 5 onward. An entry in HIGH_COST_N_OVERRIDES ends the
             ramp immediately — checked at launch, so no further ramp session
             is run — and sends the animal to Phase 5 with that N.
  Phase 5  — Cue 2 intro (HCHR always first; positions 2-4 configured per-monkey)
             Mix: 30% HCHR / 50% Cue2 / 20% plain.
             Pass: 2 consecutive sessions, 50 trials/session
             Threshold: 60% if Cue2=HCLR, 75% if Cue2=LCLR or LCHR
  Phase 6  — Cue 3 intro.
             Mix: 20% HCHR / 20% Cue2 / 50% Cue3 / 10% plain.
             Pass: >=70% correct; 2 consecutive sessions, 50 trials/session
  Phase 7  — Cue 4 intro (LCHR).
             Mix: 15% HCHR / 20% Cue2 / 20% Cue3 / 35% LCHR / 10% plain.
             Pass: >=70% correct; 2 consecutive sessions, 50 trials/session.
             Timeout: 40 s per touch.
  Phase 8  — full task (recording-ready), all four cue types present.
             Mix: 25% HCHR / 25% HCLR / 25% LCLR / 15% LCHR / 10% plain.
             Pass: independent per-cue rolling windows (3 consecutive sessions):
               HCHR >=65%, HCLR >=35%, LCLR >=75%, LCHR >=80%
             Timeout: 20 s for all cue types.

  Cue positions 2, 3, and 4 are configurable per-monkey via CUE_ORDER_OVERRIDES.
  HCHR is always position 1. Default order: HCHR, HCLR, LCLR, LCHR.

SHAPE ASSETS (all drawn procedurally — no external files required):
  HCHR  — diamond polygon, gold/tan (#CCA15D)
  HCLR  — 6-lobe scalloped flower (overlapping circles), grey (128,128,128)
  LCLR  — equilateral triangle polygon, soft pink (#FFC7D7)
  LCHR  — 5-pointed star polygon, purple (#A785E8)

RELAY NOTES (ADU200):
  3 pellets for HCHR and LCHR; 1 pellet for HCLR, LCLR, and PLAIN.
  Pulse and inter-pellet timing tuned via PELLET_PULSE_MS and PELLET_INTER_MS.
  All relay calls are commented out for hardware-free testing. Uncomment to activate.

FILE STRUCTURE:
  {monkey_name}/
    {monkey_name}_training_log.csv   — trial-by-trial data
    {monkey_name}_session_log.csv    — one row per session with start/end times + summary
  (no image assets — every cue shape is drawn procedurally)

INTER-TOUCH LOCKOUT (HCHR / HCLR):
  After each touch except the last, the screen goes dark for a random
  100-300 ms (uniform, re-drawn every gap). Touches inside it are discarded and
  counted as premature; the composite reappears when the next touch is valid.
  See INTER_TOUCH_LOCKOUT_ENABLED.

BREAKING-POINT ENGAGEMENT MEASURE:
  Engagement is HCHR trials per hour of SCREEN TIME, not trials per session.
  Trials per session is flat by construction (a session only ends when the
  quota is filled, and Phase 4 is 75% HCHR, so every completed session holds
  ~37 HCHR trials at any N), which made three of the five criteria unable to
  fire. Screen time excludes gaps longer than SCREEN_TIME_MAX_GAP_S, because
  "Session #" only advances on a filled quota and so can span several calendar
  days. See the notes above the engagement constants.

LOG HEADER MIGRATION:
  Every CSV this program writes is opened through _ensure_csv, which migrates an
  older header in place (after a timestamped backup) by re-keying rows BY COLUMN
  NAME, and refuses to touch a file that is not a known version of that schema.
  When adding a column: append it at the END of the header list and record the
  previous list as a new version in the matching *_HISTORY.
"""

import pygame
import csv
import shutil
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import hid
import pandas as pd

# ============================================================
# TUNABLE CONSTANTS
# ============================================================
PELLET_PULSE_MS     = 80     # ms relay stays closed per pellet — calibrate per dispenser
PELLET_INTER_MS     = 300    # ms between pellet pulses (reset time for mechanism)
CUE_APPEAR_DELAY_MS = 50    # ms between circle touch and cue composite appearing
FLICKER_OFF_MS      = 150    # ms composite disappears on each HCHR/HCLR touch (feedback)

# ---- Inter-touch lockout (multi-touch cues only: HCHR and HCLR) ----
# After every touch except the last, the screen goes dark for a random lockout
# drawn uniformly from [MIN, MAX] ms, independently for each gap. Touches that
# land inside the lockout are DISCARDED (and counted as premature); the
# composite reappears exactly when the next touch becomes valid, so its
# reappearance is the go signal. The per-touch timeout starts at reappearance,
# so the lockout never eats into the time allowed to respond.
#
# HCHR and HCLR share this through _run_multi_touch_cue, so the only difference
# between them is still the reward. Single-touch cues (LCLR, LCHR) and PLAIN
# have no inter-touch gap and are unaffected.
#
# When disabled, the original fixed FLICKER_OFF_MS flicker is used unchanged.
INTER_TOUCH_LOCKOUT_ENABLED = True
INTER_TOUCH_LOCKOUT_MIN_MS  = 100
INTER_TOUCH_LOCKOUT_MAX_MS  = 300
# Touches within this many ms of a valid touch are treated as screen bounce and
# ignored, rather than counted as premature responses.
PREMATURE_DEBOUNCE_MS       = 50


def lockout_label():
    """Short description of the lockout setting, written into the logs."""
    return (f"{INTER_TOUCH_LOCKOUT_MIN_MS}-{INTER_TOUCH_LOCKOUT_MAX_MS}"
            if INTER_TOUCH_LOCKOUT_ENABLED else "off")
RESPONSE_TIMEOUT_RAMP_MS  = 60_000  # Phases 4–6: 1 minute per touch
RESPONSE_TIMEOUT_PHASE7MS = 40_000  # Phase 7 (LCHR intro): 40 seconds per touch
RESPONSE_TIMEOUT_FULL_MS  = 20_000  # Phase 8 (full task): 20 seconds for all cue types
PENALTY_ITI_MS      = 8000   # ms ITI after a failed trial (longer than normal)
NORMAL_ITI_MS       = 4000   # ms ITI after a correct trial
# ---- Relay / console ----
# write_to_adu and read_from_adu used to print on every call: ~10 prints per
# rewarded trial. On a Pi console (or over SSH) each print flushes and blocks.
# Set True only when debugging the relay.
RELAY_VERBOSE = False

# ---- Session-start priming pellet ----
# One free pellet delivered right after Shift+S, before the first trial.
# Two purposes:
#   1. Gives the hand-tracking camera a clean, unambiguous reach-and-retrieve
#      movement at a known timestamp, before any decision is involved.
#   2. Absorbs the first-write loss described in dispense_pellets() — the
#      first HID output report after enumeration is frequently dropped, which
#      is why the first trial of a session often produced no pellet.
PRIMING_PELLET_ENABLED      = True
PRIMING_PELLET_COUNT        = 1
# Delay between the "recording" cue and the pellet firing. When a camera is
# added, set this long enough for it to be actually recording (a few seconds).
PRIMING_PELLET_PRE_DELAY_MS = 3000

CIRCLE_COLOR    = (0, 250, 250)    # cyan
DIAMOND_COLOR   = (204, 161,  93)  # HCHR: warm gold/tan  (#CCA15D)
FLOWER_COLOR    = (128,  128,  128)  # HCLR: grey
TRIANGLE_COLOR  = (255, 199, 215)  # LCLR: soft pink      (#FFC7D7)
STAR_COLOR      = (167, 133, 232)  # LCHR: medium purple  (#A785E8) — matches uploaded reference

# ============================================================
# RELAY SETUP
# ============================================================
VENDOR_ID  = 0x0a07   # Ontrak
PRODUCT_ID = 200      # ADU200

device = hid.device()
device.open(VENDOR_ID, PRODUCT_ID)

# HID devices frequently drop the first OUTPUT report sent immediately after
# enumeration, and the input buffer can hold stale bytes from initialisation.
# That is the most likely reason the first trial of a session often produced
# no pellet: the very first 'SK0' was swallowed. Settle briefly, send a
# throwaway relay-open, then drain anything sitting in the read buffer.
time.sleep(0.20)
try:
    device.write((chr(0x01) + 'RK0' + chr(0) * 4).encode())   # throwaway, may be dropped
    time.sleep(0.05)
    device.set_nonblocking(1)
    while device.read(8):                                      # drain stale input
        pass
    device.set_nonblocking(0)
except Exception as _e:
    print(f"Relay warm-up skipped: {_e}")
print(f'Connected to ADU {PRODUCT_ID}')


# ============================================================
# RELAY HELPERS
# ============================================================
def sleep_ms(ms, pump=True):
    """
    Non-busy sleep.

    pygame.time.delay() spins the CPU to stay accurate; pygame.time.wait()
    sleeps and yields it. With 4 s normal ITIs and 8 s penalty ITIs, using
    delay() pegged a core for most of every session — the main cause of
    system-wide lag on the Raspberry Pi, and it would also starve a camera
    process. Sleeping in short chunks and pumping the event queue keeps the
    window responsive without burning CPU.
    """
    ms = int(ms)
    if ms <= 0:
        return
    if not pump:
        pygame.time.wait(ms)
        return
    end = pygame.time.get_ticks() + ms
    while True:
        remaining = end - pygame.time.get_ticks()
        if remaining <= 0:
            break
        pygame.time.wait(min(remaining, 50))
        pygame.event.pump()      # keep the OS from marking the window hung


def write_to_adu(device, command):
    if RELAY_VERBOSE:
        print(f'WRITING COMMAND {command}')
    byte_str = chr(0x01) + command + chr(0) * max(7 - len(command), 0)
    try:
        num_bytes = device.write(byte_str.encode())
    except IOError as e:
        print(f'IOError writing to device: {e}')
        return None
    return num_bytes


def read_from_adu(device, timeout_ms):
    if RELAY_VERBOSE:
        print(f'READING DATA timeout={timeout_ms}ms')
    try:
        data = device.read(8, timeout_ms)
    except IOError as e:
        print(f'IOError reading from device: {e}')
        return None
    byte_str = ''.join(chr(n) for n in data[1:])
    result_str = byte_str.split('\x00', 1)[0]
    if len(result_str) == 0:
        if RELAY_VERBOSE:
            print('No data received from device.')
        return None
    return result_str


def dispense_pellets(device, n_pellets):
    """
    Fire relay K0 n_pellets times in sequence.
    Each cycle: SK0 (close) → wait PELLET_PULSE_MS → RK0 (open) → wait PELLET_INTER_MS.
    Returns "Yes" if all sensor reads confirm delivery, "No" if any fail.
    """
    all_delivered = True
    for i in range(n_pellets):
        write_to_adu(device, 'SK0')
        time.sleep(PELLET_PULSE_MS / 1000)
        write_to_adu(device, 'RK0')
        write_to_adu(device, 'RPA0')
        sensor_data = read_from_adu(device, 200)
        if sensor_data is None or sensor_data.strip() != "1":
            all_delivered = False
        if i < n_pellets - 1:
            time.sleep(PELLET_INTER_MS / 1000)
    return "Yes" if all_delivered else "No"


def deliver_priming_pellet(screen, clock, mono_ref=None):
    """
    Deliver a free pellet at session start, before the first trial.

    Returns (mono_offset_seconds, delivered) where mono_offset is relative to
    mono_ref if given, else absolute time.monotonic(). Logging the timestamp
    lets the reach-and-retrieve movement be located exactly in the video.

    The pre-delay exists so an external camera has time to actually be
    recording before the pellet fires. With no camera attached the delay is
    harmless; when one is added, raise PRIMING_PELLET_PRE_DELAY_MS to cover
    its start-up latency.
    """
    if not PRIMING_PELLET_ENABLED:
        return ("", "")

    W, H = screen.get_width(), screen.get_height()
    font_big = pygame.font.Font(None, 78)
    font_sm  = pygame.font.Font(None, 40)
    end = pygame.time.get_ticks() + PRIMING_PELLET_PRE_DELAY_MS

    # Countdown so the operator can confirm the camera is rolling.
    while True:
        left = end - pygame.time.get_ticks()
        if left <= 0:
            break
        screen.fill((15, 15, 20))
        a = font_big.render("Priming pellet", True, (255, 255, 255))
        screen.blit(a, a.get_rect(center=(W // 2, H // 2 - 70)))
        b = font_sm.render(
            "camera should be recording" if CAMERA_RECORDING_ENABLED
            else "(no camera configured)", True, (170, 170, 180))
        screen.blit(b, b.get_rect(center=(W // 2, H // 2)))
        c = font_big.render(f"{left / 1000:.1f}", True, (120, 220, 255))
        screen.blit(c, c.get_rect(center=(W // 2, H // 2 + 80)))
        pygame.display.flip()
        pygame.time.wait(30)
        pygame.event.pump()

    screen.fill('#008000')
    pygame.display.flip()
    mono = time.monotonic()
    delivered = dispense_pellets(device, PRIMING_PELLET_COUNT)
    sleep_ms(400)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    write_to_adu(device, 'RK0')
    pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION])

    offset = round(mono - mono_ref, 4) if mono_ref is not None else round(mono, 4)
    print(f"Priming pellet delivered={delivered} at mono {offset}")
    return (offset, delivered)


# ============================================================
# OPERATOR MENUS
# ============================================================
def monkey_user_menu_pygame(screen, clock):
    """
    Fullscreen touchscreen monkey selection menu.
    Displays four large buttons — operator taps one to select the monkey.
    Returns the monkey name string (e.g. 'Uva').
    """
    monkeys = [
        ('U', 'Uva'),
        ('V', 'Vino'),
        ('R', 'Ramen'),
        ('Y', 'Yoshi'),
    ]

    button_colors = [
        (155, 89, 182),   # purple - Uva
        (231, 76, 60),    # red   - Vino
        (46, 204, 113),   # green  - Ramen
        (52, 152, 219),   # blue   - Yoshi

    ]
    hover_colors = [
        (125, 60, 152),
        (192, 57, 43),
        (39, 174, 96),
        (41, 128, 185),
    ]

    font_title  = pygame.font.Font(None, 72)
    font_button = pygame.font.Font(None, 100)
    font_key    = pygame.font.Font(None, 56)

    W = screen.get_width()
    H = screen.get_height()

    pad      = 40
    btn_w    = (W - pad * 3) // 2
    btn_h    = (H - pad * 3 - 120) // 2
    grid_top = 120

    buttons = []
    for i, (key, name) in enumerate(monkeys):
        col = i % 2
        row = i // 2
        x = pad + col * (btn_w + pad)
        y = grid_top + row * (btn_h + pad)
        rect = pygame.Rect(x, y, btn_w, btn_h)
        buttons.append((rect, key, name, button_colors[i], hover_colors[i]))

    selected = None

    while selected is None:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                key_char = pygame.key.name(event.key).upper()
                for _, k, name, _, _ in buttons:
                    if key_char == k:
                        selected = name
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, key, name, color, hcolor in buttons:
                    if rect.collidepoint(event.pos):
                        selected = name

        screen.fill((15, 15, 20))

        title_surf = font_title.render("Who is using the touchscreen?", True, (220, 220, 220))
        screen.blit(title_surf, title_surf.get_rect(centerx=W // 2, top=pad))

        for rect, key, name, color, hcolor in buttons:
            is_hovered = rect.collidepoint(mouse_pos)
            draw_color = hcolor if is_hovered else color

            shadow_rect = rect.move(4, 4)
            pygame.draw.rect(screen, (10, 10, 10), shadow_rect, border_radius=20)
            pygame.draw.rect(screen, draw_color, rect, border_radius=20)

            name_surf = font_button.render(name, True, (255, 255, 255))
            name_rect = name_surf.get_rect(center=rect.center)
            name_rect.centery = rect.centery - 18
            screen.blit(name_surf, name_rect)

            key_surf = font_key.render(f"[{key}]", True, (200, 200, 200))
            key_rect = key_surf.get_rect(centerx=rect.centerx, top=name_rect.bottom + 8)
            screen.blit(key_surf, key_rect)

        pygame.display.flip()
        clock.tick(60)

    # Brief confirmation flash
    screen.fill((30, 30, 30))
    confirm_font = pygame.font.Font(None, 90)
    confirm_surf = confirm_font.render(f"{selected} selected", True, (80, 220, 80))
    screen.blit(confirm_surf, confirm_surf.get_rect(center=(W // 2, H // 2)))
    pygame.display.flip()
    sleep_ms(1000)
    pygame.event.clear()

    return selected


def wait_for_start_command(screen, clock):
    font = pygame.font.Font(None, 48)
    while True:
        screen.fill((0, 0, 0))
        text = font.render("Press Shift+S to start", True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2)))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_s and (event.mod & pygame.KMOD_SHIFT):
                    return True
        clock.tick(60)


def wait_for_exit_command(screen, clock):
    font = pygame.font.Font(None, 48)
    while True:
        screen.fill((0, 0, 0))
        text = font.render("Task stopped. Press Shift+E to exit.", True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2)))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e and (event.mod & pygame.KMOD_SHIFT):
                    return
        clock.tick(60)


# ============================================================
# DRAWING HELPERS
# ============================================================
ARC_COLOR      = (0, 128, 0)
ARC_RING_SCALE = 1.35
ARC_RING_WIDTH = 15


def _draw_arc_progress(screen, center, radius, touches_done, n_touches_required):
    if n_touches_required <= 0 or touches_done <= 0:
        return
    progress = min(touches_done / n_touches_required, 1.0)
    ring_r = int(radius * ARC_RING_SCALE)
    rect = pygame.Rect(0, 0, ring_r * 2, ring_r * 2)
    rect.center = center
    start_angle = math.pi / 2
    sweep       = 2 * math.pi * progress
    end_angle   = start_angle
    start_angle = start_angle - sweep
    pygame.draw.arc(screen, ARC_COLOR, rect, start_angle, end_angle, ARC_RING_WIDTH)


# ------------------------------------------------------------------
# Cached shape surfaces
# ------------------------------------------------------------------
# The cue shape behind the cyan circle never changes within a session — only
# the progress arc does. Re-rasterising it on every redraw is wasted work, and
# on a Raspberry Pi it is software-rasterised: the HCLR flower is six large
# filled circles, roughly 30-50 ms per redraw versus ~10 ms for a simple
# polygon. That cost is paid twice per touch (blank, then redraw), so on a
# 6-touch HCLR trial it adds up to a visible pause between touches.
#
# Each shape is therefore rasterised ONCE into a transparent Surface, keyed by
# (cue, radius), and thereafter just blitted — turning ~40 ms into ~2 ms.
# The circle and the arc are still drawn live on top, since the arc changes
# with every touch.
_SHAPE_CACHE = {}


def _shape_geometry(cue, radius):
    """Return (extent_px, draw_fn) for a cue. extent = furthest point from centre."""
    if cue == "HCHR":
        dr = int(radius * 1.5)
        def draw(surf, c):
            pygame.draw.polygon(surf, DIAMOND_COLOR,
                                [(c, c - dr), (c - dr, c), (c, c + dr), (c + dr, c)])
        return dr, draw

    if cue == "HCLR":
        lobe_r = int(radius * 0.65)
        orbit  = lobe_r
        def draw(surf, c):
            for deg in range(0, 360, 60):
                rad = math.radians(deg)
                pygame.draw.circle(
                    surf, FLOWER_COLOR,
                    (c + int(orbit * math.sin(rad)), c - int(orbit * math.cos(rad))),
                    lobe_r)
        return orbit + lobe_r, draw

    if cue == "LCLR":
        cr = 2 * radius
        def draw(surf, c):
            pygame.draw.polygon(surf, TRIANGLE_COLOR, [
                (c + int(cr * math.sin(math.radians(a))),
                 c - int(cr * math.cos(math.radians(a)))) for a in (0, 120, 240)])
        return cr, draw

    if cue == "LCHR":
        outer_r = int(radius * 1.85)
        inner_r = int(outer_r * 0.42)
        def draw(surf, c):
            pts = []
            for i in range(5):
                a_out = math.radians(-90 + i * 72)
                pts.append((c + int(outer_r * math.cos(a_out)),
                            c + int(outer_r * math.sin(a_out))))
                a_in = math.radians(-90 + i * 72 + 36)
                pts.append((c + int(inner_r * math.cos(a_in)),
                            c + int(inner_r * math.sin(a_in))))
            pygame.draw.polygon(surf, STAR_COLOR, pts)
        return outer_r, draw

    raise ValueError(f"Unknown cue '{cue}'")


def _get_shape_surface(cue, radius):
    """
    Return the cached Surface for this cue/radius, building it on first use.
    Falls back to None if the surface cannot be created, in which case callers
    draw the shape directly (correctness never depends on the cache).
    """
    key = (cue, int(radius))
    surf = _SHAPE_CACHE.get(key)
    if surf is not None:
        return surf
    try:
        extent, draw = _shape_geometry(cue, radius)
        size = extent * 2 + 4
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        draw(surf, size // 2)
        surf = surf.convert_alpha()      # match display format for fast blitting
        _SHAPE_CACHE[key] = surf
        return surf
    except Exception as e:
        print(f"Shape cache unavailable for {cue} r={radius} ({e}); drawing directly.")
        return None


def prebuild_shape_cache(radius, cues=("HCHR", "HCLR", "LCLR", "LCHR")):
    """
    Build every shape surface up front so the first trial does not pay the
    rasterisation cost. Call once per session after the radius is known.
    """
    for cue in cues:
        _get_shape_surface(cue, radius)
    print(f"Shape cache ready for radius {radius}: {sorted({k[0] for k in _SHAPE_CACHE})}")


def _blit_shape(screen, center, radius, cue):
    """Blit the cached shape centred on `center`; draw directly if unavailable."""
    surf = _get_shape_surface(cue, radius)
    if surf is None:
        extent, draw = _shape_geometry(cue, radius)
        tmp = pygame.Surface((extent * 2 + 4, extent * 2 + 4), pygame.SRCALPHA)
        draw(tmp, (extent * 2 + 4) // 2)
        surf = tmp
    screen.blit(surf, surf.get_rect(center=center))


def _render_hchr(screen, center, radius, touches_done, n_touches_required):
    """HCHR: 4-point diamond (gold/tan), from cached surface."""
    _blit_shape(screen, center, radius, "HCHR")
    pygame.draw.circle(screen, CIRCLE_COLOR, center, radius)
    _draw_arc_progress(screen, center, radius, touches_done, n_touches_required)


def _render_hclr(screen, center, radius, touches_done, n_touches_required):
    """
    HCLR: 6-lobe scalloped flower (grey), from cached surface.
    Six filled circles of radius 0.65x, centres on a ring of that same radius
    at 60 degree intervals; the overlap gives the rounded-petal silhouette.
    Arc-progress overlay drawn live on top, same as HCHR.
    """
    _blit_shape(screen, center, radius, "HCLR")
    pygame.draw.circle(screen, CIRCLE_COLOR, center, radius)
    _draw_arc_progress(screen, center, radius, touches_done, n_touches_required)


def _render_lclr(screen, center, radius):
    """LCLR: equilateral triangle (soft pink), from cached surface."""
    _blit_shape(screen, center, radius, "LCLR")
    pygame.draw.circle(screen, CIRCLE_COLOR, center, radius)


def _render_lchr(screen, center, radius):
    """
    LCHR: 5-pointed star (purple), from cached surface.
    Ten alternating vertices; inner/outer ratio 0.42 gives the plump star that
    matches the reference. First outer vertex at -90 deg so the top point faces up.
    Single-touch cue — no arc-progress overlay.
    """
    _blit_shape(screen, center, radius, "LCHR")
    pygame.draw.circle(screen, CIRCLE_COLOR, center, radius)


def draw_circle_only(screen, center, radius):
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, CIRCLE_COLOR, center, radius)
    pygame.display.flip()


def draw_hchr_composite(screen, center, radius, touches_done=0, n_touches_required=1):
    screen.fill((0, 0, 0))
    _render_hchr(screen, center, radius, touches_done, n_touches_required)
    pygame.display.flip()


def draw_hclr_composite(screen, center, radius, touches_done=0, n_touches_required=1):
    screen.fill((0, 0, 0))
    _render_hclr(screen, center, radius, touches_done, n_touches_required)
    pygame.display.flip()


def draw_lclr_composite(screen, center, radius):
    screen.fill((0, 0, 0))
    _render_lclr(screen, center, radius)
    pygame.display.flip()


def draw_lchr_composite(screen, center, radius):
    screen.fill((0, 0, 0))
    _render_lchr(screen, center, radius)
    pygame.display.flip()


def hit_circle(pos, center, radius):
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    return (dx * dx + dy * dy) ** 0.5 <= radius


# ============================================================
# OUTCOME SCREENS
# ============================================================
def show_session_result_screen(screen, clock, phase_num, substage_idx,
                               session_num, pct_correct, passed, advanced_to):
    font_large = pygame.font.Font(None, 64)
    font_small = pygame.font.Font(None, 40)
    display_ms = 3000
    start = pygame.time.get_ticks()

    if passed and advanced_to:
        adv_phase, adv_sub = advanced_to
        status_text  = "STAGE PASSED"
        status_color = (80, 220, 80)
        next_text    = f"Advancing to Phase {adv_phase}, Substage {adv_sub}"
    else:
        status_text  = f"{pct_correct:.1f}% correct — continuing"
        status_color = (220, 220, 80)
        next_text    = f"Phase {phase_num}, Substage {substage_idx}, Session {session_num + 1}"

    while True:
        elapsed = pygame.time.get_ticks() - start
        if elapsed >= display_ms:
            break
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "escaped"

        screen.fill((0, 0, 0))
        remaining = max(0, (display_ms - elapsed) // 1000 + 1)
        lines = [
            (f"Session {session_num} complete", (200, 200, 200), font_small),
            (status_text,                        status_color,    font_large),
            (next_text,                          (180, 180, 255), font_small),
            (f"Starting in {remaining}s…",       (120, 120, 120), font_small),
        ]
        total_h = sum(f.size(t)[1] + 16 for t, _, f in lines)
        y = (screen.get_height() - total_h) // 2
        for text, color, font in lines:
            surf = font.render(text, True, color)
            rect = surf.get_rect(centerx=screen.get_width() // 2, top=y)
            screen.blit(surf, rect)
            y += font.size(text)[1] + 16
        pygame.display.flip()
        clock.tick(30)

    return "ok"


# ============================================================
# CUE INTERACTION — shared multi-touch mechanic (HCHR + HCLR)
# ============================================================
def _wait_inter_touch_lockout(lockout_ms, center, radius, touch_tick):
    """
    Keep the screen dark for `lockout_ms`, DISCARDING every touch that lands in
    that window. Returns (premature_touches_on_target, escaped).

    Discarding is the whole point, and a plain sleep does not do it: pygame
    keeps queueing touches while the program sleeps and delivers them as soon
    as it wakes. A sleep-based "lockout" would therefore let a touch made during
    it be counted the instant it ended — nothing would actually be prevented.
    (The old fixed 150 ms flicker had exactly this property: a tap on the black
    screen was queued and then counted when the composite reappeared.)

    Premature taps on the target area are counted rather than silently thrown
    away, since tapping during an enforced wait is itself behaviour. Anything
    within PREMATURE_DEBOUNCE_MS of the valid touch is treated as screen bounce.
    """
    end = pygame.time.get_ticks() + int(lockout_ms)
    premature = 0

    def drain():
        nonlocal premature
        now = pygame.time.get_ticks()
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return True
            if (ev.type == pygame.MOUSEBUTTONDOWN
                    and hit_circle(ev.pos, center, radius)
                    and now - touch_tick >= PREMATURE_DEBOUNCE_MS):
                premature += 1
        return False

    while True:
        if drain():
            return premature, True
        remaining = end - pygame.time.get_ticks()
        if remaining <= 0:
            break
        pygame.time.wait(min(remaining, 10))
    if drain():                       # anything that arrived in the final slice
        return premature, True
    return premature, False


def _run_multi_touch_cue(screen, clock, center, radius, n_touches_required,
                          response_timeout_ms, draw_composite_fn, error_prefix,
                          touch_log=None, cue_stats=None):
    """
    Shared touch-counting/timeout logic for any multi-touch high-cost cue.
    HCHR and HCLR are mechanically identical (same touch-counting state
    machine, same timeout behavior, same inter-touch lockout) and differ only in
    which shape/color is drawn (draw_composite_fn) and which error-label prefix
    is used ("hchr" vs "hclr") so the logged Error Type always matches the
    actual Trial Type.

    Inter-touch lockout: see INTER_TOUCH_LOCKOUT_ENABLED. After every touch but
    the last, the screen stays dark for a random lockout and touches in it are
    discarded; after the last touch there is nothing to lock out, so the
    original fixed FLICKER_OFF_MS feedback flicker is used.

    touch_log (optional list) receives one dict per valid touch, including the
    lockout that preceded it, premature taps during that lockout, and the RT
    from when the target became available — the clean motor measure, since
    raw inter-touch intervals now include the enforced lockout.

    cue_stats (optional dict) receives {"premature_touches": int} on EVERY
    exit, including timeouts, so taps made during a lockout that was followed
    by abandonment are not lost.
    """
    touches_done      = 0
    premature_total   = 0
    pending_lockout   = ""       # lockout that preceded the next valid touch
    pending_premature = ""       # premature taps during that lockout

    def finish(result):
        if cue_stats is not None:
            cue_stats["premature_touches"] = premature_total
        return result

    draw_composite_fn(screen, center, radius,
                       touches_done=0, n_touches_required=n_touches_required)
    sequence_start    = pygame.time.get_ticks()
    available_tick    = sequence_start        # when the target last became touchable
    first_touch_rt_ms = "N/A"

    while touches_done < n_touches_required:
        touch_start = pygame.time.get_ticks()   # timeout runs from availability

        waiting_for_touch = True
        while waiting_for_touch:
            now = pygame.time.get_ticks()

            if response_timeout_ms is not None:
                if (now - touch_start) >= response_timeout_ms:
                    elapsed = now - sequence_start
                    error = (f"{error_prefix}_start_timeout" if touches_done == 0
                             else f"{error_prefix}_incomplete_timeout")
                    return finish((False, touches_done, elapsed, first_touch_rt_ms, error))

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    elapsed = pygame.time.get_ticks() - sequence_start
                    return finish((False, touches_done, elapsed, first_touch_rt_ms, "escaped"))

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if hit_circle(event.pos, center, radius):
                        touch_tick = pygame.time.get_ticks()
                        if touches_done == 0:
                            first_touch_rt_ms = touch_tick - touch_start
                        touches_done += 1
                        if touch_log is not None:
                            # monotonic clock + screen position of EVERY touch,
                            # for offline alignment with hand-tracking video
                            touch_log.append({
                                "index": touches_done,
                                "mono":  time.monotonic(),
                                "wall":  datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                "x": event.pos[0], "y": event.pos[1],
                                "lockout_ms":        pending_lockout,
                                "premature_before":  pending_premature,
                                "rt_from_available": touch_tick - available_tick,
                            })

                        screen.fill((0, 0, 0))
                        pygame.display.flip()

                        if touches_done < n_touches_required and INTER_TOUCH_LOCKOUT_ENABLED:
                            lockout = random.randint(INTER_TOUCH_LOCKOUT_MIN_MS,
                                                     INTER_TOUCH_LOCKOUT_MAX_MS)
                            premature, escaped = _wait_inter_touch_lockout(
                                lockout, center, radius, touch_tick)
                            premature_total += premature
                            if escaped:
                                elapsed = pygame.time.get_ticks() - sequence_start
                                return finish((False, touches_done, elapsed,
                                               first_touch_rt_ms, "escaped"))
                            pending_lockout, pending_premature = lockout, premature
                        else:
                            # Last touch (nothing left to lock out), or lockout
                            # disabled: the original fixed feedback flicker.
                            sleep_ms(FLICKER_OFF_MS)
                            pending_lockout, pending_premature = "", ""

                        draw_composite_fn(screen, center, radius,
                                           touches_done=touches_done,
                                           n_touches_required=n_touches_required)
                        available_tick = pygame.time.get_ticks()
                        if touches_done >= n_touches_required:
                            sleep_ms(200)

                        waiting_for_touch = False
                        break

            clock.tick(60)

    elapsed = pygame.time.get_ticks() - sequence_start
    return finish((True, touches_done, elapsed, first_touch_rt_ms, None))


def run_hchr_cue(screen, clock, center, radius, n_touches_required, response_timeout_ms,
                 touch_log=None, cue_stats=None):
    return _run_multi_touch_cue(
        screen, clock, center, radius, n_touches_required, response_timeout_ms,
        draw_composite_fn=draw_hchr_composite, error_prefix="hchr",
        touch_log=touch_log, cue_stats=cue_stats,
    )


def run_hclr_cue(screen, clock, center, radius, n_touches_required, response_timeout_ms,
                 touch_log=None, cue_stats=None):
    return _run_multi_touch_cue(
        screen, clock, center, radius, n_touches_required, response_timeout_ms,
        draw_composite_fn=draw_hclr_composite, error_prefix="hclr",
        touch_log=touch_log, cue_stats=cue_stats,
    )


# ============================================================
# CUE INTERACTION — single-touch low-cost cues (LCLR and LCHR)
# ============================================================
def _run_single_touch_cue(screen, clock, center, radius, response_timeout_ms,
                           draw_fn, render_fn, error_label, touch_log=None):
    """
    Shared single-touch state machine for LCLR and LCHR.
    Both cues require exactly one touch; they differ only in which
    composite is drawn and which error label is used.
    """
    draw_fn(screen, center, radius)
    cue_start = pygame.time.get_ticks()
    first_touch_rt_ms = "N/A"

    while True:
        now = pygame.time.get_ticks()
        if response_timeout_ms is not None:
            if (now - cue_start) >= response_timeout_ms:
                elapsed = now - cue_start
                return False, elapsed, "N/A", error_label

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                elapsed = pygame.time.get_ticks() - cue_start
                return False, elapsed, "N/A", "escaped"

            if event.type == pygame.MOUSEBUTTONDOWN:
                if hit_circle(event.pos, center, radius):
                    elapsed = pygame.time.get_ticks() - cue_start
                    first_touch_rt_ms = elapsed
                    if touch_log is not None:
                        touch_log.append({
                            "index": 1,
                            "mono":  time.monotonic(),
                            "wall":  datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            "x": event.pos[0], "y": event.pos[1],
                            "lockout_ms": "", "premature_before": "",
                            "rt_from_available": elapsed,
                        })
                    screen.fill((0, 0, 0))
                    render_fn(screen, center, radius)
                    _draw_arc_progress(screen, center, radius, 1, 1)
                    pygame.display.flip()
                    sleep_ms(200)
                    return True, elapsed, first_touch_rt_ms, None

        clock.tick(60)


def run_lclr_cue(screen, clock, center, radius, response_timeout_ms, touch_log=None):
    return _run_single_touch_cue(
        screen, clock, center, radius, response_timeout_ms,
        draw_fn=draw_lclr_composite, render_fn=_render_lclr,
        error_label="lclr_timeout", touch_log=touch_log,
    )


def run_lchr_cue(screen, clock, center, radius, response_timeout_ms, touch_log=None):
    """
    LCHR: single touch (low cost), 3 pellets (high reward), star shape (purple).
    Mechanically identical to LCLR — one touch, same timeout logic — but uses
    the star composite and pays 3 pellets instead of 1.
    """
    return _run_single_touch_cue(
        screen, clock, center, radius, response_timeout_ms,
        draw_fn=draw_lchr_composite, render_fn=_render_lchr,
        error_label="lchr_timeout", touch_log=touch_log,
    )


# ============================================================
# TRIAL TYPE SAMPLING
# ============================================================
def build_trial_list(trial_type_probs, n_trials):
    types = list(trial_type_probs.keys())
    counts = {}
    assigned = 0
    for t in types[:-1]:
        counts[t] = round(trial_type_probs[t] * n_trials)
        assigned += counts[t]
    counts[types[-1]] = n_trials - assigned

    trial_list = []
    for t, c in counts.items():
        trial_list.extend([t] * c)
    random.shuffle(trial_list)
    return trial_list


# ============================================================
# PER-MONKEY CUE PRESENTATION ORDER
# ============================================================
# HCHR is ALWAYS the first cue (Phases 3-4) and may NEVER be moved.
# Positions 2, 3, and 4 (introduced in Phases 5, 6, and 7 respectively)
# can appear in any order via DEFAULT_CUE_ORDER or CUE_ORDER_OVERRIDES.
# LCHR is always introduced last (position 4 / Phase 7) in the default order,
# but like positions 2 and 3, it can be moved to any non-first position if a
# specific monkey's history requires it.
#
# Default for all monkeys starting fresh:
DEFAULT_CUE_ORDER = ["HCHR", "HCLR", "LCLR", "LCHR"]

# Explicit per-monkey overrides.
# Example — a monkey who already learned HCHR then LCLR under the old 3-cue
# protocol; HCLR must come third and LCHR fourth:
#   "Uva": ["HCHR", "LCLR", "HCLR", "LCHR"],
CUE_ORDER_OVERRIDES = {
    "Vino": ["HCHR", "LCLR", "HCLR", "LCHR"],
}

_ALL_CUES = {"HCHR", "HCLR", "LCLR", "LCHR"}


def get_cue_order(monkey_name):
    """
    Return this monkey's full 4-cue presentation order.
    HCHR must always be in position 1; positions 2-4 are freely configurable.
    Raises AssertionError if the order is invalid.
    """
    order = CUE_ORDER_OVERRIDES.get(monkey_name, DEFAULT_CUE_ORDER)
    assert order[0] == "HCHR", (
        f"HCHR must always be first (introduced in Phases 3-4). "
        f"Got order={order} for monkey {monkey_name!r}."
    )
    assert set(order) == _ALL_CUES, (
        f"Cue order for {monkey_name!r} must contain exactly {sorted(_ALL_CUES)}. "
        f"Got {sorted(set(order))}."
    )
    assert len(order) == 4, f"Cue order must have exactly 4 entries, got {len(order)}."
    return order


def get_cue2(monkey_name):
    """Cue introduced in Phase 5 (position 2 in this monkey's order)."""
    return get_cue_order(monkey_name)[1]


def get_cue3(monkey_name):
    """Cue introduced in Phase 6 (position 3 in this monkey's order)."""
    return get_cue_order(monkey_name)[2]


def get_cue4(monkey_name):
    """Cue introduced in Phase 7 (position 4 in this monkey's order). Always LCHR in default order."""
    return get_cue_order(monkey_name)[3]


# Phase 5 pass threshold: 60% when Cue 2 is the high-cost HCLR (harder to
# learn at low reward), 75% for any low-cost cue (LCLR or LCHR).
PHASE5_PASS_THRESHOLD_HCLR = 60   # HCLR as Cue 2
PHASE5_PASS_THRESHOLD_LC   = 75   # LCLR or LCHR as Cue 2


def get_phase5_pass_threshold(monkey_name):
    cue2 = get_cue2(monkey_name)
    if cue2 == "HCLR":
        return PHASE5_PASS_THRESHOLD_HCLR
    elif cue2 in ("LCLR", "LCHR"):
        return PHASE5_PASS_THRESHOLD_LC
    else:
        raise ValueError(f"Unexpected Cue 2 value '{cue2}' for monkey {monkey_name!r}.")


# ============================================================
# STAGE CONFIG
# ============================================================
def get_stage_config(phase_num, substage_idx, monkey_name=None):
    if phase_num == 1:
        return {
            "description":         "Touch big circle",
            "trials_per_session":  100,
            "size_fraction":       1.0,
            "random_circle_size":  False,
            "trial_probs":         {"PLAIN": 1.0},
            "hchr_n":              1,
            "response_timeout_ms": None,
            "normal_iti_ms":       NORMAL_ITI_MS,
            "penalty_iti_ms":      PENALTY_ITI_MS,
            "required_sessions":   1,
            "required_trials":     100,
            "pass_threshold":      100,
        }

    elif phase_num == 2:
        return {
            "description":         "Variable size circle",
            "trials_per_session":  100,
            "size_fraction":       0.45,
            "random_circle_size":  True,
            "trial_probs":         {"PLAIN": 1.0},
            "hchr_n":              1,
            "response_timeout_ms": None,
            "normal_iti_ms":       NORMAL_ITI_MS,
            "penalty_iti_ms":      PENALTY_ITI_MS,
            "required_sessions":   1,
            "required_trials":     100,
            "pass_threshold":      100,
        }

    elif phase_num == 3:
        return {
            "description":         "HCHR intro N=1",
            "trials_per_session":  100,
            "size_fraction":       0.45,
            "random_circle_size":  False,
            "trial_probs":         {"PLAIN": 0.50, "HCHR": 0.50},
            "hchr_n":              1,
            "response_timeout_ms": None,
            "normal_iti_ms":       NORMAL_ITI_MS,
            "penalty_iti_ms":      PENALTY_ITI_MS,
            "required_sessions":   1,
            "required_trials":     100,
            "pass_threshold":      80,
        }

    elif phase_num == 4:
        hchr_n_values = PHASE4_N_VALUES     # single source of truth
        hchr_n = hchr_n_values[min(substage_idx, len(hchr_n_values) - 1)]
        required_sessions = 3 if substage_idx == len(hchr_n_values) - 1 else 2
        return {
            "description":         f"HCHR ramp N={hchr_n}",
            "trials_per_session":  50,
            "size_fraction":       0.45,
            "random_circle_size":  False,
            "trial_probs":         {"PLAIN": 0.25, "HCHR": 0.75},
            "hchr_n":              hchr_n,
            "response_timeout_ms": RESPONSE_TIMEOUT_RAMP_MS,
            "normal_iti_ms":       NORMAL_ITI_MS,
            "penalty_iti_ms":      PENALTY_ITI_MS,
            "required_sessions":   required_sessions,
            "required_trials":     50,
            "pass_threshold":      75,
        }

    elif phase_num == 5:
        # CUE 2 PRESENTATION. Single substage (substage_idx is always 0 here;
        # accepted as a parameter for interface consistency with the other
        # phases, but Phase 5 does not ramp across substages).
        cue2 = get_cue2(monkey_name)
        return {
            "description":         f"Cue 2 intro ({cue2})",
            "trials_per_session":  50,
            "size_fraction":       0.45,
            "random_circle_size":  False,
            "trial_probs":         {"HCHR": 0.30, cue2: 0.50, "PLAIN": 0.20},
            "hchr_n":              None,   # resolved at runtime via find_high_cost_touch_requirement
            "response_timeout_ms": RESPONSE_TIMEOUT_RAMP_MS,
            "normal_iti_ms":       NORMAL_ITI_MS,
            "penalty_iti_ms":      PENALTY_ITI_MS,
            "required_sessions":   2,
            "required_trials":     50,
            # Conditional: 60% if Cue 2 is HCLR, 75% if Cue 2 is LCLR.
            "pass_threshold":      get_phase5_pass_threshold(monkey_name),
        }

    elif phase_num == 6:
        # CUE 3 PRESENTATION. Single substage, same interface note as Phase 5.
        cue2 = get_cue2(monkey_name)
        cue3 = get_cue3(monkey_name)
        return {
            "description":         f"Cue 3 intro ({cue3})",
            "trials_per_session":  50,
            "size_fraction":       0.45,
            "random_circle_size":  False,
            "trial_probs":         {"HCHR": 0.20, cue2: 0.20, cue3: 0.50, "PLAIN": 0.10},
            "hchr_n":              None,   # resolved at runtime via find_high_cost_touch_requirement
            "response_timeout_ms": RESPONSE_TIMEOUT_RAMP_MS,
            "normal_iti_ms":       NORMAL_ITI_MS,
            "penalty_iti_ms":      PENALTY_ITI_MS,
            "required_sessions":   2,
            "required_trials":     50,
            "pass_threshold":      70,
        }

    elif phase_num == 7:
        # CUE 4 INTRODUCTION (LCHR — Low-Cost, High-Reward).
        # All three previously learned cues are present alongside the new one.
        # LCHR is always position 4 in the default order; get_cue4() returns
        # whichever cue this monkey has in position 4 of their order.
        cue2 = get_cue2(monkey_name)
        cue3 = get_cue3(monkey_name)
        cue4 = get_cue4(monkey_name)   # typically LCHR
        return {
            "description":         f"Cue 4 intro ({cue4})",
            "trials_per_session":  50,
            "size_fraction":       0.45,
            "random_circle_size":  False,
            "trial_probs":         {"HCHR": 0.15, cue2: 0.20, cue3: 0.20, cue4: 0.35, "PLAIN": 0.10},
            "hchr_n":              None,   # resolved at runtime
            "response_timeout_ms": RESPONSE_TIMEOUT_PHASE7MS,   # 40 s
            "normal_iti_ms":       NORMAL_ITI_MS,
            "penalty_iti_ms":      PENALTY_ITI_MS,
            "required_sessions":   2,
            "required_trials":     50,
            "pass_threshold":      70,
        }

    elif phase_num == 8:
        # FULL TASK (recording-ready). All four cue types present.
        # Pass criteria are NOT a single threshold — see PHASE8_CUE_THRESHOLDS
        # and pass_phase8_check. pass_threshold is None; it is not used.
        return {
            "description":         "Full task (recording)",
            "trials_per_session":  100,
            "size_fraction":       0.45,
            "random_circle_size":  False,
            "trial_probs":         {"HCHR": 0.25, "HCLR": 0.25, "LCLR": 0.25, "LCHR": 0.15, "PLAIN": 0.10},
            "hchr_n":              None,   # resolved at runtime
            "response_timeout_ms": RESPONSE_TIMEOUT_FULL_MS,   # 20 s for all cues
            "normal_iti_ms":       NORMAL_ITI_MS,
            "penalty_iti_ms":      PENALTY_ITI_MS,
            "required_sessions":   3,
            "required_trials":     100,
            "pass_threshold":      None,   # see pass_phase8_check
        }

    else:
        raise ValueError(f"Phase {phase_num} not implemented.")


def advance_stage(phase_num, substage_idx, monkey_name=None):
    if phase_num == 1:
        return 2, 0
    elif phase_num == 2:
        return 3, 0
    elif phase_num == 3:
        return 4, 0
    elif phase_num == 4:
        hchr_n_values = PHASE4_N_VALUES
        if substage_idx < len(hchr_n_values) - 1:
            return 4, substage_idx + 1
        else:
            return 5, 0
    elif phase_num == 5:
        return 6, 0   # Cue 2 intro → Cue 3 intro
    elif phase_num == 6:
        return 7, 0   # Cue 3 intro → Cue 4 intro
    elif phase_num == 7:
        return 8, 0   # Cue 4 intro → full task
    elif phase_num == 8:
        print("Phase 8 complete — training protocol finished.")
        return 8, 0
    else:
        raise ValueError(f"Advance logic for phase {phase_num} not implemented.")


# ============================================================
# BREAKING POINT DETECTION
# ============================================================
# Five criteria, evaluated in order as the ramp ascends. The FIRST criterion to
# fire defines the breaking point, and the high-cost touch requirement is set
# AT that N (not N-1), because the abandonment bar is deliberately low (35%) —
# an animal failing ~a third of trials at some N is still working at that level,
# so that level is the right cost to use.
#
# Criteria 3-5 were added after two real cases exposed blind spots:
#   * Uva stalled at Phase 4 for 7 weeks with ~20-32% abandonment and only
#     19/50 trials in a session. Neither original criterion fired: abandonment
#     never approached the old 60% bar, and the adjacent-N engagement drop was
#     51% (just under the 60% bar). C3 (absolute floor) catches her at N=4.
#   * Yoshi completed the entire ramp to N=6 with 0.9% abandonment. Nothing
#     fired, so the fallback silently returned 6 -- a FLOOR on his true
#     breaking point, not an estimate. The fallback is now a loud flagged
#     status instead of a silent default.
#
# C3 and C5 describe being STUCK AT THE CURRENT LEVEL, so they only evaluate at
# the highest N present in the data. Without that restriction Uva would fire the
# stall criterion at N=2 (4 sessions there) even though she passed it and moved
# on, wrongly fixing her at a very low N.

# ---- Engagement is measured as a RATE, not a count ----
# C2, C3 and C4 originally compared HCHR trials per session. That measure
# cannot work, and the logs prove it: a session only ends when
# trials_per_session trials are done, and Phase 4 draws 75% HCHR, so EVERY
# completed session contains ~37 HCHR trials no matter how the animal feels
# about the cost. Uva's log gives 37.8 / 39.0 / 39.0 / 38.5 / 37.5 trials per
# session at N = 2 / 3 / 4 / 5 / 6 / 8 — flat to within one trial. Three of the
# five criteria were dead code: they could only ever fire on a session the
# operator had abandoned part-way.
#
# The live measure is HCHR trials per hour of SCREEN TIME. Screen time is not
# wall-clock span, because "Session #" only advances when the trial quota is
# filled, so one session routinely spans several calendar days (Uva's Phase 4
# sub1 session 2 runs 07-08 to 07-16). Dividing by that span measures the
# experimenter's calendar, not the animal. Instead, consecutive trial starts
# more than SCREEN_TIME_MAX_GAP_S apart are treated as the animal not being at
# the screen and excluded; what remains is summed.
#
# On Uva's log that turns a flat line into a signal: 64 / 83 / 121 / 116 / 107
# / 118 HCHR per hour at N = 2 / 3 / 4 / 5 / 6 / 8.
SCREEN_TIME_MAX_GAP_S       = 300.0  # >5 min between trial starts = off-screen
DEFAULT_ITI_FALLBACK_S      = 16.0   # median Phase 4 inter-trial interval

ABANDON_RATE_THRESHOLD      = 0.35   # C1: >35% of HCHR trials at N abandoned
ENGAGEMENT_DROP_THRESHOLD   = 0.60   # C2: >=60% lower rate than at N-1
ENGAGEMENT_FLOOR_RATE       = 15.0   # C3: <15 HCHR trials per hour of screen time
CUMULATIVE_DROP_THRESHOLD   = 0.60   # C4: >=60% below the best N observed
STALL_SESSION_LIMIT         = 4      # C5: >=4 sessions at the current N w/o passing

# A rate computed over a few minutes is noise. C2/C3/C4 are skipped at any N
# with less screen time than this; C1 (abandonment) and C5 (stall) still apply.
ENGAGEMENT_MIN_SCREEN_MIN   = 20.0

# Why 15/hour: an animal that refuses every high-cost trial still produces a
# row per trial, because the trial times out and is logged. At a 20 s response
# timeout plus the ITI that is on the order of 100+ rows an hour. A rate of 15
# an hour means roughly four minutes of nothing per trial while the animal is
# at the screen — not slow working, but not working. This is a floor, not a
# working-rate target; the animal-relative comparisons (C2, C4) are what catch
# a decline from an individual's own normal pace.

# Retained only for the fallback path below, where timestamps are unusable.
ENGAGEMENT_FLOOR_FRACTION   = 0.50   # <50% of trials_per_session completed

PHASE4_TRIALS_PER_SESSION   = 50     # must match get_stage_config Phase 4

# ---- In-session breaking-point check (Phase 4 only) ----
# When True, the five criteria are evaluated after every completed Phase 4
# session. If one fires, the subject leaves the ramp immediately and moves to
# Phase 5 with the measured N — instead of grinding at a level they have
# already demonstrated they cannot sustain.
#
# Without this, the criteria are only ever consulted when resolving N for
# Phases 5-8, so an animal stuck mid-ramp (e.g. failing to reach the 75% bar
# at N=4) would stay there indefinitely with nothing to release them.
PHASE4_INSESSION_BP_CHECK = True

# A single bad session should not end the ramp. The current N must have at
# least this many completed sessions before the in-session check can fire.
# (Criterion 5 already requires 4 sessions by its own definition.)
PHASE4_BP_MIN_SESSIONS_AT_N = 2

# Fallback N used when the Phase 4 data cannot be read at all (missing file,
# corrupt log, or no Phase 4 HCHR rows). This is deliberately NOT tied to
# PHASE4_N_VALUES[-1]: when the ramp cap was raised from 6 to 12, that
# expression silently doubled the fallback, so a subject with an unreadable
# log would have been handed the hardest possible requirement with no warning.
# A mid-range value is the safer guess, and the NO_DATA status is reported
# loudly so it is never mistaken for a measurement.
NO_DATA_FALLBACK_N = 6

# Single source of truth for the Phase 4 ramp. get_stage_config, advance_stage
# and the breaking-point analysis all read this, so they cannot disagree.
# Extended from [2,3,4,5,6] to cap 12 after Yoshi ceilinged at 6.
# Substage index -> N:  0->2 1->3 2->4 3->5 4->6 5->8 6->10 7->12
PHASE4_N_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]

# Per-animal manual override of the high-cost touch requirement. Takes
# precedence over the computed value. Use for animals whose breaking point was
# measured outside the normal ramp (e.g. via PROBE mode) or who ceilinged.
#   "Yoshi": 10,
HIGH_COST_N_OVERRIDES = {
}


def _screen_time_seconds(times, max_gap_s=SCREEN_TIME_MAX_GAP_S):
    """
    Seconds of screen time represented by a sorted list of trial-start times.

    Sums the intervals between consecutive trial starts, discarding any gap
    longer than max_gap_s (the program was not running, or the animal was not
    there), and adds one median interval for the final trial, which has no
    successor to measure against. Returns (seconds, n_bouts, median_interval).

    n_bouts is the number of continuous stretches, i.e. discarded gaps + 1. A
    high bout count at a given N means the animal needed many visits to fill
    the quota, which is worth seeing even though it does not feed a criterion.
    """
    times = [t for t in times if pd.notna(t)]
    if len(times) < 2:
        return 0.0, (1 if times else 0), DEFAULT_ITI_FALLBACK_S

    times = sorted(times)
    kept, gaps = [], 0
    for a, b in zip(times, times[1:]):
        d = (b - a).total_seconds()
        if 0.0 <= d <= max_gap_s:
            kept.append(d)
        else:
            gaps += 1                      # off-screen, or a clock going backwards

    if not kept:
        return 0.0, gaps + 1, DEFAULT_ITI_FALLBACK_S

    kept.sort()
    mid = len(kept) // 2
    median = kept[mid] if len(kept) % 2 else (kept[mid - 1] + kept[mid]) / 2.0
    return sum(kept) + median, gaps + 1, median


def _phase4_per_n_session_stats(df):
    """
    Per-N HCHR stats for Phase 4.

    Returns { N: {"sessions", "n_sessions", "attempted", "abandoned",
                  "substages", "screen_min", "bouts", "rate", "median_iti_s",
                  "timed"} }

    "Attempted"  = every HCHR row at that N (engagement).
    "Abandoned"  = the incorrect subset (failure among attempts).
    "sessions"   = HCHR trials per session — kept for reporting and for the
                   fallback path, but NOT what the criteria use (see the note
                   on the engagement constants above: it is flat by
                   construction).
    "rate"       = HCHR trials per hour of screen time, the live measure.
    "timed"      = whether this N has usable timestamps and enough screen time
                   for the rate to mean anything.

    Screen time is computed over ALL Phase 4 trials in the session, not just
    HCHR ones: the interleaved PLAIN trials are part of the animal's time at
    the screen, so excluding them would inflate the rate.
    """
    ph4_all = df[df["Phase #"] == 4].copy()
    if ph4_all.empty:
        return {}

    # Date + Time Stim On. Time is HH:MM:SS with no date of its own, so the
    # Date column is what keeps a session that crosses midnight in order.
    try:
        ph4_all["_t"] = pd.to_datetime(
            ph4_all["Date"].astype(str) + " " + ph4_all["Time Stim On"].astype(str),
            format="%m-%d-%Y %H:%M:%S", errors="coerce")
    except Exception:
        ph4_all["_t"] = pd.NaT

    ph4_all["Cue Touches Required"] = pd.to_numeric(
        ph4_all["Cue Touches Required"], errors="coerce")
    ph4_all["Correct Trial?"] = pd.to_numeric(
        ph4_all["Correct Trial?"], errors="coerce")

    ph4 = ph4_all[ph4_all["Trial Type"] == "HCHR"]
    if ph4.empty:
        return {}

    stats = {}
    for n in PHASE4_N_VALUES:
        n_df = ph4[ph4["Cue Touches Required"] == n]
        if n_df.empty:
            continue

        keys        = list(n_df.groupby(["Substage #", "Session #"]).size().index)
        per_session = n_df.groupby(["Substage #", "Session #"]).size().tolist()

        screen_s, bouts, itis = 0.0, 0, []
        for sub, sess in keys:
            sess_rows = ph4_all[(ph4_all["Substage #"] == sub) &
                                (ph4_all["Session #"] == sess)]
            s, b, med = _screen_time_seconds(sess_rows["_t"].tolist())
            screen_s += s
            bouts    += b
            if s > 0:
                itis.append(med)

        screen_min = screen_s / 60.0
        rate  = (len(n_df) / (screen_s / 3600.0)) if screen_s > 0 else None
        timed = rate is not None and screen_min >= ENGAGEMENT_MIN_SCREEN_MIN

        stats[n] = {
            "sessions":     per_session,
            "n_sessions":   len(per_session),
            "attempted":    int(len(n_df)),
            "abandoned":    int((n_df["Correct Trial?"] == 0).sum()),
            "substages":    set(n_df["Substage #"].dropna().astype(int).tolist()),
            "screen_min":   screen_min,
            "bouts":        bouts,
            "rate":         rate,
            "median_iti_s": (sum(itis) / len(itis)) if itis else None,
            "timed":        bool(timed),
        }
    return stats


def analyze_breaking_point(log_path, monkey_name=None,
                            trials_per_session=PHASE4_TRIALS_PER_SESSION):
    """
    Full breaking-point analysis.

    Returns {"status", "high_cost_n", "breaking_point", "criterion",
             "detail", "per_n"}

    status:
      MEASURED             a criterion fired; high_cost_n == breaking_point
      CEILING_NOT_MEASURED completed the whole ramp without breaking. The value
                           is the top of the ramp, which is a FLOOR on the true
                           breaking point, not an estimate. NEEDS ATTENTION.
      OVERRIDE             taken from HIGH_COST_N_OVERRIDES
      NO_DATA              no Phase 4 HCHR rows
    """
    if monkey_name and monkey_name in HIGH_COST_N_OVERRIDES:
        n = int(HIGH_COST_N_OVERRIDES[monkey_name])
        return {"status": "OVERRIDE", "high_cost_n": n, "breaking_point": None,
                "criterion": None, "per_n": {},
                "detail": f"Manual override from HIGH_COST_N_OVERRIDES: N={n}."}

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        return {"status": "NO_DATA", "high_cost_n": NO_DATA_FALLBACK_N,
                "breaking_point": None, "criterion": None, "per_n": {},
                "detail": f"Could not read log: {e}"}

    stats = _phase4_per_n_session_stats(df)
    if not stats:
        return {"status": "NO_DATA", "high_cost_n": NO_DATA_FALLBACK_N,
                "breaking_point": None, "criterion": None, "per_n": {},
                "detail": "No Phase 4 HCHR data found."}

    ns        = sorted(stats.keys())
    prev_eng  = None
    best_eng  = 0.0
    current_n = max(ns)          # C3/C5 only apply at the animal's current level

    # Engagement is a rate (HCHR trials per hour of screen time) wherever the
    # timestamps support one. Where they do not — an old log, a log with an
    # unexpected time format, or an N with less than ENGAGEMENT_MIN_SCREEN_MIN
    # of screen time — fall back per-N to the old trials-per-session count so
    # the function still returns something, and say so in the detail text.
    floor_count = ENGAGEMENT_FLOOR_FRACTION * trials_per_session

    def engagement(s):
        """-> (value, floor, unit, is_rate)"""
        if s["timed"]:
            return s["rate"], ENGAGEMENT_FLOOR_RATE, "HCHR/h of screen time", True
        avg = sum(s["sessions"]) / len(s["sessions"])
        return avg, floor_count, "trials/session (untimed fallback)", False

    for n in ns:
        s = stats[n]
        ab = s["abandoned"] / s["attempted"] if s["attempted"] else 0.0
        eng, floor, unit, is_rate = engagement(s)

        def fire(crit, msg):
            # High-cost N is set AT the breaking point, not one below it.
            return {"status": "MEASURED", "high_cost_n": n, "breaking_point": n,
                    "criterion": crit, "per_n": stats,
                    "detail": (f"Breaking point at N={n} via {crit}. {msg} "
                               f"High-cost touch requirement = {n}.")}

        if ab > ABANDON_RATE_THRESHOLD:                                    # C1
            return fire("abandonment",
                        f"{ab:.1%} of {s['attempted']} trials abandoned "
                        f"(> {ABANDON_RATE_THRESHOLD:.0%}).")

        # C2/C3/C4 compare engagement rates, so they are only meaningful
        # between levels measured the same way. A level without a usable rate
        # is reported but does not end the ramp on an engagement argument.
        rate_ok = s["timed"] or not any(stats[m]["timed"] for m in ns)

        if rate_ok and prev_eng and prev_eng > 0:                          # C2
            drop = (prev_eng - eng) / prev_eng
            if drop >= ENGAGEMENT_DROP_THRESHOLD:
                return fire("engagement drop vs previous N",
                            f"{eng:.1f} vs {prev_eng:.1f} {unit} at the "
                            f"previous N ({drop:.1%} drop).")

        if rate_ok and n == current_n and eng < floor:                     # C3
            return fire("absolute engagement floor",
                        f"only {eng:.1f} {unit}, below the floor of {floor:.1f} "
                        f"({s['screen_min']:.0f} min of screen time across "
                        f"{s['bouts']} bout(s)).")

        if rate_ok and best_eng > 0:                                       # C4
            cum = (best_eng - eng) / best_eng
            if cum >= CUMULATIVE_DROP_THRESHOLD:
                return fire("cumulative drop from best",
                            f"{eng:.1f} {unit} vs best of {best_eng:.1f} "
                            f"({cum:.1%} below peak).")

        if n == current_n and s["n_sessions"] >= STALL_SESSION_LIMIT:      # C5
            return fire("stall",
                        f"{s['n_sessions']} sessions at this N without passing "
                        f"(limit {STALL_SESSION_LIMIT}).")

        if rate_ok:
            prev_eng = eng
            best_eng = max(best_eng, eng)

    top = max(ns)
    return {"status": "CEILING_NOT_MEASURED", "high_cost_n": top,
            "breaking_point": None, "criterion": None, "per_n": stats,
            "detail": (f"No criterion fired across N={ns}. The subject completed "
                       f"the entire ramp without breaking. N={top} is a FLOOR on "
                       f"the true breaking point, not an estimate.")}


def print_breaking_point_report(res, monkey_name=""):
    """Print the result, making a ceiling outcome impossible to miss."""
    print("-" * 70)
    print(f"BREAKING POINT{' — ' + monkey_name if monkey_name else ''}")
    print(f"  status      : {res['status']}")
    print(f"  high-cost N : {res['high_cost_n']}")
    if res.get("criterion"):
        print(f"  criterion   : {res['criterion']}")
    print(f"  {res['detail']}")

    per_n = res.get("per_n") or {}
    if per_n:
        print()
        print("  per-N profile  (engagement = HCHR trials per hour of screen time;")
        print("                  trials/sess is flat by design and is shown for reference only)")
        print(f"    {'N':>3} {'sess':>5} {'HCHR':>5} {'trials/sess':>11} "
              f"{'screen min':>10} {'bouts':>5} {'HCHR/h':>7} {'med ITI':>8} {'abandon':>8}")
        for n in sorted(per_n):
            s    = per_n[n]
            tps  = sum(s["sessions"]) / len(s["sessions"]) if s["sessions"] else 0.0
            ab   = s["abandoned"] / s["attempted"] if s["attempted"] else 0.0
            rate = s.get("rate")
            iti  = s.get("median_iti_s")
            flag = "" if s.get("timed") else "  (rate not used: too little screen time)"
            print(f"    {n:>3} {s['n_sessions']:>5} {s['attempted']:>5} {tps:>11.1f} "
                  f"{s.get('screen_min', 0.0):>10.0f} {s.get('bouts', 0):>5} "
                  f"{(f'{rate:.1f}' if rate else '—'):>7} "
                  f"{(f'{iti:.0f}s' if iti else '—'):>8} {ab:>7.1%}{flag}")
    if res["status"] == "NO_DATA":
        print()
        print("  " + "!" * 62)
        print("  !!  WARNING: NO PHASE 4 DATA — N IS A FALLBACK, NOT A MEASUREMENT.")
        print(f"  !!  Using NO_DATA_FALLBACK_N = {res['high_cost_n']}.")
        print("  !!  Check that this subject's training log is present and readable")
        print("  !!  before collecting any data with this value.")
        print("  " + "!" * 62)
    if res["status"] == "CEILING_NOT_MEASURED":
        print()
        print("  " + "!" * 62)
        print("  !!  WARNING: BREAKING POINT WAS NOT ACTUALLY MEASURED.")
        print("  !!  This animal never broke within the ramp, so the value above")
        print("  !!  is only a lower bound. Cost sensitivity may be at ceiling and")
        print("  !!  the healthy baseline may be uninformative.")
        print("  !!  -> Run PROBE mode to measure the real value, then set")
        print("  !!     HIGH_COST_N_OVERRIDES for this animal.")
        print("  " + "!" * 62)
    print("-" * 70)


def find_high_cost_touch_requirement(log_path, fallback=None, monkey_name=None):
    """Backward-compatible wrapper returning just the integer N."""
    res = analyze_breaking_point(log_path, monkey_name=monkey_name)
    print_breaking_point_report(res, monkey_name or "")
    return res["high_cost_n"]


# ============================================================
# PASS CHECKING
# ============================================================
def pass_phase_check(log_path, phase_num, substage_idx,
                     pass_threshold, required_trials, required_sessions):
    df = pd.read_csv(log_path)
    stage_df = df[
        (df["Phase #"] == phase_num) &
        (df["Substage #"] == substage_idx)
    ].copy()

    if stage_df.empty:
        print(f"No data for Phase {phase_num}, Substage {substage_idx}.")
        return False

    final_df = stage_df[stage_df["Trial #"] == required_trials].copy()
    if final_df.empty:
        print(f"No completed sessions for Phase {phase_num}, Substage {substage_idx}.")
        return False

    final_df = final_df.sort_values(["Session #", "Trial #"])
    final_df = final_df.groupby("Session #", as_index=False).last()

    if len(final_df) < required_sessions:
        print(f"Phase {phase_num} sub{substage_idx}: only {len(final_df)} session(s), need {required_sessions}.")
        return False

    last_sessions = final_df.sort_values("Session #").tail(required_sessions)
    session_nums = last_sessions["Session #"].tolist()
    expected = list(range(session_nums[0], session_nums[0] + required_sessions))
    if session_nums != expected:
        print(f"Phase {phase_num} sub{substage_idx}: sessions not consecutive: {session_nums}.")
        return False

    pct_values = last_sessions["Current % Correct"].tolist()
    if all(pct >= pass_threshold for pct in pct_values):
        print(f"Phase {phase_num} sub{substage_idx} PASSED. Sessions {session_nums}, values {[round(p,1) for p in pct_values]}.")
        return True

    print(f"Phase {phase_num} sub{substage_idx} not passed. Values: {[round(p,1) for p in pct_values]}.")
    return False


PHASE8_CUE_THRESHOLDS = {
    "HCHR": 65,    # ≥65% completion rate across 3 consecutive sessions
    "HCLR": 35,    # ≥35%
    "LCLR": 75,    # ≥75%  (was 80% in old Phase 7 — updated per protocol revision)
    "LCHR": 80,    # ≥80%  (new cue, highest bar — high reward, low cost)
}
PHASE8_CONSECUTIVE_SESSIONS_REQUIRED = 3


def _phase8_complete_sessions(log_path, required_trials):
    """
    Return a DataFrame of all Phase 8 rows belonging to fully-completed
    sessions (exactly `required_trials` rows). Phase 8 has no substages.
    """
    df = pd.read_csv(log_path)
    ph8 = df[df["Phase #"] == 8].copy()
    if ph8.empty:
        return ph8
    session_counts = ph8.groupby("Session #")["Trial #"].count()
    complete_sessions = session_counts[session_counts == required_trials].index.tolist()
    ph8 = ph8[ph8["Session #"].isin(complete_sessions)]
    return ph8.sort_values(["Session #", "Trial #"])


def _phase8_per_cue_session_rates(ph8_df):
    """
    For each cue type and each completed Phase 8 session, compute
    completion rate = correct / attempted.
    Returns { cue: { session_num: rate_or_None } }.
    rate is None if the cue had zero attempted trials in that session.
    """
    rates = {cue: {} for cue in PHASE8_CUE_THRESHOLDS}
    for cue in PHASE8_CUE_THRESHOLDS:
        cue_df = ph8_df[ph8_df["Trial Type"] == cue]
        for session_num, sess_df in cue_df.groupby("Session #"):
            attempted = len(sess_df)
            if attempted == 0:
                rates[cue][session_num] = None
            else:
                correct = pd.to_numeric(sess_df["Correct Trial?"], errors="coerce").sum()
                rates[cue][session_num] = correct / attempted
    return rates


def _find_qualifying_window(session_nums_sorted, rate_by_session, threshold_pct, n_consecutive):
    """
    Scan forward through session_nums_sorted (chronological order) for the
    FIRST run of n_consecutive CONSECUTIVE session numbers where
    rate_by_session[session] * 100 >= threshold_pct for every session in the
    run. Returns the winning window (list of session numbers) or None.

    "Consecutive" means consecutive SESSION NUMBERS (e.g. 4,5,6), not just
    consecutive list entries — mirrors the same consecutiveness check used in
    pass_phase_check.
    """
    for start_idx in range(len(session_nums_sorted) - n_consecutive + 1):
        window = session_nums_sorted[start_idx:start_idx + n_consecutive]
        expected = list(range(window[0], window[0] + n_consecutive))
        if window != expected:
            continue
        rates_in_window = [rate_by_session.get(s) for s in window]
        if any(r is None for r in rates_in_window):
            continue
        if all((r * 100) >= threshold_pct for r in rates_in_window):
            return window
    return None


def pass_phase8_check(log_path, required_trials=100):
    """
    Phase 8 pass logic: HCHR, HCLR, LCLR, and LCHR each have their own
    independent completion-rate threshold (PHASE8_CUE_THRESHOLDS) and must
    each independently hold that rate across some run of
    PHASE8_CONSECUTIVE_SESSIONS_REQUIRED consecutive sessions. The four
    qualifying windows need not be the same sessions or concurrent. Once a
    cue has found its qualifying window it is permanently satisfied — a later
    session where that cue's rate drops does not un-satisfy it.

    Returns (passed: bool, details: dict) where details reports, per cue,
    whether it has been satisfied and which session window satisfied it.
    """
    ph8_df = _phase8_complete_sessions(log_path, required_trials)
    if ph8_df.empty:
        print("Phase 8: no completed sessions yet.")
        return False, {cue: {"satisfied": False, "window": None} for cue in PHASE8_CUE_THRESHOLDS}

    rates = _phase8_per_cue_session_rates(ph8_df)
    all_session_nums = sorted(ph8_df["Session #"].unique().tolist())

    details = {}
    all_satisfied = True
    for cue, threshold_pct in PHASE8_CUE_THRESHOLDS.items():
        window = _find_qualifying_window(
            all_session_nums, rates[cue], threshold_pct,
            PHASE8_CONSECUTIVE_SESSIONS_REQUIRED,
        )
        details[cue] = {"satisfied": window is not None, "window": window}
        if window is None:
            all_satisfied = False

    for cue, info in details.items():
        if info["satisfied"]:
            print(f"Phase 8: {cue} satisfied (>={PHASE8_CUE_THRESHOLDS[cue]}% across sessions {info['window']}).")
        else:
            print(f"Phase 8: {cue} not yet satisfied "
                  f"(needs {PHASE8_CONSECUTIVE_SESSIONS_REQUIRED} consecutive sessions "
                  f">={PHASE8_CUE_THRESHOLDS[cue]}%).")

    if all_satisfied:
        print("Phase 8 PASSED — all four cue types independently satisfied their completion-rate criteria.")
    return all_satisfied, details


# ============================================================
# FILE / FOLDER HELPERS
# ============================================================
def get_monkey_folder(monkey_name):
    """
    Return (and create if needed) the per-monkey folder.
    All files for a monkey live inside: ./{monkey_name}/
    """
    folder = Path(monkey_name)
    folder.mkdir(exist_ok=True)
    return folder


def get_log_paths(monkey_name):
    """
    Return (training_log_path, session_log_path) inside the monkey folder.
    """
    folder = get_monkey_folder(monkey_name)
    training_log = folder / f"{monkey_name}_simplified_training_log.csv"
    session_log  = folder / f"{monkey_name}_session_log.csv"
    return training_log, session_log


# ============================================================
# SESSION LOG  (start/stop times + summary)
# ============================================================
SESSION_LOG_HEADERS = [
    # Identification
    "Date",
    "Monkey",
    "Session #",
    "Phase # (start)",
    "Substage # (start)",
    "Phase Description (start)",
    "Phase # (end)",
    "Substage # (end)",
    "Phase Description (end)",
    # Timing
    "Program Start Time",
    "Program End Time",
    "Total Duration (hh:mm:ss)",
    "Total Duration (s)",
    # Trial counts
    "Trials Completed",
    "Correct Trials",
    "Incorrect Trials",
    "% Correct",
    # Per-type breakdown
    "PLAIN Trials",
    "PLAIN Correct",
    "PLAIN % Correct",
    "HCHR Trials",
    "HCHR Correct",
    "HCHR % Correct",
    "HCHR N (touches required)",
    "HCLR Trials",
    "HCLR Correct",
    "HCLR % Correct",
    "LCLR Trials",
    "LCLR Correct",
    "LCLR % Correct",
    "LCHR Trials",
    "LCHR Correct",
    "LCHR % Correct",
    # Errors / timeouts
    "Timeouts (HCHR start)",
    "Timeouts (HCHR incomplete)",
    "Timeouts (HCLR start)",
    "Timeouts (HCLR incomplete)",
    "Timeouts (LCLR)",
    "Timeouts (LCHR)",
    "Total Timeouts",
    # Pellets
    "Pellets Dispensed (total)",
    "Pellets Confirmed Delivered",
    "Pellets Unconfirmed",
    # Phase advancement
    "Phase Passed?",
    "Advanced To Phase",
    "Advanced To Substage",
    # Phase 8 only: per-cue rolling-window satisfaction status (blank for all other phases)
    "Phase 8: HCHR Satisfied?",
    "Phase 8: HCLR Satisfied?",
    "Phase 8: LCLR Satisfied?",
    "Phase 8: LCHR Satisfied?",
    # Exit
    "Exit Reason",
]

# ---- Session-log header history ----
# Same rule as the collection logs: never insert a column mid-list, append at
# the END, and record the previous list here.
#
# v1 (38 columns) is everything that existed before HCLR and LCHR were added as
# cues and before the Phase 8 per-cue rolling-window columns. Those additions
# landed mid-list, so a file written by both versions mixes 38- and 51-field
# rows under a 38-field header and pandas refuses to read it at all
# ("Expected 38 fields in line 23, saw 51"). Every animal whose session log
# predates the HCLR/LCHR work is in that state.
#
# Defined by exclusion from the current header so the two lists cannot drift
# apart, and verified below against the real 38-column order.
_SESSION_LOG_V1 = [c for c in SESSION_LOG_HEADERS if c not in (
    "HCLR Trials", "HCLR Correct", "HCLR % Correct",
    "LCHR Trials", "LCHR Correct", "LCHR % Correct",
    "Timeouts (HCLR start)", "Timeouts (HCLR incomplete)", "Timeouts (LCHR)",
    "Phase 8: HCHR Satisfied?", "Phase 8: HCLR Satisfied?",
    "Phase 8: LCLR Satisfied?", "Phase 8: LCHR Satisfied?",
)]
assert len(_SESSION_LOG_V1) == 38, len(_SESSION_LOG_V1)

SESSION_LOG_HISTORY = [_SESSION_LOG_V1, SESSION_LOG_HEADERS]


def _ensure_session_log(session_log_path):
    """
    Create the session log if absent, and migrate an older header in place
    (after a timestamped backup) if one is found.

    This used to write the header only when the file did not exist, which is
    the same bug that made the collection logs unparseable: as columns were
    added, DictWriter kept appending wider rows under the original narrow
    header. Rows are re-keyed BY COLUMN NAME via SESSION_LOG_HISTORY, so values
    cannot land under the wrong heading even though the new cue columns were
    inserted mid-list.
    """
    _ensure_csv(session_log_path, SESSION_LOG_HEADERS, SESSION_LOG_HISTORY)


def write_session_log(
    session_log_path,
    monkey_name,
    session_num,
    phase_start, substage_start,
    phase_end, substage_end,
    start_dt, end_dt,
    trial_stats,          # dict with per-trial accumulators (see run_session)
    phase_passed,
    advanced_to,          # (phase, substage) tuple or None
    exit_reason,
    phase8_details=None,  # dict from pass_phase8_check, only set when phase_end == 8
):
    _ensure_session_log(session_log_path)

    duration_s = (end_dt - start_dt).total_seconds()
    h = int(duration_s // 3600)
    m = int((duration_s % 3600) // 60)
    s = int(duration_s % 60)
    duration_hms = f"{h:02d}:{m:02d}:{s:02d}"

    def pct(correct, total):
        return f"{correct / total * 100:.1f}" if total > 0 else "N/A"

    ts = trial_stats  # shorthand

    phase_desc_start = ""
    phase_desc_end   = ""
    try:
        phase_desc_start = get_stage_config(phase_start, substage_start, monkey_name)["description"]
    except Exception:
        pass
    try:
        phase_desc_end = get_stage_config(phase_end, substage_end, monkey_name)["description"]
    except Exception:
        pass

    adv_phase = advanced_to[0] if advanced_to else "N/A"
    adv_sub   = advanced_to[1] if advanced_to else "N/A"

    row = {
        "Date":                        end_dt.strftime("%m-%d-%Y"),
        "Monkey":                      monkey_name,
        "Session #":                   session_num,
        "Phase # (start)":             phase_start,
        "Substage # (start)":          substage_start,
        "Phase Description (start)":   phase_desc_start,
        "Phase # (end)":               phase_end,
        "Substage # (end)":            substage_end,
        "Phase Description (end)":     phase_desc_end,
        "Program Start Time":          start_dt.strftime("%H:%M:%S"),
        "Program End Time":            end_dt.strftime("%H:%M:%S"),
        "Total Duration (hh:mm:ss)":   duration_hms,
        "Total Duration (s)":          f"{duration_s:.1f}",
        "Trials Completed":            ts["total_trials"],
        "Correct Trials":              ts["correct_trials"],
        "Incorrect Trials":            ts["total_trials"] - ts["correct_trials"],
        "% Correct":                   pct(ts["correct_trials"], ts["total_trials"]),
        "PLAIN Trials":                ts["plain_total"],
        "PLAIN Correct":               ts["plain_correct"],
        "PLAIN % Correct":             pct(ts["plain_correct"], ts["plain_total"]),
        "HCHR Trials":                 ts["hchr_total"],
        "HCHR Correct":                ts["hchr_correct"],
        "HCHR % Correct":              pct(ts["hchr_correct"], ts["hchr_total"]),
        "HCHR N (touches required)":   ts["hchr_n"],
        "HCLR Trials":                 ts["hclr_total"],
        "HCLR Correct":                ts["hclr_correct"],
        "HCLR % Correct":              pct(ts["hclr_correct"], ts["hclr_total"]),
        "LCLR Trials":                 ts["lclr_total"],
        "LCLR Correct":                ts["lclr_correct"],
        "LCLR % Correct":              pct(ts["lclr_correct"], ts["lclr_total"]),
        "LCHR Trials":                 ts["lchr_total"],
        "LCHR Correct":                ts["lchr_correct"],
        "LCHR % Correct":              pct(ts["lchr_correct"], ts["lchr_total"]),
        "Timeouts (HCHR start)":       ts["timeout_hchr_start"],
        "Timeouts (HCHR incomplete)":  ts["timeout_hchr_incomplete"],
        "Timeouts (HCLR start)":       ts["timeout_hclr_start"],
        "Timeouts (HCLR incomplete)":  ts["timeout_hclr_incomplete"],
        "Timeouts (LCLR)":             ts["timeout_lclr"],
        "Timeouts (LCHR)":             ts["timeout_lchr"],
        "Total Timeouts":              (
            ts["timeout_hchr_start"] + ts["timeout_hchr_incomplete"] +
            ts["timeout_hclr_start"] + ts["timeout_hclr_incomplete"] +
            ts["timeout_lclr"] + ts["timeout_lchr"]
        ),
        "Pellets Dispensed (total)":   ts["pellets_dispensed"],
        "Pellets Confirmed Delivered": ts["pellets_confirmed"],
        "Pellets Unconfirmed":         ts["pellets_unconfirmed"],
        "Phase Passed?":               "Yes" if phase_passed else "No",
        "Advanced To Phase":           adv_phase,
        "Advanced To Substage":        adv_sub,
        "Phase 8: HCHR Satisfied?":    ("Yes" if phase8_details["HCHR"]["satisfied"] else "No") if phase8_details else "",
        "Phase 8: HCLR Satisfied?":    ("Yes" if phase8_details["HCLR"]["satisfied"] else "No") if phase8_details else "",
        "Phase 8: LCLR Satisfied?":    ("Yes" if phase8_details["LCLR"]["satisfied"] else "No") if phase8_details else "",
        "Phase 8: LCHR Satisfied?":    ("Yes" if phase8_details["LCHR"]["satisfied"] else "No") if phase8_details else "",
        "Exit Reason":                 exit_reason,
    }

    with open(session_log_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SESSION_LOG_HEADERS).writerow(row)


# ============================================================
# RESUME STATE
# ============================================================
def get_resume_state(file_path, headers, monkey_name=None):
    if not file_path.exists():
        with open(file_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()
        print(f"New log: {file_path}. Starting Phase 1, Session 1.")
        return 1, 0, 1, 0, 0

    # The trial log carries the same latent header-drift bug the session and
    # collection logs had: the header is written once at creation, while every
    # later append uses the current HEADERS. All existing trial logs happen to
    # be consistent today (18 columns throughout), so this is a no-op — it is
    # here so that the next column added to HEADERS repairs old logs instead of
    # making them unreadable. On a healthy file _ensure_csv returns without
    # touching a byte; if a column is ever appended, record the previous list
    # as a TRAINING_TRIAL_HISTORY version and pass it here.
    _ensure_csv(file_path, headers)

    with open(file_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return 1, 0, 1, 0, 0

    last = rows[-1]
    phase   = int(last["Phase #"])
    sub     = int(last["Substage #"])
    session = int(last["Session #"])
    trial   = int(last["Trial #"])

    try:
        config = get_stage_config(phase, sub, monkey_name)
        tps = config["trials_per_session"]
    except Exception:
        tps = None

    if tps is not None and trial >= tps:
        print(f"Resuming. Phase {phase} sub{sub} session {session} was complete. Starting session {session+1}.")
        return phase, sub, session + 1, 0, 0

    # BUGFIX: starting_correct used to be reverse-derived from the rounded
    # "Current % Correct" column (round((pct/100) * trial)), which loses
    # precision and can be off by 1 or more trials, especially early in a
    # session. The exact correct-trial count is already in the CSV — sum the
    # "Correct Trial?" column for this session directly instead of
    # reconstructing it from a lossy percentage.
    same_session_rows = [
        r for r in rows
        if int(r["Phase #"]) == phase
        and int(r["Substage #"]) == sub
        and int(r["Session #"]) == session
    ]
    starting_correct = sum(int(r["Correct Trial?"]) for r in same_session_rows)

    print(f"Resuming mid-session. Phase {phase} sub{sub} session {session} trial {trial}. "
          f"Exact correct count so far: {starting_correct}/{trial}. Next: trial {trial+1}.")
    return phase, sub, session, trial, starting_correct


# ============================================================
# MAIN TRIAL FUNCTION
# ============================================================
def run_session(clock, screen, screen_info, log_path,
                phase_num, substage_idx, session_num,
                monkey_name,
                trial_idx=0, starting_correct_trials=0):
    config = get_stage_config(phase_num, substage_idx, monkey_name)
    date_str = datetime.today().strftime("%m-%d-%Y")

    shortest_dim       = min(screen_info.current_w, screen_info.current_h)
    circle_center      = (screen_info.current_w // 2, screen_info.current_h // 2)
    trials_per_session = config["trials_per_session"]
    random_circle_size = config["random_circle_size"]
    size_fraction      = config["size_fraction"]
    trial_probs        = config["trial_probs"]
    hchr_n             = config["hchr_n"]
    if hchr_n is None:
        # Shared high-cost touch requirement for BOTH HCHR and HCLR, derived once
        # from this subject's Phase 4 breaking point (see
        # find_high_cost_touch_requirement). Both cue types use the same N; only
        # the pellet reward differs between them.
        hchr_n = find_high_cost_touch_requirement(log_path, monkey_name=monkey_name)
        print(f"Phase {phase_num}: resolved high-cost touch requirement N={hchr_n} from breaking-point analysis.")
    resp_timeout  = config["response_timeout_ms"]
    normal_iti    = config["normal_iti_ms"]
    penalty_iti   = config["penalty_iti_ms"]

    def get_radius():
        if random_circle_size:
            pct = random.randint(30, 90)
            return int(shortest_dim * (pct / 100) / 2)
        return int(shortest_dim * size_fraction / 2)

    circle_radius   = get_radius()
    # Pre-rasterise the cue shapes so the first trial does not pay the cost.
    # Phase 2 re-rolls the radius per trial, but it is PLAIN-only (no shapes),
    # so nothing is wasted there; any new radius is cached on first use anyway.
    prebuild_shape_cache(circle_radius)
    all_trial_types = build_trial_list(trial_probs, trials_per_session)
    remaining_types = all_trial_types[trial_idx:]

    cum_correct = starting_correct_trials
    pct_correct = (cum_correct / trial_idx * 100) if trial_idx > 0 else 0.0
    running     = True
    exit_reason = "completed"

    # ----------------------------------------------------------
    # Per-session accumulators for the session log summary
    # ----------------------------------------------------------
    trial_stats = {
        "total_trials":           trial_idx,       # include any already-done trials
        "correct_trials":         starting_correct_trials,
        "plain_total":            0,
        "plain_correct":          0,
        "hchr_total":             0,
        "hchr_correct":           0,
        "hchr_n":                 hchr_n,
        "hclr_total":             0,
        "hclr_correct":           0,
        "lclr_total":             0,
        "lclr_correct":           0,
        "lchr_total":             0,
        "lchr_correct":           0,
        "timeout_hchr_start":     0,
        "timeout_hchr_incomplete":0,
        "timeout_hclr_start":     0,
        "timeout_hclr_incomplete":0,
        "timeout_lclr":           0,
        "timeout_lchr":           0,
        "pellets_dispensed":      0,
        "pellets_confirmed":      0,
        "pellets_unconfirmed":    0,
    }

    # BUGFIX: when resuming mid-session (trial_idx > 0), the per-trial-type
    # breakdown above was previously left at all-zeros, even though
    # total_trials/correct_trials WERE seeded. That mismatch is what produced
    # "mixed" numbers in the session log: PLAIN+HCHR+HCLR+LCLR trial counts
    # would not sum to Trials Completed, and per-type Correct counts would be
    # too low relative to overall Correct Trials, for any session that had
    # been interrupted and resumed. Re-derive the exact breakdown for the
    # already-completed portion of this session from the CSV itself (which
    # has the precise Trial Type / Correct Trial? / Error Type for every row)
    # and fold it in before the loop starts adding the rest of the session.
    if trial_idx > 0:
        try:
            prior_df = pd.read_csv(log_path)
            prior_df = prior_df[
                (prior_df["Phase #"] == phase_num) &
                (prior_df["Substage #"] == substage_idx) &
                (prior_df["Session #"] == session_num)
            ]
            for ttype, total_key, correct_key in (
                ("PLAIN", "plain_total", "plain_correct"),
                ("HCHR",  "hchr_total",  "hchr_correct"),
                ("HCLR",  "hclr_total",  "hclr_correct"),
                ("LCLR",  "lclr_total",  "lclr_correct"),
                ("LCHR",  "lchr_total",  "lchr_correct"),
            ):
                t_df = prior_df[prior_df["Trial Type"] == ttype]
                trial_stats[total_key]   = int(len(t_df))
                trial_stats[correct_key] = int(t_df["Correct Trial?"].sum())

            err_counts = prior_df["Error Type"].value_counts()
            trial_stats["timeout_hchr_start"]      = int(err_counts.get("hchr_start_timeout", 0))
            trial_stats["timeout_hchr_incomplete"] = int(err_counts.get("hchr_incomplete_timeout", 0))
            trial_stats["timeout_hclr_start"]      = int(err_counts.get("hclr_start_timeout", 0))
            trial_stats["timeout_hclr_incomplete"] = int(err_counts.get("hclr_incomplete_timeout", 0))
            trial_stats["timeout_lclr"]            = int(err_counts.get("lclr_timeout", 0))
            trial_stats["timeout_lchr"]            = int(err_counts.get("lchr_timeout", 0))

            # Same reconciliation for pellet counts, using the per-trial
            # "Pellets Dispensed" / "Pellet Delivered?" columns already in the
            # training log.
            pellets_df = prior_df[pd.to_numeric(prior_df["Pellets Dispensed"], errors="coerce") > 0]
            trial_stats["pellets_dispensed"]   = int(pd.to_numeric(pellets_df["Pellets Dispensed"], errors="coerce").sum())
            trial_stats["pellets_confirmed"]   = int(
                pd.to_numeric(pellets_df.loc[pellets_df["Pellet Delivered?"] == "Yes", "Pellets Dispensed"], errors="coerce").sum()
            )
            trial_stats["pellets_unconfirmed"] = int(
                pd.to_numeric(pellets_df.loc[pellets_df["Pellet Delivered?"] != "Yes", "Pellets Dispensed"], errors="coerce").sum()
            )
        except Exception as e:
            print(f"Warning: could not reconstruct per-type breakdown for resumed session "
                  f"(falling back to zeros for the pre-resume portion): {e}")

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)

        while running and len(remaining_types) > 0:

            circle_touched = False
            draw_circle_only(screen, circle_center, circle_radius)
            t_stim     = time.perf_counter()
            t_stim_str = datetime.today().strftime('%H:%M:%S')

            while not circle_touched and running:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                        exit_reason = "escaped"
                        break
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if hit_circle(event.pos, circle_center, circle_radius):
                            circle_touched = True
                            break
                clock.tick(60)

            if not running:
                break

            t_touch        = time.perf_counter()
            t_touch_str    = datetime.today().strftime('%H:%M:%S')
            touch_circle_ms = (t_touch - t_stim) * 1000.0

            trial_type = remaining_types[0]

            correct_trial    = 0
            cue_touches_req  = "N/A"
            cue_touches_done = "N/A"
            cue_elapsed_ms   = "N/A"
            first_touch_rt   = "N/A"
            error_type       = "N/A"
            pellet_delivered = "N/A"
            n_pellets        = 0

            # ================================================================
            # PLAIN TRIAL
            # ================================================================
            if trial_type == "PLAIN":
                correct_trial    = 1
                cue_touches_req  = 0
                cue_touches_done = 0
                error_type       = None
                n_pellets        = 1
                trial_stats["plain_total"] += 1

            # ================================================================
            # HCHR TRIAL
            # ================================================================
            elif trial_type == "HCHR":
                screen.fill((0, 0, 0))
                pygame.display.flip()
                sleep_ms(CUE_APPEAR_DELAY_MS)
                pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

                success, touches_done, elapsed, ftr, err = run_hchr_cue(
                    screen, clock, circle_center, circle_radius,
                    hchr_n, resp_timeout
                )

                if err == "escaped":
                    running = False
                    exit_reason = "escaped"
                    break

                cue_touches_req  = hchr_n
                cue_touches_done = touches_done
                cue_elapsed_ms   = elapsed
                first_touch_rt   = ftr
                error_type       = err

                trial_stats["hchr_total"] += 1
                if err == "hchr_start_timeout":
                    trial_stats["timeout_hchr_start"] += 1
                elif err == "hchr_incomplete_timeout":
                    trial_stats["timeout_hchr_incomplete"] += 1

                if success:
                    correct_trial = 1
                    n_pellets     = 3
                    trial_stats["hchr_correct"] += 1

            # ================================================================
            # HCLR TRIAL  (High-Cost, Low-Reward)
            # Scalloped flower (red), same multi-touch mechanic and touch
            # requirement (hchr_n) as HCHR. Differs: shape/color and reward
            # (HCLR pays 1 pellet, HCHR pays 3).
            # ================================================================
            elif trial_type == "HCLR":
                screen.fill((0, 0, 0))
                pygame.display.flip()
                sleep_ms(CUE_APPEAR_DELAY_MS)
                pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

                success, touches_done, elapsed, ftr, err = run_hclr_cue(
                    screen, clock, circle_center, circle_radius,
                    hchr_n, resp_timeout
                )

                if err == "escaped":
                    running = False
                    exit_reason = "escaped"
                    break

                cue_touches_req  = hchr_n
                cue_touches_done = touches_done
                cue_elapsed_ms   = elapsed
                first_touch_rt   = ftr
                error_type       = err

                trial_stats["hclr_total"] += 1
                if err == "hclr_start_timeout":
                    trial_stats["timeout_hclr_start"] += 1
                elif err == "hclr_incomplete_timeout":
                    trial_stats["timeout_hclr_incomplete"] += 1

                if success:
                    correct_trial = 1
                    n_pellets     = 1
                    trial_stats["hclr_correct"] += 1

            # ================================================================
            # LCLR TRIAL  (Low-Cost, Low-Reward)
            # Triangle (soft pink). 1 touch, 1 pellet.
            # ================================================================
            elif trial_type == "LCLR":
                screen.fill((0, 0, 0))
                pygame.display.flip()
                sleep_ms(CUE_APPEAR_DELAY_MS)
                pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

                success, elapsed, ftr, err = run_lclr_cue(
                    screen, clock, circle_center, circle_radius, resp_timeout
                )

                if err == "escaped":
                    running = False
                    exit_reason = "escaped"
                    break

                cue_touches_req  = 1
                cue_touches_done = 1 if success else 0
                cue_elapsed_ms   = elapsed
                first_touch_rt   = ftr
                error_type       = err

                trial_stats["lclr_total"] += 1
                if err == "lclr_timeout":
                    trial_stats["timeout_lclr"] += 1

                if success:
                    correct_trial = 1
                    n_pellets     = 1
                    trial_stats["lclr_correct"] += 1

            # ================================================================
            # LCHR TRIAL  (Low-Cost, High-Reward)
            # Star (purple). 1 touch, 3 pellets — same effort as LCLR but
            # same reward as HCHR. Introduced in Phase 7 (Cue 4 intro).
            # ================================================================
            elif trial_type == "LCHR":
                screen.fill((0, 0, 0))
                pygame.display.flip()
                sleep_ms(CUE_APPEAR_DELAY_MS)
                pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

                success, elapsed, ftr, err = run_lchr_cue(
                    screen, clock, circle_center, circle_radius, resp_timeout
                )

                if err == "escaped":
                    running = False
                    exit_reason = "escaped"
                    break

                cue_touches_req  = 1
                cue_touches_done = 1 if success else 0
                cue_elapsed_ms   = elapsed
                first_touch_rt   = ftr
                error_type       = err

                trial_stats["lchr_total"] += 1
                if err == "lchr_timeout":
                    trial_stats["timeout_lchr"] += 1

                if success:
                    correct_trial = 1
                    n_pellets     = 3
                    trial_stats["lchr_correct"] += 1

            # ================================================================
            # OUTCOME
            # ================================================================
            if correct_trial == 1:
                screen.fill('#008000')
                pygame.display.flip()
                pellet_delivered = dispense_pellets(device, n_pellets)
                trial_stats["pellets_dispensed"] += n_pellets
                if pellet_delivered == "Yes":
                    trial_stats["pellets_confirmed"] += n_pellets
                else:
                    trial_stats["pellets_unconfirmed"] += n_pellets

                # Plain correct — track pellet after dispense call
                if trial_type == "PLAIN":
                    trial_stats["plain_correct"] += 1

                sleep_ms(300)
                pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION])
                screen.fill('#898989')
                pygame.display.flip()
                sleep_ms(normal_iti)
                pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION])
                write_to_adu(device, 'RK0')
            else:
                screen.fill('#321010')
                pygame.display.flip()
                sleep_ms(300)
                pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION])
                screen.fill('#898989')
                pygame.display.flip()
                sleep_ms(penalty_iti)
                pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION])

            # ================================================================
            # BOOKKEEPING
            # ================================================================
            trial_idx      += 1
            remaining_types = remaining_types[1:]
            cum_correct    += correct_trial
            pct_correct     = (cum_correct / trial_idx) * 100

            trial_stats["total_trials"]   = trial_idx
            trial_stats["correct_trials"] = cum_correct

            writer.writerow([
                date_str,
                phase_num,
                substage_idx,
                session_num,
                trial_idx,
                t_stim_str,
                t_touch_str,
                f"{touch_circle_ms:.1f}",
                trial_type,
                cue_touches_req,
                cue_touches_done,
                cue_elapsed_ms,
                first_touch_rt,
                correct_trial,
                f"{pct_correct:.2f}",
                n_pellets,
                pellet_delivered,
                error_type if error_type is not None else "None",
            ])
            f.flush()
            os.fsync(f.fileno())

            if trial_idx >= trials_per_session:
                running = False
                exit_reason = "completed"
                break

            circle_radius = get_radius()

    return pct_correct, trial_idx, exit_reason, trial_stats


# ============================================================
# MAIN ENTRY POINT
# ============================================================
# ============================================================
# CAMERA / HAND TRACKING
# ============================================================
# Hand-tracking rig is still being built. Leave CAMERA_RECORDING_ENABLED False
# until the hardware is in place. When True, the task writes video-sync
# metadata (sync-flash timestamps, video filename) and plays the sync flash at
# session start. All per-touch monotonic timestamps are logged either way, so
# turning this on later does not change the trial/touch data format.
CAMERA_RECORDING_ENABLED = False

# How the video filename is recorded. "prompt" asks on screen; "auto" derives
# it from the SessionUID; "none" leaves it blank.
CAMERA_FILENAME_MODE = "auto"
SYNC_FLASH_COUNT     = 3


def sync_flash(screen, clock, n_flashes=SYNC_FLASH_COUNT):
    """
    Full-screen white flashes at session start so an external camera can be
    aligned to the task clock offline. Returns monotonic timestamps of each
    flash onset. No-op returning [] when the camera is disabled.
    """
    if not CAMERA_RECORDING_ENABLED:
        return []
    stamps = []
    for _ in range(n_flashes):
        screen.fill((255, 255, 255)); pygame.display.flip()
        stamps.append(time.monotonic())
        sleep_ms(200)
        screen.fill((0, 0, 0)); pygame.display.flip()
        sleep_ms(300)
    return stamps


# ============================================================
# GENERIC TOUCHSCREEN MENUS
# ============================================================
def choice_menu_pygame(screen, clock, title, options, subtitle=""):
    """Generic fullscreen chooser. options = [(key, label), ...]. ESC -> None."""
    font_title = pygame.font.Font(None, 64)
    font_sub   = pygame.font.Font(None, 36)
    font_btn   = pygame.font.Font(None, 52)
    font_key   = pygame.font.Font(None, 32)

    W, H = screen.get_width(), screen.get_height()
    pad, top = 26, 150
    n = len(options)
    btn_h = min(110, max(58, (H - top - pad * (n + 1)) // max(n, 1)))
    btn_w = int(W * 0.7)
    x0 = (W - btn_w) // 2
    rects = [pygame.Rect(x0, top + i * (btn_h + pad), btn_w, btn_h) for i in range(n)]

    palette = [(52, 152, 219), (46, 204, 113), (155, 89, 182), (230, 126, 34),
               (231, 76, 60), (26, 188, 156), (241, 196, 15), (149, 165, 166)]

    while True:
        mouse = pygame.mouse.get_pos()
        screen.fill((25, 25, 30))
        t = font_title.render(title, True, (255, 255, 255))
        screen.blit(t, t.get_rect(center=(W // 2, 55)))
        if subtitle:
            s = font_sub.render(subtitle, True, (170, 170, 180))
            screen.blit(s, s.get_rect(center=(W // 2, 105)))
        for i, ((key, label), rect) in enumerate(zip(options, rects)):
            base = palette[i % len(palette)]
            col = tuple(max(0, c - 40) for c in base) if rect.collidepoint(mouse) else base
            pygame.draw.rect(screen, col, rect, border_radius=14)
            pygame.draw.rect(screen, (255, 255, 255), rect, 3, border_radius=14)
            lab = font_btn.render(str(label), True, (255, 255, 255))
            screen.blit(lab, lab.get_rect(center=rect.center))
            k = font_key.render(f"[{key}]", True, (235, 235, 235))
            screen.blit(k, (rect.left + 14, rect.top + 10))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                ch = pygame.key.name(event.key).upper()
                for (key, label) in options:
                    if ch == str(key).upper():
                        return label
            if event.type == pygame.MOUSEBUTTONDOWN:
                for (key, label), rect in zip(options, rects):
                    if rect.collidepoint(event.pos):
                        return label
        clock.tick(60)


def number_entry_pygame(screen, clock, prompt, allow_blank=True):
    """Fullscreen numeric keypad. Returns a string (possibly ""), or None on ESC."""
    font_title = pygame.font.Font(None, 56)
    font_val   = pygame.font.Font(None, 88)
    font_btn   = pygame.font.Font(None, 54)

    W, H = screen.get_width(), screen.get_height()
    buf = ""
    keys = ["7", "8", "9", "4", "5", "6", "1", "2", "3", ".", "0", "<"]
    kw, kh, gap = 128, 86, 14
    gw = kw * 3 + gap * 2
    gx, gy = (W - gw) // 2, 290
    rects = {}
    for i, k in enumerate(keys):
        r, c = divmod(i, 3)
        rects[k] = pygame.Rect(gx + c * (kw + gap), gy + r * (kh + gap), kw, kh)
    ok_rect   = pygame.Rect(W // 2 - 210, gy + 4 * (kh + gap) + 8, 200, 76)
    skip_rect = pygame.Rect(W // 2 + 10,  gy + 4 * (kh + gap) + 8, 200, 76)

    while True:
        mouse = pygame.mouse.get_pos()
        screen.fill((25, 25, 30))
        t = font_title.render(prompt, True, (255, 255, 255))
        screen.blit(t, t.get_rect(center=(W // 2, 85)))
        v = font_val.render(buf if buf else "—", True, (120, 220, 255))
        screen.blit(v, v.get_rect(center=(W // 2, 195)))
        for k, r in rects.items():
            col = (100, 115, 135) if r.collidepoint(mouse) else (70, 80, 95)
            pygame.draw.rect(screen, col, r, border_radius=10)
            pygame.draw.rect(screen, (200, 200, 210), r, 2, border_radius=10)
            lab = font_btn.render(k, True, (255, 255, 255))
            screen.blit(lab, lab.get_rect(center=r.center))
        for rect, label, col in ((ok_rect, "OK", (46, 204, 113)),
                                 (skip_rect, "SKIP", (149, 165, 166))):
            if label == "SKIP" and not allow_blank:
                continue
            c = tuple(max(0, x - 40) for x in col) if rect.collidepoint(mouse) else col
            pygame.draw.rect(screen, c, rect, border_radius=12)
            pygame.draw.rect(screen, (255, 255, 255), rect, 3, border_radius=12)
            lab = font_btn.render(label, True, (255, 255, 255))
            screen.blit(lab, lab.get_rect(center=rect.center))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and (buf or allow_blank):
                    return buf
                if event.key == pygame.K_BACKSPACE:
                    buf = buf[:-1]
                ch = event.unicode
                if ch and (ch.isdigit() or (ch == "." and "." not in buf)):
                    buf += ch
            if event.type == pygame.MOUSEBUTTONDOWN:
                for k, r in rects.items():
                    if r.collidepoint(event.pos):
                        if k == "<":
                            buf = buf[:-1]
                        elif not (k == "." and "." in buf):
                            buf += k
                if ok_rect.collidepoint(event.pos) and (buf or allow_blank):
                    return buf
                if allow_blank and skip_rect.collidepoint(event.pos):
                    return ""
        clock.tick(60)


def info_screen_pygame(screen, clock, lines, title=""):
    """Show a block of text; wait for any tap/key."""
    font_title = pygame.font.Font(None, 62)
    font_line  = pygame.font.Font(None, 38)
    W, H = screen.get_width(), screen.get_height()
    while True:
        screen.fill((25, 25, 30))
        if title:
            t = font_title.render(title, True, (255, 255, 255))
            screen.blit(t, t.get_rect(center=(W // 2, 65)))
        y = 150
        for ln in lines:
            s = font_line.render(str(ln), True, (220, 220, 230))
            screen.blit(s, (70, y)); y += 46
        hint = font_line.render("tap / press any key to continue", True, (130, 130, 140))
        screen.blit(hint, hint.get_rect(center=(W // 2, H - 55)))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return
        clock.tick(60)


# ============================================================
# MODE SELECTION
# ============================================================
MODE_TRAINING   = "Training"
MODE_COLLECTION = "Data Collection"
MODE_PROBE      = "Breaking-Point Probe"


def mode_select_menu(screen, clock, monkey_name, suggested=None):
    """
    Mode chooser. `suggested` is shown in the subtitle so the operator can see
    at a glance where the animal currently is (from its training log), but the
    choice is always explicit — the code never silently switches modes.
    """
    sub = f"Subject: {monkey_name}"
    if suggested:
        sub += f"   |   {suggested}"
    choice = choice_menu_pygame(
        screen, clock, "Select mode", subtitle=sub,
        options=[("T", MODE_TRAINING), ("D", MODE_COLLECTION), ("P", MODE_PROBE)])
    if choice is None:
        pygame.quit(); raise SystemExit
    return choice


def describe_training_state(training_log, monkey_name):
    """One-line summary of where this animal is, for the mode menu subtitle."""
    try:
        df = pd.read_csv(training_log)
        df = df[~df["Trial Type"].astype(str).str.upper().str.contains("MARKER", na=False)]
        if df.empty:
            return "no training data yet"
        last = df.iloc[-1]
        ph = int(last["Phase #"])
        if ph >= 8:
            return "Phase 8 (fully trained) — ready for data collection"
        return f"currently Phase {ph}, substage {int(last['Substage #'])}"
    except Exception:
        return "no training log found"


# ============================================================
# DATA COLLECTION
# ============================================================
CONDITIONS = ["Healthy", "MPTP", "PD", "PD+KET", "Washout"]

CUE_COST_REWARD = {
    "HCHR":  ("High", "High"),
    "HCLR":  ("High", "Low"),
    "LCLR":  ("Low",  "Low"),
    "LCHR":  ("Low",  "High"),
    "PLAIN": ("None", "Low"),
}

# Balanced 2x2 for the factorial design + PLAIN engagement probe.
COLLECTION_TRIAL_PROBS = {
    "HCHR": 0.225, "HCLR": 0.225, "LCLR": 0.225, "LCHR": 0.225, "PLAIN": 0.10,
}
COLLECTION_BLOCK_SIZE   = 200      # pre-generated queue; refilled as needed
COLLECTION_TIMEOUT_MS   = 20_000   # 20 s per touch (matches Phase 8)
COLLECTION_HARD_CAP_MIN = 180      # safety cap only; operator ends with ESC

# ---- Log schemas and their history ----
# RULE FOR FUTURE CHANGES: never insert a column mid-list. Append new columns
# at the END, and record the previous list as a new version in the matching
# *_HISTORY. _ensure_csv uses the history to migrate files written by any
# earlier version: each existing row is matched to the version with the same
# number of fields and re-keyed BY COLUMN NAME, so no value can land under the
# wrong heading — even in a file that mixes rows from several versions.
_COLLECTION_TRIAL_V1 = [
    "Subject", "SessionUID", "Date", "Condition", "InfusionNumber",
    "DaysSinceInfusion", "DaysSinceMPTP", "ParkinsonianScore", "Rig", "HighCostN",
    "Trial #", "Trial Type", "Cost", "Reward",
    "Time Stim On", "Time Circle Touched", "Circle Touch Latency (ms)",
    "Cue Touches Required", "Cue Touches Completed",
    "Cue Interaction Duration (ms)", "First Cue Touch RT (ms)",
    "Correct Trial?", "Current % Correct",
    "Pellets Dispensed", "Pellet Delivered?", "Error Type",
    "CumulativeTrials", "CumulativePellets", "MinutesIntoSession",
    "PrevTrialType", "PrevTrialOutcome",
    "Mono_TrialStart", "Mono_CueOnset", "Mono_Outcome", "Mono_PelletCommand",
]
# v2: inter-touch lockout. InterTouchLockout records the setting ("100-300" or
# "off") so data from before and after the change stay separable.
COLLECTION_TRIAL_HEADERS = _COLLECTION_TRIAL_V1 + ["InterTouchLockout", "PrematureTouches"]
COLLECTION_TRIAL_HISTORY = [_COLLECTION_TRIAL_V1, COLLECTION_TRIAL_HEADERS]

_COLLECTION_TOUCH_V1 = [
    "SessionUID", "Trial #", "Trial Type", "Touch Index",
    "Mono_Touch", "Wall_Touch", "X", "Y",
]
# v2: per-touch lockout detail. Raw inter-touch intervals now include the
# enforced lockout, so RT_From_Available_ms is the clean motor measure.
COLLECTION_TOUCH_HEADERS = _COLLECTION_TOUCH_V1 + [
    "Lockout_Before_ms", "Premature_Before", "RT_From_Available_ms"]
COLLECTION_TOUCH_HISTORY = [_COLLECTION_TOUCH_V1, COLLECTION_TOUCH_HEADERS]

_COLLECTION_SESSION_V2 = [
    "SessionUID", "Subject", "Date", "Condition", "InfusionNumber",
    "DaysSinceInfusion", "DaysSinceMPTP", "ParkinsonianScore", "Rig", "HighCostN",
    "Start Wallclock", "End Wallclock", "Duration (min)",
    "Mono_SessionStart", "CameraEnabled", "SyncFlash_Mono", "VideoFilename",
    "PrimingPellet_Mono", "PrimingPellet_Delivered",
    "Trials", "Correct", "% Correct",
    "HCHR n", "HCHR %", "HCLR n", "HCLR %",
    "LCLR n", "LCLR %", "LCHR n", "LCHR %", "PLAIN n",
    "Pellets Dispensed", "Pellets Confirmed", "Exit Reason",
]
# v1: before the priming pellet. The PrimingPellet columns were inserted
# mid-list (positions 17-18), which is what made the earlier drift unreadable;
# name-based re-keying handles it.
_COLLECTION_SESSION_V1 = [c for c in _COLLECTION_SESSION_V2
                          if c not in ("PrimingPellet_Mono", "PrimingPellet_Delivered")]
# v3: inter-touch lockout setting.
COLLECTION_SESSION_HEADERS = _COLLECTION_SESSION_V2 + ["InterTouchLockout"]
COLLECTION_SESSION_HISTORY = [_COLLECTION_SESSION_V1, _COLLECTION_SESSION_V2,
                              COLLECTION_SESSION_HEADERS]


def get_collection_paths(monkey_name):
    """Collection files live in the SAME per-monkey folder as the training log,
    but are entirely separate files — training data is never mixed with
    experimental data."""
    folder = get_monkey_folder(monkey_name)
    return (folder / f"{monkey_name}_collection_trials.csv",
            folder / f"{monkey_name}_collection_touches.csv",
            folder / f"{monkey_name}_collection_sessions.csv")


def _ensure_csv(path, headers, history=None):
    """
    Create `path` with `headers` if absent. If it exists with an OLDER header,
    migrate it in place, after writing a timestamped backup.

    Why: this used to write the header only when the file did not exist. When
    columns were later added to the code, existing files kept their old header
    while csv.DictWriter wrote the new, wider rows — the file became
    unparseable ("Expected 32 fields in line 4, saw 34").

    How: `history` lists every header version the file may contain (oldest
    first, current last). Each existing row is matched to the version with the
    same number of fields and re-keyed BY COLUMN NAME into the current header;
    columns a row never had are left blank. Matching by name makes this correct
    even when columns were inserted mid-list, and even when a single file mixes
    rows written by several versions.
    """
    headers = list(headers)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)
        return

    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except Exception as e:
        print(f"Could not read {path.name} to check its header: {e}")
        return
    if not rows:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)
        return

    on_disk = rows[0]
    if on_disk == headers and all(len(r) == len(headers) for r in rows[1:]):
        return                                          # already current

    # SAFETY: only migrate a file that is recognisably an older version of THIS
    # schema. Without this check the function rewrites whatever it is given to
    # whatever header it is given — handed the wrong headers for a file, it
    # blanks almost every column, because no name matches. (Found the hard way:
    # a test passed the session-log headers for a trial log and the trial log
    # was rewritten to 51 mostly-empty columns. The backup saved it, but a
    # migration should not need saving from.) A genuine older version's columns
    # are all still present in the current header, or it matches a recorded
    # history version exactly.
    known = [list(v) for v in (history or [])]
    recognised = (on_disk in known) or set(on_disk).issubset(set(headers))
    if not recognised:
        shared = len(set(on_disk) & set(headers))
        print(f"REFUSING to migrate {path.name}: its header is not a known version "
              f"of this schema ({len(on_disk)} columns, only {shared} of them "
              f"shared with the expected {len(headers)}). File left untouched.")
        return

    by_width = {}
    for v in [list(v) for v in (history or [])] + [on_disk, headers]:
        by_width[len(v)] = v                            # later versions win ties

    fixed, unknown = [], 0
    for r in rows[1:]:
        src_hdr = by_width.get(len(r))
        if src_hdr is None:
            unknown += 1
            src_hdr = on_disk
        rec = dict(zip(src_hdr, r))
        fixed.append([rec.get(c, "") for c in headers])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}_BACKUP_{ts}{path.suffix}")
    try:
        shutil.copy2(path, backup)
    except Exception as e:
        print(f"Could not back up {path.name} ({e}) — migration skipped, file untouched.")
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(fixed)
    print(f"Migrated {path.name} to the current {len(headers)}-column header "
          f"({len(fixed)} rows). Backup: {backup.name}")
    if unknown:
        print(f"  !! {unknown} row(s) had an unrecognised width and were aligned to "
              f"the old header as a best effort — inspect them.")


def collect_experiment_metadata(screen, clock, monkey_name):
    """Nested prompts — only asks what is relevant to the chosen condition."""
    meta = {k: "" for k in ("Condition", "InfusionNumber", "DaysSinceInfusion",
                            "DaysSinceMPTP", "ParkinsonianScore", "Rig",
                            "VideoFilename")}
    meta["Subject"] = monkey_name

    cond = choice_menu_pygame(screen, clock, "Condition?",
                              [(str(i + 1), c) for i, c in enumerate(CONDITIONS)],
                              subtitle=f"Subject: {monkey_name}")
    if cond is None:
        return None
    meta["Condition"] = cond

    if cond == "MPTP":
        meta["DaysSinceMPTP"] = number_entry_pygame(
            screen, clock, "Days since most recent MPTP dose?") or ""
    elif cond == "PD+KET":
        n = choice_menu_pygame(screen, clock, "Which ketamine infusion?",
                               [("1", "1"), ("2", "2"), ("3", "3")])
        meta["InfusionNumber"] = n or ""
        meta["DaysSinceInfusion"] = number_entry_pygame(
            screen, clock, f"Days since infusion #{n}?") or ""
    elif cond == "Washout":
        meta["DaysSinceInfusion"] = number_entry_pygame(
            screen, clock, "Days since FINAL infusion?") or ""

    # Parkinsonian score is only meaningful once the animal has received MPTP.
    # Healthy sessions are pre-lesion, so the prompt is skipped entirely and
    # the column is left blank rather than being filled with a placeholder.
    if cond in ("MPTP", "PD", "PD+KET", "Washout"):
        meta["ParkinsonianScore"] = number_entry_pygame(
            screen, clock, "Parkinsonian score?") or ""

    meta["Rig"] = choice_menu_pygame(screen, clock, "Rig?",
                                     [("1", "Rig1"), ("2", "Rig2"), ("3", "Rig3")]) or ""

    shown = [f"{k:<18}: {v}" for k, v in meta.items() if v]
    shown.append("")
    shown.append(f"{'Camera':<18}: {'ENABLED' if CAMERA_RECORDING_ENABLED else 'disabled'}")
    info_screen_pygame(screen, clock, shown, title="Confirm session metadata")
    return meta


def run_collection_session(clock, screen, screen_info, monkey_name, meta, high_cost_n):
    """
    One data-collection session. No phase state, no advancement.
    Runs until the operator presses ESC (or the hard safety cap).
    Writes trial rows, per-touch rows, and one session summary row.
    """
    trial_path, touch_path, sess_path = get_collection_paths(monkey_name)
    for p, h, hist in ((trial_path, COLLECTION_TRIAL_HEADERS,   COLLECTION_TRIAL_HISTORY),
                       (touch_path, COLLECTION_TOUCH_HEADERS,   COLLECTION_TOUCH_HISTORY),
                       (sess_path,  COLLECTION_SESSION_HEADERS, COLLECTION_SESSION_HISTORY)):
        _ensure_csv(p, h, hist)

    start_dt    = datetime.now()
    session_uid = f"{monkey_name}_{start_dt:%Y%m%d_%H%M%S}"
    mono_start  = time.monotonic()

    if CAMERA_RECORDING_ENABLED and CAMERA_FILENAME_MODE == "auto":
        meta["VideoFilename"] = f"{session_uid}.mp4"

    intro = ["Press ESC at any time to end the session.", ""]
    if CAMERA_RECORDING_ENABLED:
        intro = ["Camera recording ENABLED.",
                 "Make sure the camera is running before continuing.",
                 f"Video file: {meta.get('VideoFilename','(unset)')}",
                 "A sync flash will play now.", ""] + intro
    info_screen_pygame(screen, clock, intro, title="Ready")
    flash_stamps = sync_flash(screen, clock)

    # Free pellet before the first trial — clean reach-and-retrieve for the
    # hand tracker at a known timestamp, and it absorbs any dropped first
    # HID report so trial 1 is not silently unrewarded.
    prime_mono, prime_delivered = deliver_priming_pellet(screen, clock, mono_start)

    shortest = min(screen_info.current_w, screen_info.current_h)
    center   = (screen_info.current_w // 2, screen_info.current_h // 2)
    radius   = int(shortest * 0.45 / 2)
    prebuild_shape_cache(radius)

    stats = {c: {"n": 0, "ok": 0} for c in CUE_COST_REWARD}
    cum_correct = cum_pellets = cum_confirmed = 0
    trial_idx = 0
    prev_type = prev_outcome = ""
    exit_reason = "completed"
    running = True
    queue = build_trial_list(COLLECTION_TRIAL_PROBS, COLLECTION_BLOCK_SIZE)

    while running:
        if not queue:
            queue = build_trial_list(COLLECTION_TRIAL_PROBS, COLLECTION_BLOCK_SIZE)
        if (time.monotonic() - mono_start) / 60.0 >= COLLECTION_HARD_CAP_MIN:
            exit_reason = "hard_cap"; break

        trial_type = queue.pop(0)
        trial_idx += 1
        mono_trial = time.monotonic()
        t_stim = datetime.now()

        draw_circle_only(screen, center, radius)
        pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
        c0 = pygame.time.get_ticks()
        touched = False
        while not touched:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False; exit_reason = "escaped"; touched = True; break
                if event.type == pygame.MOUSEBUTTONDOWN and hit_circle(event.pos, center, radius):
                    touched = True; break
            clock.tick(60)
        if not running:
            trial_idx -= 1
            break

        circle_latency = pygame.time.get_ticks() - c0
        t_touch = datetime.now()

        screen.fill((0, 0, 0)); pygame.display.flip()
        sleep_ms(CUE_APPEAR_DELAY_MS)
        pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
        mono_cue = time.monotonic()
        touch_events = []

        premature = ""              # only multi-touch cues have an inter-touch lockout
        if trial_type == "PLAIN":
            ok, req, done, dur, ftr, err, pellets = True, 0, 0, "N/A", "N/A", "None", 1
        elif trial_type in ("HCHR", "HCLR"):
            fn = run_hchr_cue if trial_type == "HCHR" else run_hclr_cue
            cue_stats = {}
            ok, done, dur, ftr, err = fn(screen, clock, center, radius,
                                         high_cost_n, COLLECTION_TIMEOUT_MS,
                                         touch_log=touch_events, cue_stats=cue_stats)
            premature = cue_stats.get("premature_touches", 0)
            req = high_cost_n
            pellets = (3 if trial_type == "HCHR" else 1) if ok else 0
            err = err or "None"
        else:
            fn = run_lclr_cue if trial_type == "LCLR" else run_lchr_cue
            ok, dur, ftr, err = fn(screen, clock, center, radius,
                                   COLLECTION_TIMEOUT_MS, touch_log=touch_events)
            req, done = 1, (1 if ok else 0)
            pellets = (1 if trial_type == "LCLR" else 3) if ok else 0
            err = err or "None"

        if err == "escaped":
            running = False; exit_reason = "escaped"; trial_idx -= 1; break

        mono_outcome = time.monotonic()
        mono_pellet = ""
        if ok:
            screen.fill('#008000'); pygame.display.flip()
            mono_pellet = round(time.monotonic() - mono_start, 4)
            delivered = dispense_pellets(device, pellets)
            cum_pellets += pellets
            if delivered == "Yes":
                cum_confirmed += pellets
            sleep_ms(300)
            screen.fill('#898989'); pygame.display.flip()
            sleep_ms(NORMAL_ITI_MS)
            write_to_adu(device, 'RK0')
        else:
            delivered = "N/A"
            screen.fill('#321010'); pygame.display.flip()
            sleep_ms(300)
            screen.fill('#898989'); pygame.display.flip()
            sleep_ms(PENALTY_ITI_MS)
        pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION])

        cum_correct += int(ok)
        stats[trial_type]["n"]  += 1
        stats[trial_type]["ok"] += int(ok)
        cost, reward = CUE_COST_REWARD[trial_type]

        with open(trial_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLLECTION_TRIAL_HEADERS).writerow({
                "Subject": monkey_name, "SessionUID": session_uid,
                "Date": t_stim.strftime("%m-%d-%Y"),
                "Condition": meta.get("Condition", ""),
                "InfusionNumber": meta.get("InfusionNumber", ""),
                "DaysSinceInfusion": meta.get("DaysSinceInfusion", ""),
                "DaysSinceMPTP": meta.get("DaysSinceMPTP", ""),
                "ParkinsonianScore": meta.get("ParkinsonianScore", ""),
                "Rig": meta.get("Rig", ""), "HighCostN": high_cost_n,
                "Trial #": trial_idx, "Trial Type": trial_type,
                "Cost": cost, "Reward": reward,
                "Time Stim On": t_stim.strftime("%H:%M:%S"),
                "Time Circle Touched": t_touch.strftime("%H:%M:%S"),
                "Circle Touch Latency (ms)": round(circle_latency, 1),
                "Cue Touches Required": req, "Cue Touches Completed": done,
                "Cue Interaction Duration (ms)": dur,
                "First Cue Touch RT (ms)": ftr,
                "Correct Trial?": int(ok),
                "Current % Correct": f"{100*cum_correct/trial_idx:.2f}",
                "Pellets Dispensed": pellets, "Pellet Delivered?": delivered,
                "Error Type": err,
                "CumulativeTrials": trial_idx, "CumulativePellets": cum_pellets,
                "MinutesIntoSession": round((time.monotonic() - mono_start) / 60.0, 2),
                "PrevTrialType": prev_type, "PrevTrialOutcome": prev_outcome,
                "Mono_TrialStart": round(mono_trial - mono_start, 4),
                "Mono_CueOnset":   round(mono_cue - mono_start, 4),
                "Mono_Outcome":    round(mono_outcome - mono_start, 4),
                "Mono_PelletCommand": mono_pellet,
                "InterTouchLockout": lockout_label(),
                "PrematureTouches": premature,
            })

        if touch_events:
            with open(touch_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLLECTION_TOUCH_HEADERS)
                for te in touch_events:
                    w.writerow({
                        "SessionUID": session_uid, "Trial #": trial_idx,
                        "Trial Type": trial_type, "Touch Index": te["index"],
                        "Mono_Touch": round(te["mono"] - mono_start, 4),
                        "Wall_Touch": te["wall"], "X": te["x"], "Y": te["y"],
                        "Lockout_Before_ms":    te.get("lockout_ms", ""),
                        "Premature_Before":     te.get("premature_before", ""),
                        "RT_From_Available_ms": te.get("rt_from_available", ""),
                    })

        prev_type, prev_outcome = trial_type, ("correct" if ok else "incorrect")

    end_dt  = datetime.now()
    dur_min = round((end_dt - start_dt).total_seconds() / 60.0, 2)
    srow = {
        "SessionUID": session_uid, "Subject": monkey_name,
        "Date": start_dt.strftime("%m-%d-%Y"),
        "Condition": meta.get("Condition", ""),
        "InfusionNumber": meta.get("InfusionNumber", ""),
        "DaysSinceInfusion": meta.get("DaysSinceInfusion", ""),
        "DaysSinceMPTP": meta.get("DaysSinceMPTP", ""),
        "ParkinsonianScore": meta.get("ParkinsonianScore", ""),
        "Rig": meta.get("Rig", ""), "HighCostN": high_cost_n,
        "Start Wallclock": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "End Wallclock": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Duration (min)": dur_min,
        "Mono_SessionStart": round(mono_start, 4),
        "CameraEnabled": "Yes" if CAMERA_RECORDING_ENABLED else "No",
        "SyncFlash_Mono": ";".join(f"{s - mono_start:.4f}" for s in flash_stamps),
        "VideoFilename": meta.get("VideoFilename", ""),
        "PrimingPellet_Mono": prime_mono,
        "PrimingPellet_Delivered": prime_delivered,
        "Trials": trial_idx, "Correct": cum_correct,
        "% Correct": f"{100*cum_correct/trial_idx:.2f}" if trial_idx else "0",
        "Pellets Dispensed": cum_pellets, "Pellets Confirmed": cum_confirmed,
        "Exit Reason": exit_reason,
        "InterTouchLockout": lockout_label(),
    }
    for c in ["HCHR", "HCLR", "LCLR", "LCHR"]:
        srow[f"{c} n"] = stats[c]["n"]
        srow[f"{c} %"] = f"{100*stats[c]['ok']/stats[c]['n']:.2f}" if stats[c]["n"] else ""
    srow["PLAIN n"] = stats["PLAIN"]["n"]
    with open(sess_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLLECTION_SESSION_HEADERS).writerow(srow)

    lines = [f"Condition   : {meta.get('Condition','')}",
             f"Trials      : {trial_idx}",
             f"% correct   : {srow['% Correct']}",
             f"Duration    : {dur_min} min",
             f"High-cost N : {high_cost_n}", ""]
    for c in ["HCHR", "HCLR", "LCLR", "LCHR"]:
        lines.append(f"{c:<6}: n={stats[c]['n']:<4} {srow[f'{c} %'] or '—'}%")
    info_screen_pygame(screen, clock, lines, title="Session complete")
    return srow


# ============================================================
# BREAKING-POINT PROBE MODE
# ============================================================
PROBE_N_LEVELS       = [20, 22, 24, 26, 30]
PROBE_TRIALS_PER_N   = 30
PROBE_PLAIN_FRACTION = 0.20
PROBE_TIMEOUT_MS     = 60_000

_PROBE_V1 = [
    "Subject", "SessionUID", "Date", "Time Stim On", "N Level", "Trial #",
    "Trial Type", "Cue Touches Required", "Cue Touches Completed",
    "Cue Interaction Duration (ms)", "First Cue Touch RT (ms)",
    "Circle Touch Latency (ms)", "Correct Trial?", "Pellets Dispensed",
    "Pellet Delivered?", "Error Type",
]
# v2: lockout setting + premature taps. Earlier probe sessions ran without a
# lockout, so this column is what keeps them separable from later ones.
PROBE_HEADERS = _PROBE_V1 + ["InterTouchLockout", "PrematureTouches"]
PROBE_HISTORY = [_PROBE_V1, PROBE_HEADERS]


def get_probe_path(monkey_name):
    return get_monkey_folder(monkey_name) / f"{monkey_name}_probe_log.csv"


def run_probe_session(clock, screen, screen_info, monkey_name):
    """
    Standalone breaking-point probe: HCHR only at escalating N.
    Writes ONLY to {monkey}_probe_log.csv — never touches the training log or
    phase state, so an already-trained animal can be probed safely.
    Stops early and reports as soon as abandonment exceeds the threshold.
    """
    path = get_probe_path(monkey_name)
    _ensure_csv(path, PROBE_HEADERS, PROBE_HISTORY)

    start_dt = datetime.now()
    uid = f"{monkey_name}_probe_{start_dt:%Y%m%d_%H%M%S}"

    info_screen_pygame(screen, clock,
                       [f"N levels        : {PROBE_N_LEVELS}",
                        f"Trials per level: {PROBE_TRIALS_PER_N}",
                        f"Abandon bar     : {ABANDON_RATE_THRESHOLD:.0%}",
                        f"Touch lockout   : {lockout_label()} ms",
                        "",
                        "HCHR only (3 pellets), PLAIN interleaved.",
                        "Writes to the probe log only — phase state untouched.",
                        "Press ESC at any time to stop."],
                       title=f"Breaking-point probe — {monkey_name}")

    deliver_priming_pellet(screen, clock)

    shortest = min(screen_info.current_w, screen_info.current_h)
    center   = (screen_info.current_w // 2, screen_info.current_h // 2)
    radius   = int(shortest * 0.45 / 2)
    prebuild_shape_cache(radius)

    results, running, found = {}, True, None

    for n in PROBE_N_LEVELS:
        if not running:
            break
        seq = ["HCHR"] * PROBE_TRIALS_PER_N + \
              ["PLAIN"] * int(round(PROBE_TRIALS_PER_N * PROBE_PLAIN_FRACTION))
        random.shuffle(seq)
        attempted = completed = 0

        for i, tt in enumerate(seq, start=1):
            if not running:
                break
            t_stim = datetime.now()
            draw_circle_only(screen, center, radius)
            pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
            c0 = pygame.time.get_ticks()
            touched = False
            while not touched:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False; touched = True; break
                    if event.type == pygame.MOUSEBUTTONDOWN and hit_circle(event.pos, center, radius):
                        touched = True; break
                clock.tick(60)
            if not running:
                break
            lat = pygame.time.get_ticks() - c0

            screen.fill((0, 0, 0)); pygame.display.flip()
            sleep_ms(CUE_APPEAR_DELAY_MS)
            pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

            premature = ""
            if tt == "PLAIN":
                ok, req, done, dur, ftr, err, pel = True, 0, 0, "N/A", "N/A", "None", 1
            else:
                cue_stats = {}
                ok, done, dur, ftr, err = run_hchr_cue(screen, clock, center, radius,
                                                       n, PROBE_TIMEOUT_MS,
                                                       cue_stats=cue_stats)
                premature = cue_stats.get("premature_touches", 0)
                if err == "escaped":
                    running = False; break
                req, pel, err = n, (3 if ok else 0), (err or "None")
                attempted += 1; completed += int(ok)

            if ok:
                screen.fill('#008000'); pygame.display.flip()
                delivered = dispense_pellets(device, pel)
                sleep_ms(300)
                screen.fill('#898989'); pygame.display.flip()
                sleep_ms(NORMAL_ITI_MS)
                write_to_adu(device, 'RK0')
            else:
                delivered = "N/A"
                screen.fill('#321010'); pygame.display.flip()
                sleep_ms(300)
                screen.fill('#898989'); pygame.display.flip()
                sleep_ms(PENALTY_ITI_MS)
            pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION])

            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=PROBE_HEADERS).writerow({
                    "Subject": monkey_name, "SessionUID": uid,
                    "Date": t_stim.strftime("%m-%d-%Y"),
                    "Time Stim On": t_stim.strftime("%H:%M:%S"),
                    "N Level": n, "Trial #": i, "Trial Type": tt,
                    "Cue Touches Required": req, "Cue Touches Completed": done,
                    "Cue Interaction Duration (ms)": dur,
                    "First Cue Touch RT (ms)": ftr,
                    "Circle Touch Latency (ms)": round(lat, 1),
                    "Correct Trial?": int(ok), "Pellets Dispensed": pel,
                    "Pellet Delivered?": delivered, "Error Type": err,
                    "InterTouchLockout": lockout_label(),
                    "PrematureTouches": premature,
                })

        if attempted:
            ab = 1 - completed / attempted
            results[n] = {"attempted": attempted, "completed": completed, "abandon": ab}
            print(f"Probe N={n}: {completed}/{attempted} completed, abandon {100*ab:.1f}%")
            if ab > ABANDON_RATE_THRESHOLD:
                found = n
                info_screen_pygame(screen, clock,
                                   [f"Abandonment at N={n}: {100*ab:.0f}%",
                                    f"(bar is {ABANDON_RATE_THRESHOLD:.0%})", "",
                                    f"BREAKING POINT = {n}",
                                    f"High-cost N     = {n}", "",
                                    "Add to HIGH_COST_N_OVERRIDES:",
                                    f'    "{monkey_name}": {n},'],
                                   title="Breaking point found")
                running = False

    if results and found is None:
        lines = [f"N={n}: {r['completed']}/{r['attempted']} "
                 f"(abandon {100*r['abandon']:.0f}%)" for n, r in sorted(results.items())]
        lines += ["", "No criterion fired at any level tested.",
                  "Consider extending PROBE_N_LEVELS."]
        info_screen_pygame(screen, clock, lines, title="Probe complete")
    return results


if __name__ == "__main__":

    HEADERS = [
        "Date",
        "Phase #",
        "Substage #",
        "Session #",
        "Trial #",
        "Time Stim On",
        "Time Circle Touched",
        "Circle Touch Latency (ms)",
        "Trial Type",
        "Cue Touches Required",
        "Cue Touches Completed",
        "Cue Interaction Duration (ms)",
        "First Cue Touch RT (ms)",
        "Correct Trial?",
        "Current % Correct",
        "Pellets Dispensed",
        "Pellet Delivered?",
        "Error Type",
    ]

    write_to_adu(device, 'RK0')  # ensure relay starts open

    # Init pygame before anything else so the monkey menu can use the screen
    pygame.init()
    screen_info = pygame.display.Info()
    screen = pygame.display.set_mode(
        (0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.mouse.set_visible(True)
    clock = pygame.time.Clock()

    # Touchscreen monkey selection — no terminal needed
    monkey_choice = monkey_user_menu_pygame(screen, clock)
    print(f"Monkey: {monkey_choice}")

    # ------------------------------------------------------------------
    # MODE SELECTION — Training / Data Collection / Breaking-Point Probe
    # All three share the monkey menu, cue rendering, trial mechanics and
    # pellet delivery. They differ in session structure and output files.
    # The menu shows where the animal currently is, but the choice is always
    # explicit — the code never silently switches modes on you.
    # ------------------------------------------------------------------
    _training_log, _ = get_log_paths(monkey_choice)
    mode = mode_select_menu(screen, clock, monkey_choice,
                            suggested=describe_training_state(_training_log, monkey_choice))
    print(f"Mode: {mode}")
    print(f"Inter-touch lockout (HCHR/HCLR): {lockout_label()}"
          f"{' ms' if INTER_TOUCH_LOCKOUT_ENABLED else ''}")

    # ---------- BREAKING-POINT PROBE ----------
    if mode == MODE_PROBE:
        if not wait_for_start_command(screen, clock):
            pygame.quit(); raise SystemExit
        run_probe_session(clock, screen, screen_info, monkey_choice)
        wait_for_exit_command(screen, clock)
        pygame.quit(); raise SystemExit

    # ---------- DATA COLLECTION ----------
    if mode == MODE_COLLECTION:
        # Resolve this subject's high-cost N once, up front, and show it
        # (including any ceiling warning) before any data is collected.
        bp = analyze_breaking_point(_training_log, monkey_name=monkey_choice)
        print_breaking_point_report(bp, monkey_choice)
        warn = ([] if bp["status"] != "CEILING_NOT_MEASURED" else
                ["", "WARNING: breaking point was never measured for this",
                 "animal — the N below is only a lower bound.",
                 "Consider running PROBE mode first."])
        info_screen_pygame(screen, clock,
                           [f"High-cost N : {bp['high_cost_n']}",
                            f"Source      : {bp['status']}"] + warn,
                           title="Effort calibration")

        meta = collect_experiment_metadata(screen, clock, monkey_choice)
        if meta is None:
            pygame.quit(); raise SystemExit
        if not wait_for_start_command(screen, clock):
            pygame.quit(); raise SystemExit

        run_collection_session(clock, screen, screen_info, monkey_choice,
                               meta, bp["high_cost_n"])
        wait_for_exit_command(screen, clock)
        pygame.quit(); raise SystemExit

    # ---------- TRAINING (falls through to the existing protocol loop) ----------
    # ------------------------------------------------------------------
    # Folder + file paths  (new folder-based structure)
    # ------------------------------------------------------------------
    file_path, session_log_path = get_log_paths(monkey_choice)
    _ensure_session_log(session_log_path)

    current_phase, substage_idx, current_session, last_trial_num, starting_correct = \
        get_resume_state(file_path, HEADERS, monkey_choice)

    # ------------------------------------------------------------------
    # OVERRIDE SHORT-CIRCUITS THE PHASE 4 RAMP  (checked before the session
    # starts, so no further ramp session is run)
    #
    # The in-session check below catches this too, but only after a whole
    # extra ramp session has been collected at the wrong N. An animal with a
    # hand-set high-cost N should not run the ramp at all: the override exists
    # precisely because the ramp is not going to produce a usable number for
    # them.
    # ------------------------------------------------------------------
    if current_phase == 4 and monkey_choice in HIGH_COST_N_OVERRIDES:
        _ov_n = int(HIGH_COST_N_OVERRIDES[monkey_choice])
        print(f"{monkey_choice} is in Phase 4 but has HIGH_COST_N_OVERRIDES = {_ov_n}. "
              f"Skipping the rest of the ramp and starting Phase 5.")
        info_screen_pygame(
            screen, clock,
            [f"Was        : Phase 4 sub{substage_idx} "
             f"(N={PHASE4_N_VALUES[min(substage_idx, len(PHASE4_N_VALUES) - 1)]})",
             f"Override   : high-cost N = {_ov_n}",
             f"Starting   : Phase 5, session 1",
             "",
             "This animal's high-cost touch requirement is set by hand",
             "in HIGH_COST_N_OVERRIDES, so the Phase 4 ramp is skipped.",
             "Both HCHR and HCLR will use N = " + str(_ov_n) + ".",
             "",
             "Remove the entry to put them back on the ramp."],
            title="Phase 4 ramp skipped (override)")
        current_phase, substage_idx = 5, 0
        current_session, last_trial_num, starting_correct = 1, 0, 0

    if not wait_for_start_command(screen, clock):
        pygame.quit()
        raise SystemExit

    deliver_priming_pellet(screen, clock)

    # Record the moment Shift+S was pressed — this is the session start time
    program_start_dt = datetime.now()

    FINAL_PHASE = 8
    # BUGFIX: this used to be a heuristic —
    #     current_phase == 8 and last_trial_num == 0 and current_session > 1
    # which is true after ANY single completed Phase 8 session, regardless of
    # performance. An animal that ran one Phase 8 session at 10% correct and
    # then relaunched was told "Training protocol complete!" and locked out.
    # (Sessions 2+ only ever ran because the loop increments current_session
    # internally without re-checking, so the lockout only bit on relaunch.)
    # Now the real per-cue rolling-window check decides.
    protocol_complete = False
    if current_phase == FINAL_PHASE:
        _final_cfg = get_stage_config(FINAL_PHASE, 0, monkey_choice)
        protocol_complete, _final_details = pass_phase8_check(
            log_path=file_path,
            required_trials=_final_cfg["required_trials"],
        )
        if protocol_complete:
            print("Phase 8 criteria already satisfied — protocol complete.")
        else:
            print("Phase 8 not yet passed — continuing.")

    while not protocol_complete:
        config = get_stage_config(current_phase, substage_idx, monkey_choice)

        phase_at_session_start    = current_phase
        substage_at_session_start = substage_idx

        pct_correct, last_trial_num, exit_reason, trial_stats = run_session(
            clock, screen, screen_info, file_path,
            phase_num=current_phase,
            substage_idx=substage_idx,
            session_num=current_session,
            monkey_name=monkey_choice,
            trial_idx=last_trial_num,
            starting_correct_trials=starting_correct,
        )

        # Record the moment the session ended (ESC or natural completion)
        session_end_dt = datetime.now()

        if exit_reason == "escaped":
            # Write session log entry for the interrupted session
            write_session_log(
                session_log_path=session_log_path,
                monkey_name=monkey_choice,
                session_num=current_session,
                phase_start=phase_at_session_start,
                substage_start=substage_at_session_start,
                phase_end=current_phase,
                substage_end=substage_idx,
                start_dt=program_start_dt,
                end_dt=session_end_dt,
                trial_stats=trial_stats,
                phase_passed=False,
                advanced_to=None,
                exit_reason="escaped",
            )
            wait_for_exit_command(screen, clock)
            break

        # ------------------------------------------------------------------
        # IN-SESSION BREAKING-POINT CHECK  (Phase 4 only)
        # Evaluated before the normal pass check. If a criterion fires, the
        # subject has reached their effort limit and leaves the ramp now,
        # regardless of whether they hit the 75% bar for this substage.
        # ------------------------------------------------------------------
        _bp_fired = False
        if current_phase == 4 and PHASE4_INSESSION_BP_CHECK:
            _bp = analyze_breaking_point(file_path, monkey_name=monkey_choice)

            if _bp["status"] == "OVERRIDE":
                # An entry in HIGH_COST_N_OVERRIDES is the experimenter's own
                # decision and outranks the ramp.
                #
                # Previously the override was only consulted when Phases 5-8
                # resolved their N, so an animal sitting in Phase 4 never saw
                # it: setting {"Uva": 6} left her climbing to N=8, 10, 12 and
                # she could only reach the override by passing the whole ramp
                # first — the exact situation it exists to short-circuit.
                #
                # No minimum-sessions guard here: that guard exists to stop one
                # bad session from ending the ramp, which does not apply to a
                # value set by hand.
                _bp_fired = True
                print_breaking_point_report(_bp, monkey_choice)
                info_screen_pygame(
                    screen, clock,
                    [f"High-cost N : {_bp['high_cost_n']}  (manual override)",
                     "",
                     "This animal has an entry in HIGH_COST_N_OVERRIDES,",
                     "so the Phase 4 ramp does not apply to them.",
                     "Leaving the ramp and moving to Phase 5.",
                     "",
                     "Remove the entry to put them back on the ramp."],
                    title="High-cost N set by override")

            elif _bp["status"] == "MEASURED":
                _stats_at_n = _bp["per_n"].get(config["hchr_n"], {})
                _n_sessions = _stats_at_n.get("n_sessions", 0)
                # Guard: never end the ramp off a single session.
                if _n_sessions >= PHASE4_BP_MIN_SESSIONS_AT_N or \
                        _bp["criterion"] == "stall":
                    _bp_fired = True
                    print_breaking_point_report(_bp, monkey_choice)
                    info_screen_pygame(
                        screen, clock,
                        [f"Criterion   : {_bp['criterion']}",
                         f"Breaking pt : N = {_bp['breaking_point']}",
                         f"High-cost N : {_bp['high_cost_n']}",
                         "",
                         "This subject has reached their effort limit.",
                         "Leaving the Phase 4 ramp and moving to Phase 5.",
                         "",
                         "The same N will be recomputed from this log at the",
                         "start of every later session, so it stays stable.",
                         "To pin it explicitly, add to HIGH_COST_N_OVERRIDES:",
                         f'    "{monkey_choice}": {_bp["high_cost_n"]},'],
                        title="Breaking point reached")
                else:
                    print(f"Breaking-point criterion '{_bp['criterion']}' met, but only "
                          f"{_n_sessions} session(s) at N={config['hchr_n']} — need "
                          f"{PHASE4_BP_MIN_SESSIONS_AT_N}. Continuing the ramp.")

        if _bp_fired:
            _phase8_details = None
            passed = True                     # leave the ramp
        elif current_phase == 8:
            # Phase 8 uses independent per-cue-type rolling windows across
            # four cue types — fundamentally different from pass_phase_check.
            passed, _phase8_details = pass_phase8_check(
                log_path=file_path,
                required_trials=config["required_trials"],
            )
        else:
            _phase8_details = None
            passed = pass_phase_check(
                log_path=file_path,
                phase_num=current_phase,
                substage_idx=substage_idx,
                pass_threshold=config["pass_threshold"],
                required_trials=config["required_trials"],
                required_sessions=config["required_sessions"],
            )

        advanced_to = None
        if passed:
            if _bp_fired:
                # Skip the remaining ramp substages entirely.
                next_phase, next_sub = 5, 0
            else:
                next_phase, next_sub = advance_stage(current_phase, substage_idx, monkey_choice)

            if next_phase == current_phase and next_sub == substage_idx:
                # Protocol fully complete (Phase 8 loop exit)
                write_session_log(
                    session_log_path=session_log_path,
                    monkey_name=monkey_choice,
                    session_num=current_session,
                    phase_start=phase_at_session_start,
                    substage_start=substage_at_session_start,
                    phase_end=current_phase,
                    substage_end=substage_idx,
                    start_dt=program_start_dt,
                    end_dt=session_end_dt,
                    trial_stats=trial_stats,
                    phase_passed=True,
                    advanced_to=(next_phase, next_sub),
                    exit_reason="protocol_complete",
                    phase8_details=_phase8_details,
                )
                show_session_result_screen(
                    screen, clock, current_phase, substage_idx,
                    current_session, pct_correct, passed=True,
                    advanced_to=(next_phase, next_sub)
                )
                protocol_complete = True
                break

            advanced_to = (next_phase, next_sub)

            write_session_log(
                session_log_path=session_log_path,
                monkey_name=monkey_choice,
                session_num=current_session,
                phase_start=phase_at_session_start,
                substage_start=substage_at_session_start,
                phase_end=current_phase,
                substage_end=substage_idx,
                start_dt=program_start_dt,
                end_dt=session_end_dt,
                trial_stats=trial_stats,
                phase_passed=True,
                advanced_to=advanced_to,
                exit_reason="completed",
                phase8_details=_phase8_details,
            )

            result = show_session_result_screen(
                screen, clock, current_phase, substage_idx,
                current_session, pct_correct, passed=True, advanced_to=advanced_to
            )
            if result == "escaped":
                wait_for_exit_command(screen, clock)
                break

            current_phase    = next_phase
            substage_idx     = next_sub
            current_session  = 1
            last_trial_num   = 0
            starting_correct = 0
            # Update start time for the next session in the same run
            program_start_dt = datetime.now()

        else:
            write_session_log(
                session_log_path=session_log_path,
                monkey_name=monkey_choice,
                session_num=current_session,
                phase_start=phase_at_session_start,
                substage_start=substage_at_session_start,
                phase_end=current_phase,
                substage_end=substage_idx,
                start_dt=program_start_dt,
                end_dt=session_end_dt,
                trial_stats=trial_stats,
                phase_passed=False,
                advanced_to=None,
                exit_reason="completed",
                phase8_details=_phase8_details,
            )

            result = show_session_result_screen(
                screen, clock, current_phase, substage_idx,
                current_session, pct_correct, passed=False, advanced_to=None
            )
            if result == "escaped":
                wait_for_exit_command(screen, clock)
                break

            current_session += 1
            last_trial_num   = 0
            starting_correct = 0
            # Update start time for the next session in the same run
            program_start_dt = datetime.now()

    if protocol_complete:
        font = pygame.font.Font(None, 72)
        screen.fill((0, 0, 0))
        surf = font.render("Training protocol complete!", True, (80, 220, 80))
        screen.blit(surf, surf.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2)))
        pygame.display.flip()
        sleep_ms(5000)

    pygame.quit()
    if 'device' in globals():
        device.close()
