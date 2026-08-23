"""Tunable parameters for the gesture controller.

Everything here is a knob. Values marked PENDING FIT are placeholders to be
fitted against the labeled fixture set (see record.py and evaluate.py), not
numbers anyone should trust. Several have already been set from live summary
statistics and had to be revised; the fixtures exist so that stops happening.
"""

MODEL_PATH = "models/hand_landmarker.task"

# --- capture ---
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

# --- detection gates ---
# A detection failing either of these never reaches the motion buffer.
# MIN_PALM_PX exists because a 6.3px "palm" was observed live, which is
# physically impossible - confidence alone does not filter junk.
MIN_CONFIDENCE = 0.6
MIN_PALM_PX = 20.0

# --- pose ---
EXTENDED_RATIO = 1.15      # tip must be this much further from the wrist than the joint

# --- motion buffer ---
# Velocity is always in PALM-LENGTHS per second, never normalized image units,
# so a threshold means the same physical gesture at 0.4m and 1.5m - a measured
# 4x swing in apparent hand size.
RING_BUFFER_S = 0.5
BUFFER_GAP_MS = 150        # a longer gap invalidates the buffer entirely
VELOCITY_LOOKBACK_S = 0.1  # window for the velocity estimate (~3 frames at 30fps)
PALM_SMOOTH_N = 5          # median over this many frames, to steady the scale unit

# --- swipe classifier ---
# PENDING FIT. Measured live: ambient motion reaches p99 = 4.5 palm/s and
# deliberate swipes sit around p95 = 3.9, so these populations OVERLAP and no
# velocity threshold separates them cleanly. Velocity is a supporting filter;
# the pose and height gates carry the discrimination.
SWIPE_MIN_VELOCITY = 4.0   # palm-lengths/sec, sustained
SWIPE_CONSISTENT_N = 4     # consecutive same-direction frames required to fire
SWIPE_MAX_Y_DRIFT = 1.5    # palm-lengths of vertical travel; rejects diagonal flails
# Coupled to the two above: a run of SWIPE_CONSISTENT_N frames at
# SWIPE_MIN_VELOCITY already travels ~0.5 palm-lengths, so any value beyond
# that delays firing past the N-frame confirmation. The spec's original 2.5
# was structurally incompatible with firing on velocity rather than on
# completed displacement.
SWIPE_MIN_DISPLACE = 0.8   # palm-lengths accumulated during the run

# Direction semantics. In an UNFLIPPED frame the presenter moving their hand to
# their own right produces DECREASING image x. False maps that to "next".
SWIPE_INVERT = False

# --- debug loop ---
DEFAULT_WORK_MS = 5.1      # stands in for inference when tracking is disabled
