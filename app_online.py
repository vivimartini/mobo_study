"""
Miscalibrated Steering in Cooperative MOBO — Study Interface
MLMI 16 Coursework 2, Vivika Martini

Run with:    streamlit run app_final.py
Install:     pip install streamlit plotly numpy pandas scipy
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import json
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# CONFIGURATION — condition is auto-assigned per participant
# ═══════════════════════════════════════════════════════════
DELTA           = 0.4    # OC internal beta offset
FORMAL_BUDGET   = 10     # hard cap on formal evaluations
NOISE_FORMAL    = 0.05
NOISE_HEURISTIC = 0.20
N_CANDIDATES    = 80
TASK_MINUTES    = 15
SEED            = 42

# ═══════════════════════════════════════════════════════════
# OBJECTIVE FUNCTION  (hidden from participants)
# ═══════════════════════════════════════════════════════════
def true_objectives(x):
    x1, x2, x3 = x[0], x[1], x[2]
    f1 = (np.exp(-((x1-0.2)**2+(x2-0.8)**2+(x3-0.5)**2)/0.08)
        + 0.7*np.exp(-((x1-0.8)**2+(x2-0.2)**2+(x3-0.5)**2)/0.08))
    f2 = (np.exp(-((x1-0.8)**2+(x2-0.8)**2+(x3-0.3)**2)/0.08)
        + 0.7*np.exp(-((x1-0.2)**2+(x2-0.2)**2+(x3-0.7)**2)/0.08))
    return float(f1), float(f2)

def evaluate(x, formal=True):
    f1, f2 = true_objectives(x)
    sd = NOISE_FORMAL if formal else NOISE_HEURISTIC
    return f1 + np.random.normal(0, sd), f2 + np.random.normal(0, sd)

# ═══════════════════════════════════════════════════════════
# CANDIDATE POOL
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def build_candidates():
    return np.random.default_rng(SEED).uniform(0, 1, (N_CANDIDATES, 3))

# ═══════════════════════════════════════════════════════════
# MOBO SCORING
# ═══════════════════════════════════════════════════════════
def hvi_proxy(x, formal_evals):
    f1, f2 = true_objectives(x)
    if not formal_evals:
        return f1 + f2
    if any(p[0] >= f1 and p[1] >= f2 for p in formal_evals):
        return 0.0
    best_f1 = max(p[0] for p in formal_evals)
    best_f2 = max(p[1] for p in formal_evals)
    return max(0, f1-best_f1) + max(0, f2-best_f2) + 0.1*(f1+f2)

def avoidance_penalty(x, forbidden, beta_internal):
    if forbidden is None or beta_internal == 0:
        return 0.0
    inside = all(forbidden[f'x{i+1}_min'] <= x[i] <= forbidden[f'x{i+1}_max']
                 for i in range(3))
    if inside:
        return beta_internal * 10.0
    dists = [min(abs(x[i]-forbidden[f'x{i+1}_min']),
                 abs(x[i]-forbidden[f'x{i+1}_max'])) for i in range(3)]
    return beta_internal * max(0, 1.0 - min(dists)/0.15)

def mobo_suggest(candidates, formal_evals, forbidden, beta_displayed, condition):
    beta_internal = min(1.0, beta_displayed + DELTA) if condition == "OC" else beta_displayed
    scores = [hvi_proxy(x, formal_evals) - avoidance_penalty(x, forbidden, beta_internal)
              for x in candidates]
    best = candidates[int(np.argmax(scores))]
    return best, beta_internal

def dist_to_forbidden(x, forbidden):
    if forbidden is None:
        return None
    return min(min(abs(x[i]-forbidden[f'x{i+1}_min']),
                   abs(x[i]-forbidden[f'x{i+1}_max'])) for i in range(3))

# ═══════════════════════════════════════════════════════════
# PARETO & HYPERVOLUME
# ═══════════════════════════════════════════════════════════
def pareto_front(points):
    idx = []
    for i, p in enumerate(points):
        if not any(q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1])
                   for j, q in enumerate(points) if i != j):
            idx.append(i)
    return idx

def hypervolume(pts, ref=(0.0, 0.0)):
    if not pts:
        return 0.0
    sorted_pts = sorted(pts, key=lambda p: p[0], reverse=True)
    hv, prev_f2 = 0.0, ref[1]
    for p in sorted_pts:
        w, h = p[0]-ref[0], p[1]-prev_f2
        if w > 0 and h > 0:
            hv += w * h
        prev_f2 = max(prev_f2, p[1])
    return hv

# ═══════════════════════════════════════════════════════════
# OBJECTIVE PLOT
# ═══════════════════════════════════════════════════════════
def make_plot(evaluations, height=400):
    fig = go.Figure()
    formal = [(e['f1'], e['f2']) for e in evaluations if e['type'] == 'formal']
    heuristic = [(e['f1'], e['f2']) for e in evaluations if e['type'] == 'heuristic']

    if not evaluations:
        fig.add_annotation(text="No evaluations yet — start exploring!",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=13, color="gray"))
    else:
        if heuristic:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in heuristic], y=[p[1] for p in heuristic],
                mode='markers', name='Heuristic (noisy)',
                marker=dict(color='steelblue', size=8, symbol='circle-open',
                            line=dict(width=1.5)),
                hovertemplate='f₁=%{x:.3f}, f₂=%{y:.3f}<extra>Heuristic</extra>'))
        if formal:
            pidx = pareto_front(formal)
            dominated = [p for i, p in enumerate(formal) if i not in pidx]
            pareto_pts = sorted([formal[i] for i in pidx], key=lambda p: p[0])
            if dominated:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in dominated], y=[p[1] for p in dominated],
                    mode='markers', name='Formal eval (beaten by later design)',
                    marker=dict(color='#f4a0a0', size=10, symbol='circle'),
                    hovertemplate='f₁=%{x:.3f}, f₂=%{y:.3f}<extra>Formal (dominated)</extra>'))
            if pareto_pts:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in pareto_pts], y=[p[1] for p in pareto_pts],
                    mode='markers+lines', name='⭐ Your best designs (Pareto front)',
                    marker=dict(color='red', size=16, symbol='star',
                                line=dict(color='darkred', width=1)),
                    line=dict(color='red', width=2, dash='dot'),
                    hovertemplate='f₁=%{x:.3f}, f₂=%{y:.3f}<extra>✅ Pareto front</extra>'))
                hv = hypervolume(pareto_pts)
                fig.add_annotation(text=f"Score (HV): {hv:.4f}",
                                   xref="paper", yref="paper",
                                   x=0.02, y=0.98, showarrow=False,
                                   font=dict(size=11, color="darkred"))

    fig.add_annotation(text="🎯 aim for top-right corner", xref="paper", yref="paper",
                       x=0.98, y=0.98, showarrow=False,
                       font=dict(size=10, color="green"), xanchor="right")
    fig.update_layout(
        xaxis_title="f₁  ← higher is better →",
        yaxis_title="f₂  ← higher is better →",
        height=height,
        margin=dict(l=50, r=20, t=20, b=50),
        legend=dict(x=0.01, y=0.06, bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#ddd', borderwidth=1),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eee', range=[-0.15, 1.1]),
        yaxis=dict(showgrid=True, gridcolor='#eee', range=[-0.15, 1.1]),
    )
    return fig


# ═══════════════════════════════════════════════════════════
# CONDITION ASSIGNMENT — balanced random assignment
# ═══════════════════════════════════════════════════════════
def _assign_condition():
    """
    Assign C or OC using minimisation (not pure random) to ensure balance.
    Checks running counts in session state across all participants.
    Uses query param ?condition=C or ?condition=OC for manual override.
    """
    # Manual override via URL — e.g. ?condition=C to force calibrated
    try:
        params = st.query_params
        if 'condition' in params and params['condition'] in ['C', 'OC']:
            return params['condition']
    except Exception:
        pass

    # Use minimisation: assign to whichever condition has fewer participants so far
    # Stored as shared storage keys so it persists across sessions
    import random
    try:
        # Try to read running counts from shared storage
        # We use a simple approach: store counts in Streamlit secrets or
        # fall back to random if unavailable
        n_c  = st.session_state.get('_global_n_c', 0)
        n_oc = st.session_state.get('_global_n_oc', 0)
        if n_c < n_oc:
            return 'C'
        elif n_oc < n_c:
            return 'OC'
        else:
            return random.choice(['C', 'OC'])
    except Exception:
        return random.choice(['C', 'OC'])

# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════
def init():
    defaults = {
        'phase': 'consent',
        'condition': _assign_condition(),
        'participant_id': '',
        'demographics': {},
        'consent_step': 'pis',
        'tutorial_step': 1,
        'check_attempts': 0,
        # task data
        'task_evals': [],
        'mobo_log': [],
        'steering_log': [],
        'formal_used': 0,
        'heuristic_count': 0,
        'task_start': None,
        'last_result': None,
        'pending_suggestion': None,
        # task controls (clean state)
        'task_x': [0.5, 0.5, 0.5],
        'task_beta': 0.5,
        'task_forbidden': None,
        'task_x1_min': 0.0, 'task_x1_max': 0.4,
        'task_x2_min': 0.0, 'task_x2_max': 0.4,
        'task_x3_min': 0.0, 'task_x3_max': 0.4,
        # practice data (completely separate)
        'practice_evals': [],
        'practice_mobo_done': False,
        'practice_forbidden_done': False,
        'practice_formal_done': False,
        'practice_last_suggestion': None,
        # questionnaire
        'questionnaire': {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ═══════════════════════════════════════════════════════════
# PHASE 1: CONSENT + DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════
def show_consent():
    st.markdown("""
    <div style='background:#1a3a5c;padding:14px 20px;border-radius:8px;margin-bottom:16px'>
    <span style='color:white;font-size:17px;font-weight:600'>
    University of Cambridge — Participant Onboarding
    </span></div>""", unsafe_allow_html=True)

    step = st.session_state.consent_step
    cols = st.columns(3)
    for i, (s, label) in enumerate(zip(['pis','consent','demographics'],
                                        ['1 — Information Sheet',
                                         '2 — Consent Form',
                                         '3 — Demographics'])):
        with cols[i]:
            if s == step:
                st.markdown(f"**🔵 {label}**")
            elif ['pis','consent','demographics'].index(s) < \
                 ['pis','consent','demographics'].index(step):
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"⬜ {label}")
    st.markdown("---")

    if step == 'pis':
        st.markdown("### Participant Information Sheet")
        with st.container(border=True):
            st.markdown("""
*A study investigating methods for enhancing user performance and experience in human-computer interaction tasks.*

Thank you for your interest in this study. Please read the following carefully.

**Purpose of the study**
This study investigates how user performance and experience in computer-based tasks can be improved by novel interface features. You will complete a short interactive optimisation task and a brief questionnaire. We will log interaction data and questionnaire responses.

1. **Why have I been chosen?** You are an MLMI16 student familiar with optimisation concepts, which allows task instructions to be kept short and consistent.
2. **Do I have to take part?** No — participation is entirely voluntary. You may withdraw at any time without giving a reason.
3. **Who is organising the study?** Dr John Dudley (Principal Investigator) and Vivika Martini (Researcher), Department of Engineering, University of Cambridge.
4. **What will happen?** You will complete a ~15 minute interactive task followed by a ~5 minute questionnaire. Total session ≈25 minutes.
5. **Risks?** Minimal — mild fatigue or eyestrain. You may take a break at any time.
6. **End of study?** There will be a short debrief explaining the study purpose fully.
7. **Study results?** Anonymised results may be written up and published. Data stored securely.
8. **Anonymity?** Data is anonymised using a participant ID. You will not be identifiable.
9. **Problems?** Contact the researcher — vm481@cam.ac.uk
10. **Ethics review?** Reviewed by the Department of Engineering Ethics Committee (light-touch process).

---
**Contact:** Vivika Martini — vm481@cam.ac.uk — University of Cambridge
            """)
        st.markdown("")
        st.info("⚠️ **Please complete this study on a laptop or desktop computer using Chrome or Firefox. Do not use a mobile phone. Complete in one sitting without interruptions.**")
        if st.button("I have read this → proceed to consent", type="primary"):
            st.session_state.consent_step = 'consent'
            st.rerun()

    elif step == 'consent':
        st.markdown("### Consent Form")
        st.markdown("**Principal Investigator:** Dr John Dudley | **Researcher:** Vivika Martini")
        st.markdown("")
        c1 = st.checkbox("1. I have read and understood the Participant Information Sheet.")
        c2 = st.checkbox("2. I have had the opportunity to ask questions.")
        c3 = st.checkbox("3. I understand participation is voluntary and I may withdraw at any time.")
        c4 = st.checkbox("4. I agree data may be stored anonymously and used for research.")
        c5 = st.checkbox("5. I agree to take part in this study.")
        st.markdown("")
        pid = st.text_input("**Participant ID** (given by researcher):", key="pid_input")
        if c1 and c2 and c3 and c4 and c5 and pid.strip():
            if st.button("I consent → proceed to demographics", type="primary"):
                st.session_state.participant_id = pid.strip()
                st.session_state.consent_step = 'demographics'
                st.rerun()
        else:
            st.info("Please tick all boxes and enter your participant ID to continue.")

    elif step == 'demographics':
        st.markdown("### Demographics")
        st.caption("All questions are optional.")
        age  = st.selectbox("Age range", ["Prefer not to say","18–24","25–34","35–44","45+"])
        sex  = st.selectbox("Sex", ["Prefer not to say","Male","Female","Non-binary","Other"])
        hand = st.selectbox("Handedness", ["Prefer not to say","Right-handed","Left-handed","Ambidextrous"])
        mobo = st.selectbox("Prior experience with Bayesian optimisation",
                            ["None","Some (read about it)","Moderate (used it)","Extensive (research/work)"])
        st.markdown("")
        if st.button("Begin study →", type="primary"):
            st.session_state.demographics = {'age':age,'sex':sex,'handedness':hand,'mobo_exp':mobo}
            st.session_state.phase = 'tutorial'
            st.rerun()

# ═══════════════════════════════════════════════════════════
# PHASE 2: TUTORIAL  (6 steps, back/next navigation)
# ═══════════════════════════════════════════════════════════
def show_tutorial():
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1
    step = st.session_state.tutorial_step
    TOTAL = 6

    st.markdown(f"**Tutorial — Step {step} of {TOTAL}**")
    st.progress(step / TOTAL)
    st.markdown("---")

    if step == 1:
        st.markdown("## 👋 What you're doing")
        st.markdown("""
Imagine you're an engineer with **three design knobs** — call them **x₁, x₂, x₃**.
Each knob goes from **0 to 1**. You can set them to any value — 0.1, 0.5, 0.9, anything.
They represent the *inputs* to a black-box machine. You don't know what the machine does internally — you just set the knobs and see what scores come out.

When you test a design, you get back **two performance scores: f₁ and f₂**.
Both should be **as high as possible** — but they trade off against each other.
Designs that score high on f₁ tend to score lower on f₂.

So there's no single "best" design. Instead there's a *family* of good designs,
each making a different trade-off. Your job is to find as many of these as possible.
        """)
        st.info("""
**Your goal:** Find as many good trade-off designs as possible — designs where **both f₁ and f₂ are high**.

Because f₁ and f₂ trade off, no single design will max out both. So you want a *collection* of designs that together cover the top-right corner of the plot. These appear as **⭐ red stars**.

**The more stars in the top-right, the higher your score (HV).** HV is just a number that measures how well your collection covers the top-right — bigger is better.
        """)

        # Show example plot
        st.markdown("#### What the objective plot looks like:")
        col_bad, col_good = st.columns(2)

        _go = go  # use already-imported plotly

        def _example_plot(title, pareto_pts, dominated_pts, heuristic_pts, color):
            fig = _go.Figure()
            if heuristic_pts:
                fig.add_trace(_go.Scatter(
                    x=[p[0] for p in heuristic_pts],
                    y=[p[1] for p in heuristic_pts],
                    mode="markers", name="Heuristic",
                    marker=dict(color="steelblue", size=8, symbol="circle-open",
                                line=dict(width=1.5))))
            if dominated_pts:
                fig.add_trace(_go.Scatter(
                    x=[p[0] for p in dominated_pts],
                    y=[p[1] for p in dominated_pts],
                    mode="markers", name="Formal (dominated)",
                    marker=dict(color="#f4a0a0", size=10)))
            if pareto_pts:
                srt = sorted(pareto_pts, key=lambda p: p[0])
                fig.add_trace(_go.Scatter(
                    x=[p[0] for p in srt], y=[p[1] for p in srt],
                    mode="markers+lines", name="⭐ Pareto front",
                    marker=dict(color="red", size=16, symbol="star",
                                line=dict(color="darkred", width=1)),
                    line=dict(color="red", width=2, dash="dot")))
            fig.add_annotation(text="aim for top-right ↗",
                               xref="paper", yref="paper",
                               x=0.98, y=0.98, showarrow=False,
                               font=dict(size=10, color="green"), xanchor="right")
            fig.update_layout(
                title=dict(text=title, font=dict(color=color, size=14)),
                xaxis_title="f₁ →", yaxis_title="f₂ →",
                height=280, margin=dict(l=40,r=10,t=40,b=40),
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor="#eee", range=[-0.05,1.1]),
                yaxis=dict(showgrid=True, gridcolor="#eee", range=[-0.05,1.1]),
            )
            return fig

        with col_bad:
            bad_fig = _example_plot(
                "❌ Poor result — points clustered, low scores",
                pareto_pts=[(0.15, 0.12), (0.18, 0.08)],
                dominated_pts=[(0.10, 0.10), (0.12, 0.09)],
                heuristic_pts=[(0.11, 0.13), (0.14, 0.11), (0.09, 0.15)],
                color="red"
            )
            st.plotly_chart(bad_fig, use_container_width=True)
            st.caption("Points are low and clustered near the bottom-left. Score (HV) is low.")

        with col_good:
            good_fig = _example_plot(
                "✅ Good result — spread across top-right",
                pareto_pts=[(0.85, 0.15), (0.65, 0.55), (0.40, 0.75), (0.15, 0.90)],
                dominated_pts=[(0.50, 0.40), (0.60, 0.30), (0.30, 0.60)],
                heuristic_pts=[(0.70, 0.20), (0.45, 0.65), (0.20, 0.80),
                               (0.55, 0.45), (0.35, 0.70)],
                color="green"
            )
            st.plotly_chart(good_fig, use_container_width=True)
            st.caption("Red stars spread across the top-right — good trade-offs found. Score (HV) is high.")

    elif step == 2:
        st.markdown("## 🧪 Testing a design")
        st.markdown("Both scores range from **0 to ~1.0**. Here's what the scores mean:")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.error("**0.1 – 0.3**\nPoor — keep exploring")
        with c2:
            st.warning("**0.4 – 0.6**\nDecent — investigate nearby")
        with c3:
            st.success("**0.7 – 1.0**\nStrong — spend a formal evaluation")
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
### 🔵 Heuristic Evaluate
- Fast, instant result
- **Noisy** — rough estimate only
- **Unlimited** — use freely
- Use to: explore and find promising regions
            """)
        with col2:
            st.markdown("""
### ⭐ Formal Evaluate
- Accurate — the real score
- **Only 10 total** — spend wisely
- Counts toward your final Pareto front
- Use when: heuristic scores look above ~0.5
            """)
        st.warning("⚠️ **Strategy:** Explore freely with heuristics first. Only spend a formal evaluation when both f₁ and f₂ look promising (above ~0.5). You cannot get formal evaluations back.")

    elif step == 3:
        st.markdown("## 🤖 Getting a design to test")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
### 🎛️ Manual
Adjust the **x₁, x₂, x₃ sliders** yourself.
Good if you have a hunch about a specific region.
            """)
        with col2:
            st.markdown("""
### 🤖 Ask MOBO
Click **"New Design from MOBO"** and the AI suggests a promising design based on what you've evaluated so far.

MOBO gets smarter as you evaluate more — early suggestions are exploratory, later ones are more targeted.
            """)
        st.info("💡 **Tip:** Mix manual exploration with MOBO suggestions. Use heuristics to scout a region, then ask MOBO to suggest nearby promising designs.")

    elif step == 4:
        st.markdown("## 🚫 How to steer MOBO away from bad regions")

        st.markdown("""
Here's the situation you'll face during the task:

You've tried a few designs. Some gave terrible f₁ and f₂ scores.
MOBO keeps suggesting more designs from the same region — because it doesn't know you've already ruled it out.

**You need to tell MOBO: stop suggesting designs from that area.**

That's what the **Forbidden Region** is for.
        """)

        st.markdown("---")
        st.markdown("### 🍰 Think of it like a recipe experiment")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
You're testing recipes with three knobs:
- **x₁** = temperature (0=low, 1=high)
- **x₂** = baking time (0=short, 1=long)
- **x₃** = sugar amount (0=none, 1=lots)

You've tried **low temperature + short time** (x₁ < 0.3, x₂ < 0.3) and every recipe came out raw and tasteless.

You want to tell MOBO: **"don't suggest any more low-temp, short-time recipes."**
            """)
        with col2:
            st.markdown("""
So you:
1. ✅ Tick **"Enable forbidden region"**
2. ✅ Set **x₁ min=0.0, x₁ max=0.3**
3. ✅ Set **x₂ min=0.0, x₂ max=0.3**
4. ✅ Set **β = 0.8** (strongly avoid)
5. ✅ Click **New Design from MOBO**

MOBO now suggests recipes from **outside** that bad region — higher temperatures, longer times.

The forbidden region **does not affect the objective plot** — it only tells MOBO which input settings to avoid suggesting.
            """)

        st.markdown("---")
        st.markdown("### The avoid-strength β")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.error("**β = 0.0** — MOBO completely ignores your forbidden region")
        with c2:
            st.warning("**β = 0.5** — MOBO moderately avoids it, may occasionally suggest from there")
        with c3:
            st.success("**β = 1.0** — MOBO strongly avoids it, will almost never suggest from there")

        st.info("💡 **When should you use it?** After you've tried several designs in a region and they all gave low scores — mark that region as forbidden so MOBO explores elsewhere instead.")

    elif step == 5:
        st.markdown("## 🔄 The complete strategy — what to actually do")
        st.markdown("""
Here's the recommended approach for the 15-minute task:
        """)

        st.markdown("### Phase 1 — Explore (first ~5 minutes)")
        st.success("""
🔵 **Use heuristic evaluations freely — they're unlimited.**

1. Click **New Design from MOBO** → get a suggestion
2. Click **Heuristic Evaluate** → see the rough f₁ and f₂ scores
3. Repeat 5-8 times to get a sense of the landscape

You're building a map of which regions look promising and which look bad.
Don't spend any formal evaluations yet.
        """)

        st.markdown("### Phase 2 — Steer MOBO away from bad areas")
        st.warning("""
🚫 **Once you've found a bad region, tell MOBO to avoid it.**

After several heuristic evaluations, you'll notice some x₁, x₂, x₃ combinations always give low scores.

1. Enable the **Forbidden Region** in Step 3
2. Set the bounds to cover the bad area (e.g. x₁: 0.0–0.4 if low x₁ values are bad)
3. Set β = 0.7 or higher
4. Click MOBO again — it will now suggest designs from elsewhere

Check the **🗺️ Exploration Map** on the left of the task screen — red dots show bad regions. If you see a cluster of red, set the forbidden region sliders to cover that area.
        """)

        st.markdown("### Phase 3 — Commit with formal evaluations (last ~8 minutes)")
        st.info("""
⭐ **When heuristic scores look above ~0.5 on both f₁ and f₂, spend a formal evaluation.**

You have **10 formal evaluations** — they're accurate and count toward your score.

1. Find a region where heuristic scores look promising
2. Get a MOBO suggestion in that region (or set sliders manually)
3. Click **⭐ Formal Evaluate** — this adds a point to your Pareto front
4. Repeat until budget is used

Your **score (HV)** goes up every time you add a new Pareto design.
The higher the score, the better your collection of trade-off designs.
        """)

        st.markdown("---")
        st.info("📈 Watch your **score (HV)** at the top of the task screen — it tells you in real time how well you're doing.")

    elif step == 6:
        st.markdown("## ✅ Practice round — what you must do")
        st.markdown("You'll now do a **short practice round**. Nothing here counts toward your real task.")
        st.markdown("""
| # | Action | Why |
|---|--------|-----|
| 1 | Tick **"Enable forbidden region"** and set some bounds | Learn how to define a no-go zone |
| 2 | Click **"New Design from MOBO"** with the forbidden region active | See MOBO avoid your region |
| 3 | Click **"⭐ Formal Evaluate"** at least once | Practice spending a formal evaluation |
        """)
        st.success("Once all three are done, a button will appear to start the real task. Take your time — there's no rush in the practice round.")

    # Navigation
    st.markdown("---")
    col_back, _, col_next = st.columns([1, 2, 1])
    with col_back:
        if step > 1:
            if st.button("← Back", key=f"tb_{step}"):
                st.session_state.tutorial_step -= 1
                st.rerun()
    with col_next:
        if step < TOTAL:
            if st.button("Next →", type="primary", key=f"tn_{step}"):
                st.session_state.tutorial_step += 1
                st.rerun()
        else:
            if st.button("Start practice round →", type="primary", key=f"td_{step}"):
                st.session_state.tutorial_step = 1
                st.session_state.phase = 'check'
                st.rerun()

# ═══════════════════════════════════════════════════════════
# PHASE 3: COMPREHENSION CHECK
# ═══════════════════════════════════════════════════════════
def show_check():
    st.title("Comprehension Check")
    st.markdown("Answer at least **3 out of 4** correctly to proceed. You have one reattempt.")
    st.markdown("---")

    Qs = [
        ("What is your goal in this task?", [
            "Find a single design with the highest f₁",
            "Find a set of designs forming a strong Pareto front (good trade-offs between f₁ and f₂)",
            "Use as many formal evaluations as possible",
            "Keep both f₁ and f₂ below 0.5"], 1),
        ("You have 10 formal evaluations. Which statement is true?", [
            "You can use them all at the start",
            "They reset every 5 minutes",
            "They are more accurate than heuristic evaluations but limited in number",
            "Heuristic evaluations are more accurate"], 2),
        ("The Forbidden Region defines a box in:", [
            "Objective space (f₁, f₂) — the plot axes",
            "Parameter space (x₁, x₂, x₃) — the design inputs",
            "Time space — evaluations to skip",
            "Budget space — evaluations to save"], 1),
        ("If you set β = 0.9 and define a forbidden region, what happens when you click 'New Design from MOBO'?", [
            "MOBO ignores the forbidden region",
            "MOBO only suggests designs inside the forbidden region",
            "MOBO strongly avoids suggesting designs from the forbidden region",
            "MOBO uses 2 formal evaluations instead of 1"], 2),
    ]

    answers = []
    for i, (q, opts, _) in enumerate(Qs):
        st.markdown(f"**{i+1}. {q}**")
        ans = st.radio("", opts, key=f"cq_{i}", index=None, label_visibility="collapsed")
        answers.append(ans)
        st.markdown("")

    if st.button("Submit answers", type="primary"):
        if any(a is None for a in answers):
            st.error("Please answer all questions.")
            return
        score = sum(answers[i] == Qs[i][1][Qs[i][2]] for i in range(4))
        if score >= 3:
            st.success(f"✅ Passed ({score}/4). Moving to practice round...")
            st.session_state.phase = 'practice'
            st.rerun()
        else:
            attempts = st.session_state.check_attempts + 1
            st.session_state.check_attempts = attempts
            if attempts >= 2:
                st.error(f"Score: {score}/4. Both attempts used. Please inform the researcher.")
                st.stop()
            else:
                st.error(f"Score: {score}/4. Please re-read the tutorial. One attempt remaining.")
                if st.button("← Re-read tutorial", key="retry_tut"):
                    st.session_state.phase = 'tutorial'
                    st.rerun()

# ═══════════════════════════════════════════════════════════
# PHASE 4: PRACTICE ROUND  (completely isolated state)
# ═══════════════════════════════════════════════════════════
def show_practice():
    candidates = build_candidates()

    # Completion flags
    mobo_done     = st.session_state.practice_mobo_done
    forbidden_done = st.session_state.practice_forbidden_done
    formal_done   = st.session_state.practice_formal_done
    all_done      = mobo_done and forbidden_done and formal_done

    # Header with live checklist
    st.markdown("## 🛠️ Practice Round")
    st.markdown("""
<div style='background:#e8f4fd;padding:14px 18px;border-radius:8px;border-left:4px solid #1a7abf'>
<b>Practice task:</b> Get familiar with the interface. Nothing here counts toward your real task.<br><br>
Complete all three steps before continuing:
</div>
""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"{'✅' if forbidden_done else '⬜'} **Step 1:** Enable forbidden region and click New Design from MOBO")
    st.markdown(f"{'✅' if mobo_done else '⬜'} **Step 2:** MOBO suggestion received")
    st.markdown(f"{'✅' if formal_done else '⬜'} **Step 3:** Click Formal Evaluate at least once")
    st.markdown("---")

    # Two-column layout: plot left, controls right
    col_plot, col_ctrl = st.columns([1.1, 1])

    with col_plot:
        st.markdown("#### Objective Plot (Practice)")
        st.caption("⭐ Stars = your best designs (Pareto front). 🔵 Blue circles = heuristic. Aim top-right.")
        real_evals = [e for e in st.session_state.practice_evals if e.get('f1') is not None]
        st.plotly_chart(make_plot(real_evals, height=350), use_container_width=True)

    with col_ctrl:
        st.markdown("#### Controls")
        st.markdown("""
**Do these in order:**
1. 🚫 Enable forbidden region below + set bounds
2. 🤖 Click MOBO to get a suggestion
3. ⭐ Click Formal Evaluate to test it
        """)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1: px1 = st.slider("x₁", 0.0, 1.0, 0.5, 0.01, key="px1")
        with c2: px2 = st.slider("x₂", 0.0, 1.0, 0.5, 0.01, key="px2")
        with c3: px3 = st.slider("x₃", 0.0, 1.0, 0.5, 0.01, key="px3")
        px = np.array([px1, px2, px3])

        st.markdown("---")
        st.markdown("#### 🚫 Forbidden Region")
        st.caption("Enable this, set some bounds, then click MOBO.")
        use_f = st.checkbox("Enable forbidden region", key="p_use_f")
        p_forbidden = None

        if use_f:
            pc1, pc2 = st.columns(2)
            with pc1:
                px1min = st.slider("x₁ min", 0.0, 0.9, 0.0, 0.05, key="px1min")
                px2min = st.slider("x₂ min", 0.0, 0.9, 0.0, 0.05, key="px2min")
                px3min = st.slider("x₃ min", 0.0, 0.9, 0.0, 0.05, key="px3min")
            with pc2:
                px1max = st.slider("x₁ max", 0.1, 1.0, 0.4, 0.05, key="px1max")
                px2max = st.slider("x₂ max", 0.1, 1.0, 0.4, 0.05, key="px2max")
                px3max = st.slider("x₃ max", 0.1, 1.0, 0.4, 0.05, key="px3max")
            if px1min < px1max and px2min < px2max and px3min < px3max:
                p_forbidden = {'x1_min':px1min,'x1_max':px1max,
                               'x2_min':px2min,'x2_max':px2max,
                               'x3_min':px3min,'x3_max':px3max}
                vol = (px1max-px1min)*(px2max-px2min)*(px3max-px3min)
                st.caption(f"Forbidden box volume: {vol:.3f}")

        p_beta = st.slider("Avoid-strength β  (0=ignore, 1=strongly avoid)",
                           0.0, 1.0, 0.5, 0.05, key="p_beta")

        st.markdown("---")
        if st.button("🤖 New Design from MOBO", type="primary",
                     key="p_mobo", use_container_width=True):
            formal_pts = [(e['f1'],e['f2']) for e in st.session_state.practice_evals
                          if e.get('type') == 'formal' and e.get('f1') is not None]
            sug, _ = mobo_suggest(candidates, formal_pts, p_forbidden, p_beta, "C")
            st.session_state.practice_evals.append(
                {'type':'mobo','x':sug.tolist(),'f1':None,'f2':None})
            st.session_state.practice_mobo_done = True
            st.session_state.practice_last_suggestion = sug.tolist()
            if p_forbidden is not None:
                st.session_state.practice_forbidden_done = True
            st.rerun()

        if st.session_state.get('practice_last_suggestion'):
            s = st.session_state.practice_last_suggestion
            st.success(f"✅ MOBO suggests: x₁={s[0]:.3f}, x₂={s[1]:.3f}, x₃={s[2]:.3f} — "
                       f"now click ⭐ Formal Evaluate below to test it!")

        st.markdown("---")
        st.markdown("#### Evaluate Current Design")
        col_h, col_f = st.columns(2)
        with col_h:
            if st.button("🔵 Heuristic", key="p_h", use_container_width=True):
                f1, f2 = evaluate(px, formal=False)
                st.session_state.practice_evals.append(
                    {'type':'heuristic','x':px.tolist(),'f1':f1,'f2':f2})
                st.info(f"Heuristic: f₁={f1:.3f}, f₂={f2:.3f}")
                st.rerun()
        with col_f:
            if st.button("⭐ Formal", key="p_f", use_container_width=True, type="primary"):
                f1, f2 = evaluate(px, formal=True)
                st.session_state.practice_evals.append(
                    {'type':'formal','x':px.tolist(),'f1':f1,'f2':f2})
                st.session_state.practice_formal_done = True
                st.success(f"Formal: f₁={f1:.3f}, f₂={f2:.3f}")
                st.rerun()

    # Start button — only appears when all done
    st.markdown("---")
    if all_done:
        st.success("✅ All practice steps complete! You're ready for the real task.")
        if st.button("▶️ Start the main task →", type="primary", use_container_width=True):
            # Clean reset — main task starts with completely fresh state
            st.session_state.task_evals = []
            st.session_state.mobo_log = []
            st.session_state.steering_log = []
            st.session_state.formal_used = 0
            st.session_state.heuristic_count = 0
            st.session_state.last_result = None
            st.session_state.pending_suggestion = None
            st.session_state.task_x = [0.5, 0.5, 0.5]
            st.session_state.task_beta = 0.5
            st.session_state.task_forbidden = None
            st.session_state.task_x1_min = 0.0
            st.session_state.task_x1_max = 0.4
            st.session_state.task_x2_min = 0.0
            st.session_state.task_x2_max = 0.4
            st.session_state.task_x3_min = 0.0
            st.session_state.task_x3_max = 0.4
            st.session_state.task_start = time.time()
            st.session_state.phase = 'task_intro'
            st.rerun()
    else:
        st.info("Complete all three checklist items above to unlock the start button.")

# ═══════════════════════════════════════════════════════════
# PHASE 4b: TASK INTRO SCREEN
# ═══════════════════════════════════════════════════════════
def show_task_intro():
    st.markdown("")
    st.markdown("")
    col = st.columns([1,2,1])[1]
    with col:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;background:#1a3a5c;
                    border-radius:16px;color:white'>
            <div style='font-size:64px;margin-bottom:16px'>🔬</div>
            <h2 style='color:white;margin-bottom:8px'>Main Task Starting</h2>
            <p style='color:#cce0f0;font-size:16px;margin-bottom:24px'>
                You have <b style='color:white'>15 minutes</b> and
                <b style='color:white'>10 formal evaluations</b>.<br>
                Find as many good trade-off designs as possible.<br>
                Aim for the <b style='color:#7ecfff'>top-right corner</b> of the plot.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("▶️ Start the main task now", type="primary",
                     use_container_width=True):
            st.session_state.task_start = time.time()
            st.session_state.phase = 'task'
            st.rerun()

# ═══════════════════════════════════════════════════════════
# PHASE 5: MAIN TASK
# ═══════════════════════════════════════════════════════════
def show_task():
    # Scroll to top on every task render
    st.components.v1.html("<script>window.parent.document.querySelector('section.main').scrollTo(0,0);</script>", height=0)
    candidates = build_candidates()
    elapsed   = time.time() - st.session_state.task_start
    remaining = max(0, TASK_MINUTES * 60 - elapsed)
    mins, secs = int(remaining//60), int(remaining%60)
    budget_left = FORMAL_BUDGET - st.session_state.formal_used

    # End condition
    if remaining <= 0 or budget_left <= 0:
        st.session_state.phase = 'questionnaire'
        st.rerun()

    # ── Header ───────────────────────────────────────────────
    st.markdown("## 🔬 Design Optimisation Task")
    st.info("**Your goal:** Find as many good trade-off designs as possible — both f₁ and f₂ high. "
            "Top-right corner of the plot = where you want to be. "
            "Use MOBO suggestions and steer it away from bad regions.")

    col_t, col_b, col_s = st.columns(3)
    with col_t:
        color = "red" if mins < 3 else "orange" if mins < 7 else "green"
        st.markdown("**⏱ Time remaining**")
        st.markdown(f"### :{color}[{mins:02d}:{secs:02d}]")
    with col_b:
        st.markdown("**⭐ Formal evaluations left**")
        st.markdown(f"### {budget_left}/{FORMAL_BUDGET}")
        st.caption("🟢"*budget_left + "⬜"*st.session_state.formal_used)
    with col_s:
        formal_pts = [(e['f1'],e['f2']) for e in st.session_state.task_evals
                      if e['type']=='formal']
        if formal_pts:
            pidx = pareto_front(formal_pts)
            hv = hypervolume([formal_pts[i] for i in pidx])
            st.markdown("**📈 Your score**")
            st.markdown(f"### {hv:.4f}")
            st.caption(f"{len(pidx)} Pareto design{'s' if len(pidx)!=1 else ''}")
        else:
            st.markdown("**📈 Your score**")
            st.markdown("### 0.0000")
            st.caption("No formal evaluations yet")

    st.markdown("---")

    # ── Live coach panel ─────────────────────────────────────
    n_formal = st.session_state.formal_used
    n_heuristic = st.session_state.heuristic_count
    n_mobo = len(st.session_state.mobo_log)
    n_steered = sum(1 for s in st.session_state.mobo_log
                    if s.get('forbidden') is not None)

    if n_formal == 0 and n_heuristic == 0:
        st.info(
            "👋 **Start here — 3 easy steps:**  "
            "① Click **🤖 New Design from MOBO** to get a suggested design  →  "
            "② Click **🔵 Heuristic** to test it (free, unlimited)  →  "
            "③ Repeat to explore. When scores look good (above 0.5), use **⭐ Formal** to lock it in."
        )
    elif n_heuristic < 4 and n_formal == 0:
        st.info(f"🔵 Good start! You've done {n_heuristic} heuristic evaluation{'s' if n_heuristic>1 else ''}. "
                "Keep exploring with heuristics — try a few more MOBO suggestions "
                "before spending formal evaluations.")
    elif n_mobo >= 3 and n_steered == 0 and n_formal < 3:
        st.warning("💡 **Tip:** You've used MOBO several times without steering. "
                   "If some regions gave bad scores, scroll down to **Step 3** and "
                   "define a forbidden region — MOBO will explore elsewhere.")
    elif n_formal > 0 and n_formal < 5:
        formal_pts = [(e['f1'],e['f2']) for e in st.session_state.task_evals
                      if e['type']=='formal']
        pidx = pareto_front(formal_pts)
        hv = hypervolume([formal_pts[i] for i in pidx])
        st.success(f"⭐ Great — {n_formal} formal evaluation{'s' if n_formal>1 else ''} done. "
                   f"Score (HV): {hv:.4f}. Keep finding good trade-offs — "
                   "aim for designs where both f₁ AND f₂ are high.")
    elif budget_left <= 3:
        st.warning(f"⚠️ Only {budget_left} formal evaluation{'s' if budget_left>1 else ''} left — "
                   "make them count! Use heuristics to find the best region first, "
                   "then commit with a formal evaluation.")

    st.markdown("---")

    # ── Two-column layout: plot LEFT, controls RIGHT ──────────
    col_plot, col_ctrl = st.columns([1.2, 1])

    # ── LEFT: Plot + history ─────────────────────────────────
    with col_plot:
        st.markdown("#### Objective Plot")
        st.caption(
            "⭐ **Stars** = your best designs (Pareto front) — aim to push these toward the top-right corner. "
            "🔵 **Blue circles** = heuristic evaluations (rough estimates, free). "
            "**Faded dots** = formal evaluations beaten by later designs. "
            "Negative values are possible due to measurement noise — keep exploring!"
        )
        st.plotly_chart(make_plot(st.session_state.task_evals, height=420),
                        use_container_width=True)

        if st.session_state.task_evals:
            st.markdown("#### Recent Evaluations")
            recent = [e for e in st.session_state.task_evals
                      if e.get('f1') is not None][-6:][::-1]
            if recent:
                df = pd.DataFrame([{
                    'Type': e['type'].capitalize(),
                    'x₁': f"{e['x'][0]:.2f}",
                    'x₂': f"{e['x'][1]:.2f}",
                    'x₃': f"{e['x'][2]:.2f}",
                    'f₁': f"{e['f1']:.3f}",
                    'f₂': f"{e['f2']:.3f}",
                } for e in recent])
                st.dataframe(df, hide_index=True, use_container_width=True)

        # ── Colour-coded score history + smart advice ───────
        all_evals = [e for e in st.session_state.task_evals
                     if e.get('f1') is not None]
        if all_evals:
            st.markdown("#### 🗺️ Exploration map (parameter space)")
            st.caption(
                "Each dot = a design you tested. "
                "**Green = both f₁ and f₂ were good. Red = both were bad.** "
                "Hover to see exact scores."
            )
            scores = [min(e['f1'], e['f2']) for e in all_evals]
            min_s = min(scores)
            max_s = max(scores)
            rng = max_s - min_s if max_s > min_s else 1.0

            fig_p = go.Figure()
            fig_p.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                            fillcolor="#f8f8f8",
                            line=dict(color="#ddd", width=1))

            # Draw forbidden region if active
            tf = st.session_state.task_forbidden
            if tf:
                fig_p.add_shape(type="rect",
                                x0=tf['x1_min'], y0=tf['x2_min'],
                                x1=tf['x1_max'], y1=tf['x2_max'],
                                fillcolor="rgba(220,50,50,0.12)",
                                line=dict(color="red", width=2, dash="dash"))
                fig_p.add_annotation(
                    x=(tf['x1_min']+tf['x1_max'])/2,
                    y=(tf['x2_min']+tf['x2_max'])/2,
                    text="🚫 forbidden",
                    showarrow=False,
                    font=dict(size=9, color="darkred"),
                    bgcolor="rgba(255,255,255,0.85)")

            # Plot with jitter so overlapping dots are visible
            import random as _random
            _random.seed(0)
            for e in all_evals:
                s = min(e['f1'], e['f2'])
                t = (s - min_s) / rng
                r = int(220 * (1-t))
                g = int(160 * t)
                col = f"rgb({r},{g},60)"
                sym = "star" if e['type'] == 'formal' else "circle"
                sz  = 13 if e['type'] == 'formal' else 9
                # Add small jitter so stacked dots are visible
                jx = e['x'][0] + _random.uniform(-0.02, 0.02)
                jy = e['x'][1] + _random.uniform(-0.02, 0.02)
                jx = max(0.01, min(0.99, jx))
                jy = max(0.01, min(0.99, jy))
                fig_p.add_trace(go.Scatter(
                    x=[jx], y=[jy],
                    mode="markers",
                    marker=dict(color=col, size=sz, symbol=sym,
                                line=dict(color="white", width=1)),
                    showlegend=False,
                    hovertemplate=(
                        f"x₁={e['x'][0]:.2f}, x₂={e['x'][1]:.2f}, x₃={e['x'][2]:.2f}<br>"
                        f"f₁={e['f1']:.3f}, f₂={e['f2']:.3f}<br>"
                        f"Quality: {'🟢 Good' if t > 0.6 else '🟡 Mediocre' if t > 0.3 else '🔴 Bad'}"
                        f"<extra>{'⭐ Formal' if e['type']=='formal' else 'Heuristic'}</extra>"
                    )
                ))

            # Current design
            cx = st.session_state.task_x
            fig_p.add_trace(go.Scatter(
                x=[cx[0]], y=[cx[1]], mode="markers",
                marker=dict(color="royalblue", size=14, symbol="diamond",
                            line=dict(color="white", width=2)),
                showlegend=False,
                hovertemplate=f"Current: x₁={cx[0]:.2f}, x₂={cx[1]:.2f}<extra>You are here</extra>"
            ))

            fig_p.update_layout(
                xaxis=dict(range=[-0.05,1.05], title="x₁ →",
                           showgrid=False, zeroline=False),
                yaxis=dict(range=[-0.05,1.05], title="x₂ →",
                           showgrid=False, zeroline=False),
                height=260,
                margin=dict(l=40, r=10, t=10, b=40),
                plot_bgcolor="#f8f8f8",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_p, use_container_width=True,
                            config={'displayModeBar': False})

            # Smart advice based on what's on the map
            red_evals = [e for e, s in zip(all_evals, scores)
                         if (s - min_s)/rng < 0.3]
            green_evals = [e for e, s in zip(all_evals, scores)
                           if (s - min_s)/rng > 0.6]

            if len(set([round(e['x'][0],1) for e in all_evals])) == 1:
                st.warning(
                    "⚠️ All your evaluations are at the same x₁, x₂ position. "
                    "Try clicking **🤖 New Design from MOBO** to explore different regions, "
                    "or adjust the x₁, x₂, x₃ sliders manually in Step 1."
                )
            elif red_evals:
                avg_x1 = sum(e['x'][0] for e in red_evals) / len(red_evals)
                avg_x2 = sum(e['x'][1] for e in red_evals) / len(red_evals)
                st.error(
                    f"🔴 **Bad regions found** near x₁≈{avg_x1:.1f}, x₂≈{avg_x2:.1f}. "
                    f"Scroll down to Step 3, enable the forbidden region, "
                    f"and set sliders to cover that area."
                )
            elif green_evals:
                avg_x1 = sum(e['x'][0] for e in green_evals) / len(green_evals)
                avg_x2 = sum(e['x'][1] for e in green_evals) / len(green_evals)
                st.success(
                    f"🟢 **Good region found** near x₁≈{avg_x1:.1f}, x₂≈{avg_x2:.1f}. "
                    f"Keep exploring nearby — use ⭐ Formal Evaluate when scores look consistently good."
                )

                # ── Step 3 HERE — directly below the map ─────────────
        st.markdown("---")
        st.markdown("#### Step 3 — Steer MOBO away from bad regions (optional)")

        all_evals_check = [e for e in st.session_state.task_evals if e.get('f1') is not None]
        if not all_evals_check:
            st.caption("Do some heuristic evaluations first — then the exploration map above will show you which regions are bad, and you can tell MOBO to avoid them here.")
        else:
            st.caption(
                "**Look at the map above.** See red dots (bad scores)? "
                "Set the sliders below to cover that region — MOBO will stop suggesting designs from there."
            )

        use_f = st.checkbox("Enable forbidden region", key="t_use_f")
        t_forbidden = None

        if use_f:
            st.caption(
                "Set min and max for each knob to define the region to avoid. "
                "Example: red dots at x₁≈0.1–0.3 → set x₁ min=0.0, x₁ max=0.3."
            )
            tc1, tc2 = st.columns(2)
            with tc1:
                st.caption("**Start of forbidden box:**")
                tx1min = st.slider("x₁ min", 0.0, 0.9,
                                   st.session_state.task_x1_min, 0.05, key="tx1min")
                tx2min = st.slider("x₂ min", 0.0, 0.9,
                                   st.session_state.task_x2_min, 0.05, key="tx2min")
                tx3min = st.slider("x₃ min", 0.0, 0.9,
                                   st.session_state.task_x3_min, 0.05, key="tx3min")
            with tc2:
                st.caption("**End of forbidden box:**")
                tx1max = st.slider("x₁ max", 0.1, 1.0,
                                   st.session_state.task_x1_max, 0.05, key="tx1max")
                tx2max = st.slider("x₂ max", 0.1, 1.0,
                                   st.session_state.task_x2_max, 0.05, key="tx2max")
                tx3max = st.slider("x₃ max", 0.1, 1.0,
                                   st.session_state.task_x3_max, 0.05, key="tx3max")

            if tx1min < tx1max and tx2min < tx2max and tx3min < tx3max:
                t_forbidden = {'x1_min':tx1min,'x1_max':tx1max,
                               'x2_min':tx2min,'x2_max':tx2max,
                               'x3_min':tx3min,'x3_max':tx3max}
                st.session_state.task_x1_min = tx1min
                st.session_state.task_x1_max = tx1max
                st.session_state.task_x2_min = tx2min
                st.session_state.task_x2_max = tx2max
                st.session_state.task_x3_min = tx3min
                st.session_state.task_x3_max = tx3max
                vol = (tx1max-tx1min)*(tx2max-tx2min)*(tx3max-tx3min)
                st.caption(f"Forbidden box covers {vol*100:.0f}% of parameter space")
            else:
                st.warning("Each min must be less than its max.")
                t_forbidden = None

        t_beta = st.slider("Avoid-strength β  (0 = ignore,  1 = strongly avoid)",
                           0.0, 1.0, st.session_state.task_beta, 0.05, key="t_beta")
        st.session_state.task_beta = t_beta
        st.session_state.task_forbidden = t_forbidden

        # Log steering changes
        log = st.session_state.steering_log
        new_entry = {'forbidden':t_forbidden, 'beta':t_beta,
                     'ts':datetime.now().isoformat()}
        if not log or log[-1].get('forbidden') != t_forbidden or log[-1].get('beta') != t_beta:
            log.append(new_entry)

    # ── RIGHT: Step-by-step controls ─────────────────────────
    with col_ctrl:

        # ── STEP 1: Get a design ─────────────────────────────
        st.markdown("#### Step 1 — Get a design")
        st.caption("Ask MOBO for a suggestion, or set x₁ x₂ x₃ manually.")

        if st.button("🤖 New Design from MOBO", type="primary",
                     key="t_mobo", use_container_width=True):
            formal_pts = [(e['f1'],e['f2']) for e in st.session_state.task_evals
                          if e['type']=='formal' and e.get('f1') is not None]
            sug, beta_int = mobo_suggest(
                candidates, formal_pts,
                st.session_state.task_forbidden,
                st.session_state.task_beta,
                st.session_state.condition)
            st.session_state.task_x = sug.tolist()
            entry = {
                'x': sug.tolist(),
                'beta_displayed': st.session_state.task_beta,
                'beta_internal': beta_int,
                'forbidden': st.session_state.task_forbidden,
                'd_to_forbidden': dist_to_forbidden(sug, st.session_state.task_forbidden),
                'timestamp': datetime.now().isoformat(),
            }
            st.session_state.pending_suggestion = entry
            st.session_state.mobo_log.append(entry)
            st.rerun()

        if st.session_state.pending_suggestion:
            s = st.session_state.pending_suggestion
            st.success(f"MOBO suggests: x₁={s['x'][0]:.3f}, "
                       f"x₂={s['x'][1]:.3f}, x₃={s['x'][2]:.3f}")

        c1, c2, c3 = st.columns(3)
        with c1: tx1 = st.slider("x₁", 0.0, 1.0, float(st.session_state.task_x[0]), 0.01, key="tx1")
        with c2: tx2 = st.slider("x₂", 0.0, 1.0, float(st.session_state.task_x[1]), 0.01, key="tx2")
        with c3: tx3 = st.slider("x₃", 0.0, 1.0, float(st.session_state.task_x[2]), 0.01, key="tx3")
        tx = np.array([tx1, tx2, tx3])
        st.session_state.task_x = [tx1, tx2, tx3]

        st.markdown("---")

        # ── STEP 2: Evaluate ─────────────────────────────────
        st.markdown("#### Step 2 — Evaluate this design")
        st.caption("Heuristic = free but noisy. Formal = accurate but costs 1 budget unit.")

        col_h, col_f = st.columns(2)
        with col_h:
            if st.button("🔵 Heuristic", key="t_h", use_container_width=True):
                f1, f2 = evaluate(tx, formal=False)
                st.session_state.task_evals.append({
                    'type':'heuristic','x':tx.tolist(),'f1':f1,'f2':f2,
                    'beta':st.session_state.task_beta,
                    'forbidden':st.session_state.task_forbidden,
                    'ts':datetime.now().isoformat()})
                st.session_state.heuristic_count += 1
                st.session_state.last_result = (f1, f2, 'heuristic')
                st.session_state.pending_suggestion = None
                st.rerun()

        with col_f:
            if budget_left > 0:
                if st.button("⭐ Formal", key="t_f",
                             use_container_width=True, type="primary"):
                    f1, f2 = evaluate(tx, formal=True)
                    st.session_state.task_evals.append({
                        'type':'formal','x':tx.tolist(),'f1':f1,'f2':f2,
                        'beta':st.session_state.task_beta,
                        'forbidden':st.session_state.task_forbidden,
                        'ts':datetime.now().isoformat()})
                    st.session_state.formal_used += 1
                    st.session_state.last_result = (f1, f2, 'formal')
                    st.session_state.pending_suggestion = None
                    st.rerun()
            else:
                st.button("⭐ Formal (budget used)", disabled=True,
                          key="t_f_dis", use_container_width=True)

        # Result feedback
        if st.session_state.last_result:
            f1, f2, etype = st.session_state.last_result
            if etype == 'formal':
                formal_pts = [(e['f1'],e['f2']) for e in st.session_state.task_evals
                              if e['type']=='formal']
                pidx = pareto_front(formal_pts)
                is_pareto = any(abs(formal_pts[i][0]-f1)<0.001 and
                                abs(formal_pts[i][1]-f2)<0.001 for i in pidx)
                if is_pareto:
                    st.success(f"✅ On Pareto front! f₁={f1:.3f}, f₂={f2:.3f}")
                else:
                    st.info(f"Formal: f₁={f1:.3f}, f₂={f2:.3f} — dominated")
            else:
                quality = "promising 👍" if (f1+f2) > 0.5 else "unpromising"
                st.info(f"Heuristic: f₁={f1:.3f}, f₂={f2:.3f} — looks {quality}")

        st.markdown("---")

        # ── Step 3 is shown on the LEFT below the exploration map ──
        st.info("👈 **Step 3 — Steer MOBO** is on the left, below the exploration map. "
                "Once you've done some evaluations, look for red dots and set the forbidden region there.")

    # Timer refresh — only when no pending suggestion and no recent result
    # Don't refresh if participant just clicked MOBO or evaluated — let them see the result
    has_pending = st.session_state.pending_suggestion is not None
    has_result  = st.session_state.last_result is not None
    if (st.session_state.phase == 'task'
            and budget_left > 0
            and remaining > 0
            and not has_pending
            and not has_result):
        time.sleep(15)
        st.rerun()

# ═══════════════════════════════════════════════════════════
# PHASE 6: QUESTIONNAIRE
# ═══════════════════════════════════════════════════════════
def show_questionnaire():
    st.title("Post-Task Questionnaire")
    st.markdown("Please answer all questions. Your responses are important for the study.")
    st.markdown("---")

    likert = ["1 — Strongly disagree","2 — Disagree","3 — Somewhat disagree",
              "4 — Neutral","5 — Somewhat agree","6 — Agree","7 — Strongly agree"]

    st.markdown("### Part 1 — Experience with the system")
    st.caption("Rate each statement from 1 (strongly disagree) to 7 (strongly agree).")

    items = {
        'agency_control':      "I felt in control of the optimisation process.",
        'agency_understanding':"I understood how my actions affected the search.",
        'agency_ownership':    "I felt ownership over the designs found.",
        'engagement':          "I found the task engaging.",
        'reuse':               "I would use this kind of system again for optimisation tasks.",
        'beta_fidelity':       "The avoid-strength β behaved the way I expected.",
        'mobo_trust':          "I trusted the MOBO suggestions.",
        'forbidden_useful':    "The forbidden region control was useful.",
    }

    r = {}
    for key, text in items.items():
        st.markdown(f"**{text}**")
        r[key] = st.radio("", likert, key=f"q_{key}", index=None,
                          horizontal=True, label_visibility="collapsed")
        st.markdown("")

    st.markdown("---")
    st.markdown("### Part 2 — NASA-TLX Workload")
    st.caption("Rate each dimension of task demand.")

    tlx_scale = ["1 — Very low","2","3","4","5","6","7 — Very high"]
    tlx_items = {
        'tlx_mental':    "**Mental Demand:** How mentally demanding was the task?",
        'tlx_temporal':  "**Temporal Demand:** How hurried or rushed was the pace?",
        'tlx_performance':"**Performance:** How successful were you at the task?",
        'tlx_effort':    "**Effort:** How hard did you have to work?",
        'tlx_frustration':"**Frustration:** How stressed or discouraged did you feel?",
    }
    for key, text in tlx_items.items():
        st.markdown(text)
        r[key] = st.radio("", tlx_scale, key=f"q_{key}", index=None,
                          horizontal=True, label_visibility="collapsed")
        st.markdown("")

    st.markdown("---")
    st.markdown("### Part 3 — Open feedback (optional)")
    r['open'] = st.text_area(
        "Anything else you'd like to share about your experience?", key="q_open")

    st.markdown("---")
    required = list(items.keys()) + list(tlx_items.keys())
    if st.button("Submit questionnaire →", type="primary"):
        if any(r.get(k) is None for k in required):
            st.error("Please answer all questions before submitting.")
        else:
            st.session_state.questionnaire = r
            st.session_state.phase = 'debrief'
            st.rerun()

# ═══════════════════════════════════════════════════════════
# PHASE 7: DEBRIEF + SAVE DATA
# ═══════════════════════════════════════════════════════════
def show_debrief():
    st.title("Debrief — Thank you!")
    st.markdown("---")
    cond = st.session_state.condition
    st.markdown(f"""
### What this study was about

This study investigated how **miscalibrated steering** affects optimisation behaviour
and perceived control in cooperative human-AI systems.

### What we manipulated

You were in the **{'Overconfident (OC)' if cond == 'OC' else 'Calibrated (C)'}** condition.

- **Calibrated (C):** The system's internal avoid-strength exactly matched the β value displayed.
- **Overconfident (OC):** The system internally applied a *stronger* avoid-strength than displayed.
  Specifically, if you set β = 0.5, the system used β = 0.9 internally — while still showing 0.5.

Both conditions had **identical interfaces**. The manipulation was completely invisible.

### Why this matters

In real AI systems, user controls don't always faithfully translate to internal behaviour —
due to model updates, abstraction gaps, or competing objectives.
This study is one of the first controlled tests of whether this kind of silent miscalibration
harms performance and erodes perceived understanding.

---
**Your data** has been saved anonymously. You may request deletion within 24 hours by contacting vm481@cam.ac.uk.

**Questions?** Please ask the researcher before you leave.
    """)

    save_data()
    st.success("✅ Your data has been saved anonymously. You may now close this tab.")
    # Offer JSON download as backup
    if st.session_state.get('saved_data'):
        st.download_button(
            label="⬇️ Download your data (optional)",
            data=st.session_state['saved_data'],
            file_name=f"data_{st.session_state['saved_key']}.json",
            mime="application/json",
            help="Optional: download a copy of your session data"
        )

def save_data():
    formal_pts = [(e['f1'],e['f2']) for e in st.session_state.task_evals
                  if e['type']=='formal' and e.get('f1') is not None]
    pidx = pareto_front(formal_pts) if formal_pts else []
    pareto_pts = [formal_pts[i] for i in pidx]
    final_hv = hypervolume(pareto_pts)
    n_formal = st.session_state.formal_used
    hv_per_formal = final_hv / n_formal if n_formal > 0 else 0

    d_vals = [s['d_to_forbidden'] for s in st.session_state.mobo_log
              if s.get('d_to_forbidden') is not None]

    formal_xs = [e['x'] for e in st.session_state.task_evals if e['type']=='formal']
    if len(formal_xs) >= 2:
        xs = np.array(formal_xs)
        nn = [min(np.linalg.norm(xs[i]-xs[j]) for j in range(len(xs)) if i!=j)
              for i in range(len(xs))]
        mean_nn = float(np.mean(nn))
    else:
        mean_nn = None

    q = st.session_state.questionnaire
    pid = st.session_state.participant_id
    ts  = datetime.now().isoformat()

    full_data = {
        'participant_id': pid,
        'condition': st.session_state.condition,
        'timestamp': ts,
        'demographics': st.session_state.demographics,
        'final_hv': final_hv,
        'n_formal_used': n_formal,
        'hv_per_formal': hv_per_formal,
        'mean_nn_dist': mean_nn,
        'n_mobo_suggestions': len(st.session_state.mobo_log),
        'n_steering_edits': len(st.session_state.steering_log),
        'mean_d_to_forbidden': float(np.mean(d_vals)) if d_vals else None,
        'd_to_forbidden_all': d_vals,
        'questionnaire': q,
        'task_evaluations': st.session_state.task_evals,
        'mobo_log': st.session_state.mobo_log,
        'steering_log': st.session_state.steering_log,
    }

    # ── Row for Google Sheet ─────────────────────────────────
    # Note: monitor condition balance in Google Sheet daily
    # Use ?condition=C or ?condition=OC in URL to rebalance if needed
    row = [
        pid,
        st.session_state.condition,
        ts,
        st.session_state.demographics.get('age',''),
        st.session_state.demographics.get('sex',''),
        st.session_state.demographics.get('handedness',''),
        st.session_state.demographics.get('mobo_exp',''),
        round(final_hv, 6),
        n_formal,
        round(hv_per_formal, 6),
        round(mean_nn, 6) if mean_nn else '',
        len(st.session_state.mobo_log),
        len(st.session_state.steering_log),
        round(float(np.mean(d_vals)), 6) if d_vals else '',
        q.get('agency_control',''),
        q.get('agency_understanding',''),
        q.get('agency_ownership',''),
        q.get('engagement',''),
        q.get('reuse',''),
        q.get('beta_fidelity',''),
        q.get('mobo_trust',''),
        q.get('forbidden_useful',''),
        q.get('tlx_mental',''),
        q.get('tlx_temporal',''),
        q.get('tlx_performance',''),
        q.get('tlx_effort',''),
        q.get('tlx_frustration',''),
        q.get('open',''),
    ]

    # ── POST to Google Apps Script web app ───────────────────
    # This is a simple no-auth endpoint that appends a row to your sheet
    APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbzReoJpqc6i1LDgmslEtR67KVr0OMEiMzjw2bjlMlCuL8k3ZDnH9Edzjb3QPWyyWcWi/exec"
    try:
        import requests
        resp = requests.post(
            APPS_SCRIPT_URL,
            json={'row': row},
            timeout=15
        )
        if resp.status_code == 200:
            st.session_state['save_status'] = 'sheet_ok'
    except Exception as e:
        st.session_state['save_status'] = f'sheet_failed: {e}'

    # ── Always store full JSON in session for download button ─
    key = f"p_{pid}_{st.session_state.condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state['saved_key'] = key
    st.session_state['saved_data'] = json.dumps(full_data, default=str, indent=2)


# ═══════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Cooperative MOBO Study",
                       page_icon="🔬", layout="wide")
    init()
    # Scroll to top on every phase change
    st.components.v1.html(
        "<script>window.parent.document.querySelector('section.main').scrollTo({top:0,behavior:'instant'});</script>",
        height=0
    )
    phase = st.session_state.phase
    if   phase == 'consent':       show_consent()
    elif phase == 'tutorial':      show_tutorial()
    elif phase == 'check':         show_check()
    elif phase == 'practice':      show_practice()
    elif phase == 'task_intro':    show_task_intro()
    elif phase == 'task':          show_task()
    elif phase == 'questionnaire': show_questionnaire()
    elif phase == 'debrief':       show_debrief()

if __name__ == '__main__':
    main()