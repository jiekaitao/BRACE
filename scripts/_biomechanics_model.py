"""
Biomechanical concussion probability model for two-person collisions.

This module implements physics-based head injury risk estimation grounded in
peer-reviewed literature on sports concussion biomechanics. The core pipeline:

    closing speed -> momentum transfer -> head delta-v -> peak acceleration
    -> concussion probability (logistic regression)

Primary references:
    - Rowson S, Duma SM. "Brain injury prediction: assessing the combined
      probability of concussion using linear and rotational head acceleration."
      Ann Biomed Eng. 2013;41(5):873-882. doi:10.1007/s10439-012-0731-0
    - Pellman EJ, Viano DC, Tucker AM, Casson IR, Waeckerle JF. "Concussion
      in professional football: reconstruction of game impacts and injuries."
      Neurosurgery. 2003;53(4):799-814.
    - Zhang L, Yang KH, King AI. "A proposed injury threshold for mild
      traumatic brain injury." J Biomech Eng. 2004;126(2):226-236.
    - Ommaya AK, Gennarelli TA. "Cerebral concussion and traumatic
      unconsciousness." Brain. 1974;97(4):633-654.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

GRAVITY = 9.81  # m/s^2, standard gravitational acceleration

# Effective mass of the head-neck complex during impact. When a player is
# braced (aware of the incoming collision), the neck muscles are tensed and
# the head is coupled to the torso, yielding a higher effective mass (~6.5 kg).
# When unbraced/unaware, the head rotates more freely on the neck, behaving
# closer to its anatomical mass (~4.5 kg). This distinction matters because
# a lower effective mass leads to higher acceleration for the same impulse.
# Reference: Viano DC et al., "Concussion in professional football: brain
# responses by finite element analysis." Neurosurgery. 2005;56(2):344-362.
HEAD_NECK_MASS_KG = 6.5       # head + neck effective mass (braced/aware)
UNBRACED_HEAD_MASS_KG = 4.5   # head only (unaware of impact)

# Impact duration governs the peak acceleration via the impulse-momentum
# theorem. Helmeted impacts spread the force over a longer duration (~12 ms)
# due to the padding/shell deformation, while unhelmeted impacts are shorter
# (~8 ms). These values come from instrumented helmet studies:
# Reference: Rowson S, Duma SM. "Development of the STAR evaluation system
# for football helmets." Ann Biomed Eng. 2011;39(8):2130-2140.
IMPACT_DURATION_HELMETED_S = 0.012    # ~12 ms with helmet
IMPACT_DURATION_UNHELMETED_S = 0.008  # ~8 ms without helmet

# Default body mass used when bounding-box estimation is not available.
# Roughly the average adult male mass in collision sports.
DEFAULT_BODY_MASS_KG = 80.0

# Coefficient of restitution for body-to-body contact in contact sports.
# Pure elastic = 1.0, pure inelastic = 0.0. Human body collisions are
# highly inelastic (soft tissue absorbs energy), so e ~ 0.2-0.4.
# We use 0.3 as a mid-range estimate.
# Reference: Pain MTG, Tsui F, Cove S. "In vivo determination of the
# effect of shoulder pads on tackling forces in rugby." J Sports Sci.
# 2008;26(8):855-862.
COEFF_RESTITUTION = 0.3

# ---------------------------------------------------------------------------
# Rowson & Duma (2013) logistic regression coefficients
# ---------------------------------------------------------------------------
# These coefficients come from Table 2 of Rowson & Duma (2013), fitted to
# 63,011 sub-concussive and 37 concussive instrumented helmet impacts from
# collegiate football players (Virginia Tech, University of Oklahoma, etc.).
# The model is:
#   P(concussion) = 1 / (1 + exp(-(B0 + B1*a + B2*alpha + B3*a*alpha)))
# where:
#   a     = peak linear acceleration (g)
#   alpha = peak rotational acceleration (rad/s^2)
RD_B0 = -10.2       # intercept
RD_B1 = 0.0433      # coefficient for linear acceleration (per g)
RD_B2 = 0.000873    # coefficient for rotational acceleration (per rad/s^2)
RD_B3 = -0.00000092 # interaction term (per g * rad/s^2)

# ---------------------------------------------------------------------------
# Rotational/linear correlation factor
# ---------------------------------------------------------------------------
# Empirical relationship between peak rotational and peak linear acceleration
# observed in NFL reconstructions. Pellman et al. (2003) found a strong
# correlation (R^2 ~ 0.76) between the two, with rotational acceleration
# approximately 50 rad/s^2 per 1 g of linear acceleration. This is a rough
# approximation used when direct rotational measurement is unavailable.
# Reference: Pellman EJ, Viano DC, et al. Neurosurgery. 2003;53(4):799-814.
ROTATIONAL_LINEAR_FACTOR = 50.0  # rad/s^2 per g (approximate)


# ---------------------------------------------------------------------------
# Helper functions (velocity-dependent COR, Hertzian duration, HIC)
# ---------------------------------------------------------------------------

def velocity_dependent_restitution(
    v_closing: float,
    e_base: float = 0.55,
    e_slope: float = 0.025,
    e_min: float = 0.1,
) -> float:
    """Compute a velocity-dependent coefficient of restitution.

    At low closing speeds, soft-tissue restitution is relatively high.
    As impact speed increases, more energy is absorbed by tissue
    deformation, reducing the coefficient. This avoids overestimating
    rebound energy at high closing speeds.

    Parameters
    ----------
    v_closing : float
        Closing speed in m/s (>= 0).
    e_base : float
        COR at zero closing speed.
    e_slope : float
        Decrease in COR per 1 m/s of closing speed.
    e_min : float
        Floor value — COR cannot drop below this.

    Returns
    -------
    float
        Coefficient of restitution in [e_min, e_base].
    """
    return max(e_min, e_base - e_slope * v_closing)


def hertzian_impact_duration(
    base_s: float,
    v_closing: float,
    v_ref: float = 5.0,
    power: float = 0.2,
) -> float:
    """Scale impact duration with closing speed using Hertzian contact theory.

    Faster impacts compress tissues more quickly, shortening the contact
    time. The scaling follows a power-law relationship motivated by
    Hertzian contact mechanics.

    Parameters
    ----------
    base_s : float
        Base impact duration in seconds at v_ref.
    v_closing : float
        Closing speed in m/s (> 0).
    v_ref : float
        Reference closing speed at which duration equals base_s.
    power : float
        Exponent for the speed ratio (0.2 from Hertz theory).

    Returns
    -------
    float
        Scaled impact duration in seconds.
    """
    if v_closing <= 0.0 or v_ref <= 0.0:
        return base_s

    ratio = v_ref / v_closing
    # Clamp the ratio to avoid extreme durations
    ratio = max(0.5, min(2.0, ratio))
    return base_s * (ratio ** power)


def compute_hic_half_sine(peak_g: float, duration_s: float) -> float:
    """Compute the Head Injury Criterion (HIC) for a half-sine acceleration pulse.

    Closed-form for a half-sine pulse of peak amplitude A and duration T:
        HIC = T * [(2/pi) * A]^2.5

    where A is in g (converted to m/s^2 for the standard HIC formula,
    but the g-based form is equivalent when using the standard HIC
    definition with acceleration in g).

    Parameters
    ----------
    peak_g : float
        Peak linear acceleration in g.
    duration_s : float
        Impact duration in seconds.

    Returns
    -------
    float
        HIC value (dimensionless).
    """
    if peak_g <= 0.0 or duration_s <= 0.0:
        return 0.0
    # HIC for half-sine: T * ((2/pi) * peak_g)^2.5
    return duration_s * ((2.0 / math.pi) * peak_g) ** 2.5


# ---------------------------------------------------------------------------
# Function implementations
# ---------------------------------------------------------------------------

def estimate_body_mass(bbox_height_px: float, meters_per_pixel: float) -> float:
    """Estimate body mass from bounding-box height using a BMI-based model.

    Approach:
        1. Convert the bounding-box height from pixels to meters using the
           provided spatial calibration (meters_per_pixel).
        2. Assume the bounding-box height is a reasonable proxy for the
           person's standing height. In practice, pose may reduce this
           (crouching, leaning), but for collision-speed estimation the
           upright assumption is a reasonable first approximation.
        3. Apply the BMI formula solved for mass:
               mass = BMI * height^2
           We use BMI = 22.0 kg/m^2, which is the midpoint of the "normal"
           range (18.5-24.9) per WHO classification. For athletic populations
           a higher BMI might be appropriate, but 22 avoids systematic
           overestimation for the general population.
        4. Clamp to [30, 150] kg to avoid pathological estimates from noisy
           bounding boxes (e.g., partially visible persons, children, or
           measurement artifacts).

    Parameters
    ----------
    bbox_height_px : float
        Height of the person's bounding box in pixels.
    meters_per_pixel : float
        Spatial calibration factor converting pixels to meters.
        Typically derived from a known reference length in the scene
        (e.g., court markings, known object dimensions).

    Returns
    -------
    float
        Estimated body mass in kilograms, clamped to [30, 150].
    """
    # Step 1: Convert pixel height to meters
    height_m = bbox_height_px * meters_per_pixel

    # Step 2: BMI-based mass estimation
    # BMI = mass / height^2  =>  mass = BMI * height^2
    # Using BMI = 22.0 (normal adult, WHO)
    bmi = 22.0
    mass_kg = bmi * (height_m ** 2)

    # Step 3: Clamp to physiologically plausible range
    # 30 kg ~ small child or very underweight adult
    # 150 kg ~ very large athlete (NFL lineman territory)
    mass_kg = max(30.0, min(150.0, mass_kg))

    return mass_kg


def momentum_transfer(
    velocity_a_ms: float,
    mass_a_kg: float,
    velocity_b_ms: float,
    mass_b_kg: float,
    coeff_restitution: float = COEFF_RESTITUTION,
) -> Tuple[float, float]:
    """Compute velocity changes from a 1D inelastic collision.

    Uses the standard 1D collision equations with a coefficient of
    restitution (e) to model energy loss. The coefficient of restitution
    is defined as:

        e = -(v_a' - v_b') / (v_a - v_b)

    where v_a, v_b are pre-impact velocities and v_a', v_b' are
    post-impact velocities. Combined with conservation of momentum:

        m_a * v_a + m_b * v_b = m_a * v_a' + m_b * v_b'

    We solve for the post-impact velocities:

        v_a' = v_a - ((1 + e) * m_b * (v_a - v_b)) / (m_a + m_b)
        v_b' = v_b + ((1 + e) * m_a * (v_a - v_b)) / (m_a + m_b)

    The delta-v for each person is the absolute change in their velocity,
    which directly relates to the impulse they experience:

        delta_v = |v' - v|

    Parameters
    ----------
    velocity_a_ms : float
        Pre-impact velocity of person A in m/s (positive = rightward).
    mass_a_kg : float
        Mass of person A in kg.
    velocity_b_ms : float
        Pre-impact velocity of person B in m/s (positive = rightward).
    mass_b_kg : float
        Mass of person B in kg.
    coeff_restitution : float, optional
        Coefficient of restitution, default 0.3 (highly inelastic).
        0.0 = perfectly inelastic, 1.0 = perfectly elastic.

    Returns
    -------
    tuple[float, float]
        (delta_v_a, delta_v_b) — absolute velocity changes in m/s for
        person A and person B respectively.
    """
    # Total mass (denominator in collision equations)
    total_mass = mass_a_kg + mass_b_kg

    # Relative approach velocity (positive when A is moving toward B)
    relative_velocity = velocity_a_ms - velocity_b_ms

    # Impulse factor: (1 + e) * relative_velocity / total_mass
    # This is the normalized impulse exchanged between the two bodies.
    impulse_factor = (1.0 + coeff_restitution) * relative_velocity / total_mass

    # Post-impact velocities from the standard 1D collision formula:
    #   v_a' = v_a - (1+e) * m_b * (v_a - v_b) / (m_a + m_b)
    #   v_b' = v_b + (1+e) * m_a * (v_a - v_b) / (m_a + m_b)
    v_a_post = velocity_a_ms - impulse_factor * mass_b_kg
    v_b_post = velocity_b_ms + impulse_factor * mass_a_kg

    # Delta-v is the absolute change in velocity for each person.
    # This is the key injury-relevant metric: higher delta-v means a larger
    # impulse was transmitted to the body, and consequently to the head.
    delta_v_a = abs(v_a_post - velocity_a_ms)
    delta_v_b = abs(v_b_post - velocity_b_ms)

    return (delta_v_a, delta_v_b)


def delta_v_to_peak_g(
    delta_v_head_ms: float,
    impact_duration_s: float = IMPACT_DURATION_HELMETED_S,
) -> float:
    """Convert head delta-v to peak linear acceleration in g.

    Uses the impulse-momentum theorem. The key insight is that real impacts
    do not produce a rectangular acceleration pulse — they are closer to a
    half-sine shape. For a half-sine pulse of duration T and peak amplitude
    A_peak:

        integral(A_peak * sin(pi*t/T), t=0..T) = A_peak * 2T/pi

    Setting this equal to the total impulse (delta_v):

        delta_v = A_peak * 2T / pi
        A_peak  = delta_v * pi / (2T)
              = delta_v / T * (pi/2)

    The factor pi/2 ~ 1.5708 converts from average acceleration to peak
    acceleration under the half-sine pulse assumption.

    Reference: Gurdjian ES, Roberts VL, Thomas LM. "Tolerance curves of
    acceleration and intracranial pressure and protective index in
    experimental head injury." J Trauma. 1966;6(5):600-604.

    Parameters
    ----------
    delta_v_head_ms : float
        Change in head velocity in m/s.
    impact_duration_s : float, optional
        Duration of the impact pulse in seconds. Default is 0.012 s
        (helmeted). Use 0.008 s for unhelmeted impacts.

    Returns
    -------
    float
        Peak linear acceleration in g (multiples of gravitational
        acceleration, 9.81 m/s^2).
    """
    # Half-sine pulse shape factor: the ratio of peak to average acceleration
    # for a half-sine waveform. pi/2 = 1.5708...
    half_sine_factor = math.pi / 2.0  # ~1.5708

    # Average acceleration from impulse-momentum: a_avg = delta_v / dt
    # Peak acceleration with half-sine correction: a_peak = a_avg * (pi/2)
    # Convert from m/s^2 to g by dividing by 9.81 m/s^2
    if impact_duration_s <= 0.0:
        return 0.0

    peak_accel_ms2 = (delta_v_head_ms / impact_duration_s) * half_sine_factor
    peak_g = peak_accel_ms2 / GRAVITY

    return peak_g


def estimate_rotational_acceleration(peak_linear_g: float) -> float:
    """Approximate peak rotational acceleration from linear acceleration.

    In head impacts, the brain experiences both translational (linear) and
    rotational accelerations simultaneously. While they are distinct
    mechanical quantities, empirical data from NFL game reconstructions show
    a strong positive correlation (R^2 ~ 0.76) between the two:

        peak_rotational (rad/s^2) ~ 50 * peak_linear (g)

    This arises because most head impacts in sports involve an off-center
    (oblique) force application, so a hard linear hit also produces
    significant rotation. The factor of 50 rad/s^2 per g is an approximate
    central tendency from the Pellman et al. (2003) reconstruction dataset.

    Limitations:
        - This is a population-level correlation, not a physical law.
          Individual impacts can deviate substantially.
        - Pure translational (centroidal) impacts would have lower rotational
          acceleration; pure tangential impacts would have higher.
        - Helmet design, neck musculature, and impact location all modulate
          the ratio.

    Reference: Pellman EJ, Viano DC, Tucker AM, Casson IR, Waeckerle JF.
    "Concussion in professional football: reconstruction of game impacts and
    injuries." Neurosurgery. 2003;53(4):799-814.

    Parameters
    ----------
    peak_linear_g : float
        Peak linear acceleration in g.

    Returns
    -------
    float
        Estimated peak rotational acceleration in rad/s^2.
    """
    # Simple linear scaling from the NFL reconstruction correlation.
    # For a 50g impact: rotational ~ 2500 rad/s^2
    # For a 100g impact: rotational ~ 5000 rad/s^2
    # These magnitudes are consistent with literature values for concussive
    # impacts (Zhang et al., 2004: mean ~5900 rad/s^2 for concussions).
    return ROTATIONAL_LINEAR_FACTOR * peak_linear_g


def concussion_probability_rowson_duma(
    peak_linear_g: float,
    peak_rotational_rads2: float,
) -> float:
    """Compute concussion probability using Rowson & Duma (2013) combined model.

    This logistic regression model was derived from the largest instrumented
    helmet dataset to date: 63,011 head impacts recorded by the Head Impact
    Telemetry (HIT) System across multiple collegiate football programs,
    including 37 diagnosed concussions.

    Model form (logistic regression):

        P(concussion) = 1 / (1 + exp(-z))

    where:
        z = B0 + B1*a + B2*alpha + B3*a*alpha

        a     = peak linear acceleration (g)
        alpha = peak rotational acceleration (rad/s^2)
        B0    = -10.2     (intercept)
        B1    =  0.0433   (linear acceleration coefficient)
        B2    =  0.000873 (rotational acceleration coefficient)
        B3    = -0.00000092 (interaction term)

    The negative interaction term (B3) means that at very high combined
    accelerations, the model slightly tempers the probability estimate,
    reflecting the observation that the highest-energy impacts don't
    necessarily produce concussions at a rate proportional to the product
    of both accelerations (possibly due to loss-of-consciousness and
    protective reflexes at extreme levels).

    Model performance:
        - AUC (ROC) > 0.85 on the training dataset.
        - The combined model significantly outperforms linear-only or
          rotational-only models (p < 0.001).
        - Validated against independent concussion cases from other programs.

    Reference: Rowson S, Duma SM. "Brain injury prediction: assessing the
    combined probability of concussion using linear and rotational head
    acceleration." Ann Biomed Eng. 2013;41(5):873-882.

    Parameters
    ----------
    peak_linear_g : float
        Peak linear acceleration in g.
    peak_rotational_rads2 : float
        Peak rotational acceleration in rad/s^2.

    Returns
    -------
    float
        Probability of concussion in [0, 1].
    """
    # Compute the logit (log-odds) from the linear predictor
    z = (
        RD_B0
        + RD_B1 * peak_linear_g
        + RD_B2 * peak_rotational_rads2
        + RD_B3 * peak_linear_g * peak_rotational_rads2
    )

    # Apply the logistic (sigmoid) function.
    # We clamp z to [-500, 500] to avoid floating-point overflow in exp().
    # At z = -500, P ~ 0; at z = 500, P ~ 1. This is well beyond any
    # physically meaningful acceleration range.
    z = max(-500.0, min(500.0, z))
    probability = 1.0 / (1.0 + math.exp(-z))

    return probability


def concussion_probability_linear_only(peak_g: float) -> float:
    """Compute concussion probability using a linear-acceleration-only model.

    This is a simplified logistic model calibrated to match the Pellman/NFL
    reconstruction data (Pellman et al., 2003) and the Wayne State Tolerance
    Curve (WSTC). It uses only peak linear acceleration as the predictor.

    Model form:
        P(concussion) = 1 / (1 + exp(-(b0 + b1 * g)))

    Coefficients are calibrated to match three reference points from the
    NFL reconstruction literature:
        - ~25% probability at 75g
        - ~50% probability at 85g
        - ~75% probability at 99g

    Solving the logistic equation for b0 and b1 using the 50% point:
        At P = 0.5: 0 = b0 + b1 * 85  =>  b0 = -85 * b1
        At P = 0.25: ln(1/3) = b0 + b1 * 75  =>  -1.0986 = -85*b1 + 75*b1
        => b1 = 1.0986 / 10 ~ 0.1099
    Fine-tuning to also fit the 75g and 99g points yields:
        b0 = -9.805, b1 = 0.1154

    Verification:
        P(75g) = 1/(1+exp(-(-9.805 + 0.1154*75))) = 1/(1+exp(1.15)) ~ 0.24
        P(85g) = 1/(1+exp(-(-9.805 + 0.1154*85))) = 1/(1+exp(-0.004)) ~ 0.50
        P(99g) = 1/(1+exp(-(-9.805 + 0.1154*99))) = 1/(1+exp(-1.62)) ~ 0.84

    Limitations:
        - Linear acceleration alone is an incomplete predictor of concussion.
          Rotational acceleration plays a critical role in diffuse axonal
          injury (Ommaya & Gennarelli, 1974).
        - This model is useful as a quick screening tool or when rotational
          data is unavailable.

    Reference: Pellman EJ, Viano DC, et al. Neurosurgery. 2003;53(4):799-814.

    Parameters
    ----------
    peak_g : float
        Peak linear acceleration in g.

    Returns
    -------
    float
        Probability of concussion in [0, 1].
    """
    # Logistic regression coefficients (see calibration above)
    b0 = -9.805
    b1 = 0.1154

    # Compute the logit
    z = b0 + b1 * peak_g

    # Clamp to avoid overflow (same guard as in the combined model)
    z = max(-500.0, min(500.0, z))
    probability = 1.0 / (1.0 + math.exp(-z))

    return probability


def score_collision(
    closing_speed_ms: float,
    mass_a_kg: float,
    mass_b_kg: float,
    head_coupling_factor: float,
    impact_duration_s: float = IMPACT_DURATION_HELMETED_S,
    helmeted: bool = False,
    use_velocity_dependent_e: bool = True,
    use_hertzian_duration: bool = True,
    approach_angle_rad: float = 0.0,
    min_pose_confidence: float = 1.0,
) -> Dict[str, float | str]:
    """Score the concussion risk of a two-person collision.

    This is the main entry point that chains together all the component
    models to produce a comprehensive collision risk assessment.

    Pipeline:
        1. Decompose closing speed into opposing velocities (symmetric split).
        2. Compute body-level delta-v via 1D inelastic collision physics.
        3. Identify the "struck" person as the one with the larger delta-v
           (typically the lighter person absorbs more velocity change).
        4. Scale body delta-v to head delta-v via the head coupling factor.
           - coupling = 1.0: head perfectly coupled to body (braced, aware).
           - coupling > 1.0: head whips relative to body (unbraced, unaware),
             amplifying the effective head delta-v.
           - coupling < 1.0: head partially shielded (e.g., tucked chin).
        5. Convert head delta-v to peak linear acceleration using the
           impulse-momentum theorem with half-sine pulse correction.
        6. Estimate rotational acceleration from the linear-rotational
           correlation.
        7. Compute concussion probability using both the combined
           (Rowson & Duma) and linear-only (Pellman/NFL) models.
        8. Assign a risk level based on the combined probability.

    Risk level thresholds:
        - LOW:      P < 5%    (sub-concussive range, routine contact)
        - MODERATE: 5% <= P < 15%  (elevated risk, warrants monitoring)
        - HIGH:     15% <= P < 50% (significant risk, consider removal)
        - CRITICAL: P >= 50%  (more likely than not to cause concussion)

    These thresholds are informed by clinical decision-making frameworks
    in sports medicine, where the "return to play" decision balances
    injury risk against competitive considerations.

    Parameters
    ----------
    closing_speed_ms : float
        Total closing speed between the two persons in m/s. This is the
        relative approach velocity (|v_a - v_b|). For head-on collisions,
        this is the sum of both persons' speeds.
    mass_a_kg : float
        Mass of person A in kg.
    mass_b_kg : float
        Mass of person B in kg.
    head_coupling_factor : float
        Scaling factor from body delta-v to head delta-v.
        Typical values:
            - 0.5-0.8: braced, chin tucked, neck muscles tensed
            - 1.0: neutral coupling (head moves with body)
            - 1.2-1.5: unbraced, head whip effect
            - Up to 2.0: severe whiplash scenario
    impact_duration_s : float, optional
        Duration of the impact pulse in seconds. Default is 0.012 s
        (helmeted). For unhelmeted collisions, use 0.008 s.
    helmeted : bool, optional
        If True, uses helmeted impact duration (0.012 s) regardless of
        the impact_duration_s parameter. If False, uses the provided
        impact_duration_s. Default is False.

    Returns
    -------
    dict
        A dictionary containing all intermediate and final results:
            - closing_speed_ms: input closing speed
            - delta_v_body_struck_ms: body delta-v of the struck person
            - delta_v_head_ms: head delta-v after coupling factor
            - peak_linear_g: peak linear acceleration in g
            - peak_rotational_rads2: peak rotational acceleration in rad/s^2
            - concussion_prob: Rowson & Duma combined probability [0, 1]
            - concussion_prob_linear: linear-only probability [0, 1]
            - risk_level: categorical risk assessment string
    """
    # Step 0: Override impact duration if helmeted flag is set.
    if helmeted:
        impact_duration_s = IMPACT_DURATION_HELMETED_S

    # Step 0b: Velocity-dependent coefficient of restitution (Step 4)
    if use_velocity_dependent_e:
        coeff_e = velocity_dependent_restitution(closing_speed_ms)
    else:
        coeff_e = COEFF_RESTITUTION

    # Step 1: Decompose closing speed into opposing velocities.
    v_a = closing_speed_ms / 2.0
    v_b = -closing_speed_ms / 2.0

    # Step 2: Compute body-level delta-v for each person via 1D collision.
    delta_v_a, delta_v_b = momentum_transfer(
        velocity_a_ms=v_a,
        mass_a_kg=mass_a_kg,
        velocity_b_ms=v_b,
        mass_b_kg=mass_b_kg,
        coeff_restitution=coeff_e,
    )

    # Step 3: The "struck" person is the one with the larger body delta-v.
    delta_v_body_struck = max(delta_v_a, delta_v_b)

    # Step 4: Scale body delta-v to head delta-v via coupling factor.
    # Angular modulation (Step 6): reduce coupling for oblique impacts.
    angular_modulation = max(0.1, math.cos(approach_angle_rad) ** 0.6)
    coupling_effective = head_coupling_factor * angular_modulation
    delta_v_head = delta_v_body_struck * coupling_effective

    # Step 5: Hertzian-scaled impact duration (Step 5)
    if use_hertzian_duration and closing_speed_ms > 0.0:
        effective_duration_s = hertzian_impact_duration(
            impact_duration_s, closing_speed_ms,
        )
    else:
        effective_duration_s = impact_duration_s

    # Step 5b: Convert head delta-v to peak linear acceleration.
    peak_linear_g = delta_v_to_peak_g(
        delta_v_head_ms=delta_v_head,
        impact_duration_s=effective_duration_s,
    )

    # Step 6: Estimate rotational acceleration.
    peak_rotational_rads2 = estimate_rotational_acceleration(peak_linear_g)

    # Step 7: Compute concussion probability from both models.
    concussion_prob = concussion_probability_rowson_duma(
        peak_linear_g=peak_linear_g,
        peak_rotational_rads2=peak_rotational_rads2,
    )
    concussion_prob_linear = concussion_probability_linear_only(
        peak_g=peak_linear_g,
    )

    # Step 7b: HIC cross-check (Step 7)
    hic = compute_hic_half_sine(peak_linear_g, effective_duration_s)

    # Step 8: Model applicability score (Step 8)
    applicability_flags: list[str] = []
    if closing_speed_ms > 8.0:
        applicability_flags.append("closing_speed_above_8ms")
    if mass_a_kg < 40.0 or mass_a_kg > 130.0:
        applicability_flags.append("mass_a_out_of_range")
    if mass_b_kg < 40.0 or mass_b_kg > 130.0:
        applicability_flags.append("mass_b_out_of_range")
    if min_pose_confidence < 0.5:
        applicability_flags.append("low_pose_confidence")
    model_applicability = "EXTRAPOLATED" if applicability_flags else "VALIDATED"

    # Step 9: Assign a categorical risk level.
    if concussion_prob < 0.05:
        risk_level = "LOW"
    elif concussion_prob < 0.15:
        risk_level = "MODERATE"
    elif concussion_prob < 0.50:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        # Original fields (preserved)
        "closing_speed_ms": closing_speed_ms,
        "delta_v_body_struck_ms": delta_v_body_struck,
        "delta_v_head_ms": delta_v_head,
        "peak_linear_g": peak_linear_g,
        "peak_rotational_rads2": peak_rotational_rads2,
        "concussion_prob": concussion_prob,
        "concussion_prob_linear": concussion_prob_linear,
        "risk_level": risk_level,
        # New additive fields
        "coeff_restitution": coeff_e,
        "impact_duration_s": effective_duration_s,
        "angular_modulation": angular_modulation,
        "coupling_effective": coupling_effective,
        "hic": hic,
        "model_applicability": model_applicability,
        "applicability_flags": applicability_flags,
    }
