import katex from "katex";
import "katex/dist/katex.min.css";
import Link from "next/link";

/* ─── KaTeX helpers ─── */

function tex(latex: string): string {
  return katex.renderToString(latex, { throwOnError: false });
}

function texBlock(latex: string): string {
  return katex.renderToString(latex, {
    throwOnError: false,
    displayMode: true,
  });
}

function Tex({ children, display }: { children: string; display?: boolean }) {
  return (
    <span
      dangerouslySetInnerHTML={{
        __html: display ? texBlock(children) : tex(children),
      }}
    />
  );
}

/* ─── FormulaBox: formula on the right, variable legend on the left ─── */

function FormulaBox({
  latex,
  legend,
  caption,
}: {
  latex: string;
  legend: { symbol: string; desc: string }[];
  caption?: string;
}) {
  return (
    <div className="my-5 rounded-xl border border-[#F0F0F0] bg-[#FAFAFA] overflow-hidden">
      <div className="flex flex-col sm:flex-row">
        {/* Legend */}
        <div className="p-4 sm:border-r border-b sm:border-b-0 border-[#E5E5E5] sm:min-w-[180px] sm:max-w-[220px] flex flex-col justify-center gap-2">
          {legend.map(({ symbol, desc }, i) => (
            <div
              key={i}
              className="flex items-baseline gap-2 text-xs leading-snug"
            >
              <span
                dangerouslySetInnerHTML={{ __html: tex(symbol) }}
                className="text-[#4B4B4B] shrink-0"
              />
              <span className="text-[#777777]">{desc}</span>
            </div>
          ))}
        </div>
        {/* Formula */}
        <div className="flex-1 p-4 sm:p-5 flex items-center justify-center overflow-x-auto">
          <span dangerouslySetInnerHTML={{ __html: texBlock(latex) }} />
        </div>
      </div>
      {caption && (
        <p className="text-xs text-[#AFAFAF] text-center pb-3 px-4 border-t border-[#F0F0F0] pt-2">
          {caption}
        </p>
      )}
    </div>
  );
}

/* ─── Section wrapper ─── */

function Section({
  badge,
  badgeColor,
  title,
  children,
}: {
  badge: string;
  badgeColor: string;
  title: string;
  children: React.ReactNode;
}) {
  const colorMap: Record<string, { bg: string; fg: string }> = {
    blue: { bg: "#DDF4FF", fg: "#1899D6" },
    green: { bg: "#D7FFB8", fg: "#46A302" },
    purple: { bg: "#F3E8FF", fg: "#9B59B6" },
    orange: { bg: "#FFF3D6", fg: "#E58600" },
    red: { bg: "#FFDFE0", fg: "#CC2424" },
    gray: { bg: "#F7F7F7", fg: "#777777" },
  };
  const c = colorMap[badgeColor] || colorMap.gray;
  return (
    <section className="bg-white rounded-[16px] border-2 border-[#E5E5E5] p-6 sm:p-8 mb-6">
      <div
        className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider mb-3"
        style={{ backgroundColor: c.bg, color: c.fg }}
      >
        {badge}
      </div>
      <h2 className="text-lg sm:text-xl font-bold text-[#3C3C3C] mb-4">
        {title}
      </h2>
      {children}
    </section>
  );
}

/* ─── Pipeline step chip ─── */

function PipelineChip({
  label,
  sub,
  color,
}: {
  label: string;
  sub: string;
  color: string;
}) {
  return (
    <div
      className="px-3 py-2 sm:px-4 sm:py-2.5 rounded-xl text-center border border-[#E5E5E5]"
      style={{ backgroundColor: color }}
    >
      <div className="font-bold text-xs sm:text-sm text-[#3C3C3C]">
        {label}
      </div>
      <div className="text-[10px] sm:text-xs text-[#777777]">{sub}</div>
    </div>
  );
}

function Arrow() {
  return (
    <span className="text-[#CDCDCD] text-sm sm:text-base select-none">→</span>
  );
}

/* ─── Prose helpers ─── */

function P({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[15px] text-[#4B4B4B] leading-relaxed mb-4">
      {children}
    </p>
  );
}

function H3({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-[15px] sm:text-base font-bold text-[#3C3C3C] mt-6 mb-2">
      {children}
    </h3>
  );
}

/* ═══════════════════════════════════════════════════════════════════ */
/*  PAGE                                                              */
/* ═══════════════════════════════════════════════════════════════════ */

export default function StackPage() {
  return (
    <div className="min-h-screen px-5 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Back link */}
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-[#AFAFAF] hover:text-[#1CB0F6] transition-colors mb-8 no-underline"
        >
          ← Back
        </Link>

        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl sm:text-3xl font-bold text-[#3C3C3C] mb-2">
            How BRACE Works
          </h1>
          <p className="text-[15px] text-[#777777] max-w-lg mx-auto">
            An unsupervised approach to real-time movement analysis — no labels,
            no training data, just math.
          </p>
        </div>

        {/* Pipeline overview */}
        <div className="flex flex-wrap items-center justify-center gap-1.5 sm:gap-2 my-6">
          <PipelineChip label="Camera" sub="30 fps" color="#F7F7F7" />
          <Arrow />
          <PipelineChip label="Pose" sub="YOLO11" color="#DDF4FF" />
          <Arrow />
          <PipelineChip label="Normalize" sub="Body frame" color="#D7FFB8" />
          <Arrow />
          <PipelineChip label="Segment" sub="Velocity" color="#FFF3D6" />
          <Arrow />
          <PipelineChip label="Cluster" sub="Spectral FFT" color="#F3E8FF" />
          <Arrow />
          <PipelineChip label="Analyze" sub="Quality" color="#FFDFE0" />
        </div>

        {/* ─── SECTION 1: Core Innovation ─── */}
        <Section badge="Core Innovation" badgeColor="blue" title="Unsupervised Movement Clustering">
          <P>
            When someone does twenty squats, the last few tend to be sloppier
            than the first. A coach spots this at a glance. We wanted a camera
            to do the same — without telling it what a &ldquo;squat&rdquo; is.
          </P>
          <P>
            The challenge: the same movement looks completely different in pixel
            space depending on camera angle, body proportions, and position. We
            need a representation where identical movements produce identical
            features, regardless of context.
          </P>

          <H3>Body-frame normalization</H3>
          <P>
            We transform every pose into body-centric coordinates: origin at the
            pelvis midpoint, x-axis along the hip vector, y-axis via
            Gram-Schmidt orthogonalization from the shoulder-pelvis direction.
            All distances are measured in hip-width units, so a 6&prime;5&Prime;
            person and a 5&prime;2&Prime; person doing the same squat produce
            nearly identical features.
          </P>

          <FormulaBox
            latex={String.raw`x^*_j = \frac{(\vec{p}_j - \mathbf{o}) \cdot \hat{x}}{w_h}, \quad y^*_j = \frac{(\vec{p}_j - \mathbf{o}) \cdot \hat{y}}{w_h}`}
            legend={[
              { symbol: String.raw`x^*_j,\; y^*_j`, desc: "normalized coordinates of joint j" },
              { symbol: String.raw`\vec{p}_j`, desc: "raw position of joint j" },
              { symbol: String.raw`\mathbf{o}`, desc: "pelvis midpoint (origin)" },
              { symbol: String.raw`\hat{x}`, desc: "unit vector along hip line" },
              { symbol: String.raw`\hat{y}`, desc: "Gram-Schmidt axis from shoulder–pelvis" },
              { symbol: String.raw`w_h`, desc: "hip width (scale factor)" },
            ]}
            caption="SRP (Scale-Rotation-Position) normalization — invariant to body size, orientation, and camera angle"
          />

          <P>
            We keep 14 key joints (shoulders, elbows, wrists, hips, knees,
            ankles, feet), giving a{" "}
            <Tex>{String.raw`14 \times 2 = 28`}</Tex>-dimensional feature
            vector per frame.
          </P>

          <H3>Finding movement boundaries</H3>
          <P>
            We compute the velocity of the feature vector over time. Pauses
            between repetitions show up as valleys in this velocity curve. An
            adaptive smoothing kernel (scaled to the minimum expected segment
            duration) plus peak detection on the inverted signal gives us clean
            boundary estimates — no exercise-specific tuning needed.
          </P>

          <H3>The key insight: phase-invariant spectral distance</H3>
          <P>
            Given two movement segments, how do you decide if they&rsquo;re
            &ldquo;the same&rdquo;?
          </P>
          <P>
            Comparing trajectories frame-by-frame fails as soon as someone
            changes speed or pauses mid-rep — the sequences go out of
            alignment. Dynamic Time Warping can handle speed variation, but
            it&rsquo;s noise-sensitive and computationally expensive.
          </P>
          <P>
            We take a different approach:{" "}
            <strong>
              compare the frequency content of movements, not their timing.
            </strong>
          </P>
          <P>
            For each segment, we resample to a fixed length and compute the FFT
            power spectrum. The spectrum captures the oscillation signature of
            the movement — which joints move, how far, at what frequency —
            while being completely invariant to phase and timing. Two squats at
            different speeds produce different time-domain curves but nearly
            identical spectra.
          </P>
          <P>
            We combine this with mean-pose distance (to catch postural
            differences the spectrum misses):
          </P>

          <FormulaBox
            latex={String.raw`D(a,b) = \frac{\|\boldsymbol{\mu}_a - \boldsymbol{\mu}_b\|_2 \;+\; \|\mathbf{S}_a - \mathbf{S}_b\|_2}{\sqrt{d}}`}
            legend={[
              { symbol: String.raw`D(a,b)`, desc: "distance between segments a and b" },
              { symbol: String.raw`\boldsymbol{\mu}`, desc: "mean pose vector of a segment" },
              { symbol: String.raw`\mathbf{S}`, desc: "FFT power spectrum (DC removed)" },
              { symbol: String.raw`d`, desc: "feature dimension (28 or 42)" },
            ]}
            caption="Spectral distance — phase-invariant comparison of two movement segments"
          />

          <P>
            The <Tex>{String.raw`\sqrt{d}`}</Tex> normalization lets us use the
            same distance threshold (2.0) for both 2D and 3D input without
            retuning.
          </P>

          <H3>Hierarchical clustering</H3>
          <P>
            These pairwise distances feed into agglomerative clustering with{" "}
            <strong>average linkage</strong>, which computes inter-cluster
            distance as the mean of all pairwise distances. This avoids the
            &ldquo;chaining&rdquo; problem of single linkage, where one close
            pair can merge two otherwise-different clusters.
          </P>
          <P>
            After cutting the dendrogram at threshold 2.0, we merge adjacent
            same-cluster segments and absorb tiny clusters (&lt;&nbsp;2 reps)
            into their nearest neighbor. The result: the system automatically
            discovers &ldquo;these 8 segments are all squats&rdquo; and
            &ldquo;those 4 are lunges&rdquo; — completely unsupervised.
          </P>
        </Section>

        {/* ─── SECTION 2: Biomechanical Analysis ─── */}
        <Section
          badge="Injury Prevention"
          badgeColor="red"
          title="Per-Frame Biomechanical Analysis"
        >
          <P>
            Clustering tells us <em>what</em> movement someone is doing and
            whether it&rsquo;s drifting. But it can&rsquo;t tell us{" "}
            <em>why</em> a rep looks different — whether the knee is caving
            inward, the trunk is leaning, or the hips are dropping. For that, we
            compute specific biomechanical angles on every frame, each chosen
            because it&rsquo;s a clinically validated predictor of injury risk.
          </P>

          <H3>FPPA — Frontal Plane Projection Angle</H3>
          <P>
            Knee valgus (inward collapse) is one of the strongest predictors of
            ACL tears, especially during landing and squatting (Hewett 2005). We
            measure the same angle a clinician would eyeball — how far the knee
            deviates from the hip-to-ankle line in the frontal plane.
          </P>

          <FormulaBox
            latex={String.raw`\theta_{\text{FPPA}} = \arctan\!\left(\frac{\|\mathbf{d}_\perp\|}{\tfrac{1}{2}\|\mathbf{h} - \mathbf{a}\|}\right)`}
            legend={[
              { symbol: String.raw`\theta_{\text{FPPA}}`, desc: "frontal plane projection angle" },
              { symbol: String.raw`\mathbf{h},\;\mathbf{a}`, desc: "hip and ankle positions" },
              { symbol: String.raw`\mathbf{d}_\perp`, desc: "perpendicular deviation of knee from hip–ankle line" },
            ]}
            caption="Negative = valgus (medial collapse), Positive = varus (lateral). Thresholds: squat 12°/20°, running 10°/18°."
          />

          <H3>Hip drop</H3>
          <P>
            When one hip drops below the other during single-leg stance, it
            signals gluteal weakness on the supporting side (Noehren 2013).
            Persistent hip drop is an early marker for iliotibial band syndrome
            and patellofemoral pain, so we track pelvic obliquity on every frame.
          </P>

          <FormulaBox
            latex={String.raw`\theta_{\text{drop}} = \arcsin\!\left(\frac{\Delta y_{\text{hips}}}{w_h}\right)`}
            legend={[
              { symbol: String.raw`\theta_{\text{drop}}`, desc: "hip drop angle from horizontal" },
              { symbol: String.raw`\Delta y_{\text{hips}}`, desc: "vertical difference between left and right hip" },
              { symbol: String.raw`w_h`, desc: "hip width (normalizer)" },
            ]}
            caption="Positive = left hip higher. Thresholds: running 6°/10°, generic 8°/12°."
          />

          <H3>Trunk lean</H3>
          <P>
            Lateral trunk lean shifts the center of mass away from the base of
            support, loading one side asymmetrically. In running it correlates
            with stress fracture risk; in squats it often compensates for a weak
            hip abductor. The thresholds adapt per exercise — a deadlift expects
            forward lean, a plank does not.
          </P>

          <FormulaBox
            latex={String.raw`\theta_{\text{trunk}} = \arccos\!\left(\frac{\mathbf{t} \cdot \hat{v}}{\|\mathbf{t}\|}\right)`}
            legend={[
              { symbol: String.raw`\theta_{\text{trunk}}`, desc: "trunk lean angle from vertical" },
              { symbol: String.raw`\mathbf{t}`, desc: "vector from hip center to shoulder center" },
              { symbol: String.raw`\hat{v}`, desc: "vertical unit vector" },
            ]}
            caption="Positive = leaning right. Plank: 8°/15° (strict), deadlift: 25°/40° (expected hinge)."
          />

          <H3>Bilateral Asymmetry Index</H3>
          <P>
            Side-to-side imbalances are both a cause and consequence of injury.
            A 15%+ asymmetry in joint angles during a bilateral exercise like
            squats suggests one side is compensating — often invisible to the
            person until pain develops.
          </P>

          <FormulaBox
            latex={String.raw`\text{BAI} = \frac{|L - R|}{\max(|L|,\;|R|)} \times 100\%`}
            legend={[
              { symbol: String.raw`L,\; R`, desc: "left and right joint angle values" },
              { symbol: String.raw`\text{BAI}`, desc: "bilateral asymmetry index" },
            ]}
            caption="Disabled for lunges (asymmetry expected). Generic threshold: 15%/25%."
          />

          <H3>Angular velocity</H3>
          <P>
            Rapid joint rotation — particularly at the knee — indicates
            ballistic or uncontrolled movement. Tracking peak angular velocity
            flags the exact moments where ligament stress is highest.
          </P>

          <FormulaBox
            latex={String.raw`\omega_j = |\theta_j^{(t)} - \theta_j^{(t-1)}| \cdot f_s`}
            legend={[
              { symbol: String.raw`\omega_j`, desc: "angular velocity of joint j (°/s)" },
              { symbol: String.raw`\theta_j^{(t)}`, desc: "joint angle at frame t" },
              { symbol: String.raw`f_s`, desc: "sampling rate (30 fps)" },
            ]}
            caption="Threshold: 500°/s for knee joints."
          />

          <H3>Center of mass sway</H3>
          <P>
            Whole-body balance is the most integrated stability measure
            available from a single camera. We estimate center of mass using
            Winter&rsquo;s (1990) anthropometric model — known segment masses
            and proximal-to-distal CoM positions — then track how far it drifts
            from the base of support.
          </P>

          <FormulaBox
            latex={String.raw`\mathbf{c} = \frac{\sum_{s} m_s \cdot \mathbf{r}_s}{\sum_{s} m_s}, \quad \text{sway} = \|\mathbf{c} - \mathbf{a}_{\text{mid}}\|`}
            legend={[
              { symbol: String.raw`\mathbf{c}`, desc: "estimated center of mass" },
              { symbol: String.raw`m_s`, desc: "mass fraction of segment s" },
              { symbol: String.raw`\mathbf{r}_s`, desc: "CoM position of segment s" },
              { symbol: String.raw`\mathbf{a}_{\text{mid}}`, desc: "midpoint of ankles (base of support)" },
            ]}
            caption="Trunk = 49.7% body mass at 43% up from hip to shoulder. Limb CoM at 43% from proximal joint."
          />

          <H3>Isolation Forest anomaly scoring</H3>
          <P>
            Individual metrics catch individual problems, but some injury-risk
            postures only emerge from <em>combinations</em> — moderate knee
            valgus plus moderate hip drop plus high trunk lean might be
            dangerous even though each alone is below threshold. An Isolation
            Forest, retrained every 60 frames, catches these multivariate
            outliers without hand-crafted rules.
          </P>

          <div className="my-5 rounded-xl border border-[#F0F0F0] bg-[#FAFAFA] p-4 sm:p-5">
            <p className="text-xs font-bold text-[#3C3C3C] mb-2 uppercase tracking-wider">
              7D anomaly feature vector
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-[#4B4B4B]">
              {[
                ["FPPA (L)", "knee valgus, left"],
                ["FPPA (R)", "knee valgus, right"],
                ["Hip drop", "pelvic obliquity"],
                ["Trunk lean", "lateral torso"],
                ["BAI", "bilateral asymmetry"],
                ["Curvature", "path bending in feature space"],
                ["Jerk", "acceleration in feature space"],
              ].map(([name, desc]) => (
                <div key={name} className="bg-white rounded-lg p-2 border border-[#F0F0F0]">
                  <div className="font-semibold text-[#3C3C3C]">{name}</div>
                  <div className="text-[10px] text-[#777777]">{desc}</div>
                </div>
              ))}
            </div>
            <p className="text-xs text-[#AFAFAF] mt-3">
              Retrained every 60 frames. Score 0–1 where 1 = most anomalous.
            </p>
          </div>

          <H3>Form scoring</H3>
          <P>
            Once clustering discovers what movements belong together, the
            cluster representative (average trajectory) becomes a form
            template. Each frame is scored by per-joint deviation from this
            template — a 0–100 score where 100 is perfect reproduction. Lower
            scores pinpoint exactly which joints are drifting from the
            established pattern.
          </P>
        </Section>

        {/* ─── SECTION 3: Fatigue Detection ─── */}
        <Section
          badge="Statistical Analysis"
          badgeColor="purple"
          title="Detecting Fatigue"
        >
          <P>
            Once movements are clustered, we track how each repetition deviates
            from baseline — the first few reps, when the person is presumably
            fresh. We borrow two techniques from{" "}
            <strong>manufacturing quality control</strong>, where the goal is to
            detect when a production process goes &ldquo;out of control.&rdquo;
          </P>

          <H3>EWMA control chart</H3>
          <P>
            A single bad rep isn&rsquo;t fatigue, but a trend of worsening reps
            is. The Exponentially Weighted Moving Average tracks per-joint
            deviation with memory — recent observations matter more, but history
            isn&rsquo;t discarded — making it ideal for catching gradual
            joint-level degradation.
          </P>

          <FormulaBox
            latex={String.raw`Z_k = \lambda \, x_k + (1-\lambda) \, Z_{k-1}`}
            legend={[
              { symbol: String.raw`Z_k`, desc: "EWMA statistic at rep k" },
              { symbol: String.raw`x_k`, desc: "deviation of rep k from baseline" },
              { symbol: String.raw`\lambda`, desc: "smoothing constant (0.2)" },
              { symbol: String.raw`Z_{k-1}`, desc: "previous EWMA value" },
            ]}
          />

          <P>
            A joint is flagged as degrading when its EWMA exceeds the upper
            control limit:
          </P>

          <FormulaBox
            latex={String.raw`\text{UCL}_k = \mu_0 + L \cdot \sigma_0 \sqrt{\frac{\lambda}{2-\lambda}\Big(1-(1-\lambda)^{2(k+1)}\Big)}`}
            legend={[
              { symbol: String.raw`\text{UCL}_k`, desc: "upper control limit at rep k" },
              { symbol: String.raw`\mu_0`, desc: "baseline mean deviation" },
              { symbol: String.raw`\sigma_0`, desc: "baseline standard deviation" },
              { symbol: String.raw`L`, desc: "sensitivity factor (2.7)" },
            ]}
            caption="The correction factor inside the square root adjusts for sample size, preventing false alarms on early reps."
          />

          <H3>CUSUM change-point detection</H3>
          <P>
            EWMA is good at tracking individual joints, but it can smooth over a
            slow, sustained shift. CUSUM is more sensitive to these — it
            accumulates small positive deviations that individually look harmless
            but collectively reveal a trend, pinpointing exactly which rep the
            decline began.
          </P>

          <FormulaBox
            latex={String.raw`C_k = \max\!\left(0,\; C_{k-1} + d_k - \mu_d - \frac{\delta}{2}\right)`}
            legend={[
              { symbol: String.raw`C_k`, desc: "cumulative sum at rep k" },
              { symbol: String.raw`d_k`, desc: "distance of rep k from baseline" },
              { symbol: String.raw`\mu_d`, desc: "mean baseline distance" },
              { symbol: String.raw`\delta`, desc: "shift to detect (0.5σ)" },
            ]}
            caption={`Alarm when C_k exceeds h = 4σ. The δ/2 allowance prevents drift from normal variation from accumulating.`}
          />

          <H3>Movement smoothness</H3>
          <P>
            Fatigue doesn&rsquo;t just change <em>where</em> joints go — it
            changes <em>how smoothly</em> they get there. Two metrics from
            rehabilitation science quantify this:
          </P>

          <P>
            <strong>SPARC</strong> (Balasubramanian 2012) measures the arc
            length of the normalized power spectrum. Smooth movements have
            compact spectra; fatigued movements spread energy across more
            frequencies:
          </P>

          <FormulaBox
            latex={String.raw`\text{SPARC} = -\sum_{i}\sqrt{(\Delta\hat{f}_i)^2 + (\Delta\hat{M}_i)^2}`}
            legend={[
              { symbol: String.raw`\hat{f}_i`, desc: "normalized frequency bin" },
              { symbol: String.raw`\hat{M}_i`, desc: "normalized spectral magnitude" },
              { symbol: String.raw`\Delta`, desc: "bin-to-bin difference" },
            ]}
            caption="More negative = jerkier. Invariant to movement duration."
          />

          <P>
            <strong>LDLJ</strong> (Balasubramanian 2015) integrates squared jerk
            over the movement. The normalization by duration and peak velocity
            makes it comparable across movements at different scales:
          </P>

          <FormulaBox
            latex={String.raw`\text{LDLJ} = -\ln\!\left(\frac{T^3}{v_{\text{peak}}^2}\int_0^T\!\left\|\frac{d^3\mathbf{p}}{dt^3}\right\|^2 dt\right)`}
            legend={[
              { symbol: String.raw`T`, desc: "movement duration" },
              { symbol: String.raw`v_{\text{peak}}`, desc: "peak speed during the movement" },
              { symbol: String.raw`\mathbf{p}`, desc: "position trajectory" },
              { symbol: String.raw`\frac{d^3\mathbf{p}}{dt^3}`, desc: "jerk (third derivative of position)" },
            ]}
            caption="More negative = jerkier movement. The log-dimensionless form enables cross-exercise comparison."
          />

          <H3>Kinematic chain sequencing</H3>
          <P>
            In healthy movement, joints fire in a proximal-to-distal order —
            hips initiate, knees follow, ankles finish. When fatigue disrupts
            this sequence, movement becomes less efficient and injury risk
            increases. We find peak angular velocity timing for each joint and
            compare the actual firing order to the ideal using Kendall&rsquo;s τ
            rank correlation.
          </P>

          <FormulaBox
            latex={String.raw`\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\binom{n}{2}}, \quad \text{score} = \frac{\tau + 1}{2}`}
            legend={[
              { symbol: String.raw`\tau`, desc: "Kendall rank correlation" },
              { symbol: String.raw`n`, desc: "number of tracked joints" },
              { symbol: String.raw`\text{score}`, desc: "normalized 0–1 (1 = ideal order)" },
            ]}
            caption="Ideal order: hip → knee → ankle. Reversed firing = fatigue or compensation."
          />

          <H3>Sample entropy</H3>
          <P>
            Fresh movement is rhythmic and predictable. Fatigued movement
            becomes erratic — more complex in the information-theoretic sense.
            Sample entropy quantifies this: low values mean regular, repeating
            patterns; high values mean unpredictable, irregular motion.
          </P>

          <FormulaBox
            latex={String.raw`\text{SampEn}(m, r) = -\ln\frac{A}{B}`}
            legend={[
              { symbol: String.raw`m`, desc: "embedding dimension (2)" },
              { symbol: String.raw`r`, desc: "tolerance (0.2 × std)" },
              { symbol: String.raw`A`, desc: "count of matching (m+1)-templates" },
              { symbol: String.raw`B`, desc: "count of matching m-templates" },
            ]}
            caption="Higher = more complex/erratic. Increasing across reps signals fatigue."
          />

          <H3>Spectral median frequency</H3>
          <P>
            In electromyography, a downward shift in the median frequency of the
            power spectrum is a classic fatigue biomarker — it reflects motor
            unit recruitment changes. We apply the same principle to joint
            angular velocity: as muscles fatigue, the dominant movement frequency
            decreases.
          </P>

          <FormulaBox
            latex={String.raw`\text{MNF}: \quad \int_0^{\text{MNF}} P(f)\,df = \frac{1}{2}\int_0^{f_{\max}} P(f)\,df`}
            legend={[
              { symbol: String.raw`\text{MNF}`, desc: "spectral median frequency" },
              { symbol: String.raw`P(f)`, desc: "power spectral density of angular velocity" },
              { symbol: String.raw`f_{\max}`, desc: "Nyquist frequency (15 Hz at 30 fps)" },
            ]}
            caption="MNF decreasing across reps = fatigue (shift to lower-frequency movement)."
          />

          <H3>Composite fatigue score</H3>
          <P>
            We combine seven degradation signals into a single [0,&thinsp;1]
            score. The weights reflect each signal&rsquo;s reliability and
            clinical importance:
          </P>

          <div className="my-5 rounded-xl border border-[#F0F0F0] bg-[#FAFAFA] p-4 sm:p-5">
            <div className="space-y-2">
              {[
                { name: "ROM decay", weight: "25%", desc: "range of motion loss across reps — the most direct fatigue signal", color: "#EA2B2B" },
                { name: "EWMA", weight: "20%", desc: "per-joint degradation — which joints are failing first", color: "#FF9600" },
                { name: "CUSUM", weight: "20%", desc: "overall trend change-point — when degradation started", color: "#FF9600" },
                { name: "Correlation", weight: "15%", desc: "rep-to-rep consistency loss — form becoming unpredictable", color: "#1CB0F6" },
                { name: "SPARC", weight: "8%", desc: "smoothness decline — movement becoming jerky", color: "#CE82FF" },
                { name: "LDLJ", weight: "7%", desc: "jerk increase — acceleration becoming erratic", color: "#CE82FF" },
                { name: "Spread", weight: "5%", desc: "trajectory scatter — loop tightness degrading", color: "#AFAFAF" },
              ].map(({ name, weight, desc, color }) => (
                <div key={name} className="flex items-start gap-3">
                  <div
                    className="text-xs font-bold rounded-md px-2 py-1 shrink-0 min-w-[56px] text-center text-white"
                    style={{ backgroundColor: color }}
                  >
                    {weight}
                  </div>
                  <div>
                    <span className="text-sm font-bold text-[#3C3C3C]">
                      {name}
                    </span>
                    <span className="text-xs text-[#777777] ml-1.5">
                      {desc}
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-[#AFAFAF] mt-3 border-t border-[#F0F0F0] pt-2">
              ROM loss is weighted highest (25%) because it&rsquo;s the most
              clinically interpretable. Control chart signals (EWMA + CUSUM)
              together account for 40% — they detect degradation before it
              becomes visually obvious.
            </p>
          </div>
        </Section>

        {/* ─── SECTION 4: Results ─── */}
        <Section badge="Results" badgeColor="green" title="Does It Work?">
          <P>
            We benchmarked 18 clustering algorithms on 14 real sports videos —
            basketball, boxing, yoga, tennis, running, and more. Top results:
          </P>

          <div className="my-5 space-y-2">
            {[
              { name: "HDBSCAN", score: "100%", w: "100%" },
              { name: "OPTICS", score: "100%", w: "100%" },
              { name: "Agglomerative (tuned)", score: "93%", w: "93%" },
              { name: "Spectral Clustering", score: "86%", w: "86%" },
              { name: "Agglomerative (default)", score: "64%", w: "64%" },
            ].map(({ name, score, w }) => (
              <div key={name} className="flex items-center gap-3">
                <div className="text-sm text-[#3C3C3C] font-medium w-[180px] shrink-0">
                  {name}
                </div>
                <div className="flex-1 bg-[#F0F0F0] rounded-full h-5 overflow-hidden">
                  <div
                    className="h-full rounded-full flex items-center justify-end pr-2"
                    style={{
                      width: w,
                      backgroundColor:
                        score === "100%" ? "#58CC02" : score === "93%" ? "#1CB0F6" : "#E5E5E5",
                    }}
                  >
                    <span className="text-xs font-bold text-white">
                      {score}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <P>
            We chose agglomerative clustering for production — it&rsquo;s fast
            (2.5 ms/video), interpretable, and reaches 93% with threshold
            tuning. HDBSCAN is the accuracy winner at 100%, and we may switch to
            it in a future iteration.
          </P>

          <div className="grid grid-cols-3 gap-3 my-5">
            {[
              { value: "1.6×", label: "Normal vs. pathological gait separation" },
              { value: "473", label: "Automated tests" },
              { value: "30 fps", label: "Real-time processing" },
            ].map(({ value, label }) => (
              <div
                key={value}
                className="bg-[#FAFAFA] rounded-xl p-4 text-center border border-[#F0F0F0]"
              >
                <div className="text-xl sm:text-2xl font-bold text-[#3C3C3C]">
                  {value}
                </div>
                <div className="text-[10px] sm:text-xs text-[#777777] mt-1">
                  {label}
                </div>
              </div>
            ))}
          </div>

          <P>
            All tests use synthetic skeleton data with known ground truth
            (sinusoidal motion, controlled joint angles, seeded noise). Anomaly
            detection validated on synthetic gait: normal walking scores 0.89,
            pathological gait scores 1.40.
          </P>
        </Section>

        {/* ─── SECTION 5: Architecture ─── */}
        <Section
          badge="For the curious"
          badgeColor="gray"
          title="Architecture"
        >
          <h3 className="text-base font-bold text-[#3C3C3C] mb-3">
            The Full Stack
          </h3>

          {/* Architecture diagram */}
          <div className="my-5 rounded-xl border border-[#F0F0F0] bg-[#FAFAFA] overflow-hidden font-mono text-xs sm:text-sm leading-relaxed">
            <div className="p-4 sm:p-5 space-y-3">
              <div className="text-center">
                <span className="inline-block bg-[#DDF4FF] px-3 py-1.5 rounded-lg border border-[#B8E8FF] font-bold text-[#1899D6]">
                  Browser
                </span>
                <div className="text-[10px] text-[#AFAFAF] mt-0.5">
                  Next.js 15 · React 19 · Three.js · Canvas @ 60 fps
                </div>
              </div>

              <div className="text-center text-[#CDCDCD]">
                ↕ WebSocket · binary JPEG frames
              </div>

              <div className="text-center">
                <span className="inline-block bg-[#D7FFB8] px-3 py-1.5 rounded-lg border border-[#B8E8A0] font-bold text-[#46A302]">
                  FastAPI Backend
                </span>
                <div className="text-[10px] text-[#AFAFAF] mt-0.5">
                  YOLO11-pose → SRP Normalize → Segment → Cluster → Movement
                  Quality
                </div>
                <div className="text-[10px] text-[#AFAFAF]">
                  One-Euro Filter · Isolation Forest · CLIP-ReID
                </div>
              </div>

              <div className="text-center text-[#CDCDCD]">↓</div>

              <div className="flex flex-wrap justify-center gap-2">
                {[
                  { label: "MongoDB", sub: "Sessions & profiles", color: "#FFF3D6", border: "#FFE4A0", text: "#E58600" },
                  { label: "Gemini 2.5", sub: "Activity labels", color: "#F3E8FF", border: "#E0C8FF", text: "#9B59B6" },
                ].map(({ label, sub, color, border, text }) => (
                  <div
                    key={label}
                    className="px-3 py-1.5 rounded-lg border text-center"
                    style={{ backgroundColor: color, borderColor: border }}
                  >
                    <div className="font-bold" style={{ color: text }}>
                      {label}
                    </div>
                    <div className="text-[10px] text-[#AFAFAF]">{sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <P>
            JPEG frames travel from the browser to the backend over WebSocket at
            30 fps. The backend runs pose estimation, normalizes, segments,
            clusters, and returns a full analysis response — skeleton
            coordinates, cluster assignments, quality metrics, fatigue
            indicators — all within a single frame&rsquo;s time budget.
          </P>
          <P>
            The frontend renders the skeleton overlay and 3D UMAP embedding at
            60 fps using refs (not React state) to avoid re-render overhead. A
            ring buffer delays the displayed video by the pipeline round-trip
            time so the skeleton overlay lines up with the frame it was computed
            from.
          </P>

          <div className="text-xs text-[#AFAFAF] leading-relaxed mt-6 pt-4 border-t border-[#F0F0F0]">
            Built with Next.js 15 · React 19 · TypeScript · Three.js · FastAPI ·
            YOLO11-pose · WebSocket · Docker · MongoDB · Tailscale · Gemini 2.5
            Pro · UMAP · scikit-learn
          </div>
        </Section>
      </div>
    </div>
  );
}
