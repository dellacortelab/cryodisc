# Derivations of the core relations

A self-contained derivation of the load-bearing relations, written to be checked step by
step against an independent derivation. Notation follows the appendix (whitened Fourier
convention). Each section states its **assumptions**, the **derivation**, a **checkpoint**
you can verify independently, and — where relevant — a **flag** marking a piece that is
empirical or a modeling choice rather than a derivation.

Sections are ordered so each quantity is derived before it is used: the model and its noise
and signal scales first (§1–§3), then the per-particle discriminability and its currency
(§4–§5), the pool and marginal rungs (§6–§8), the particle count with its fraction MLE, error
formula, calibration gate and pool estimators (§9–§9c), and the calibration and empirical closes
(§10–§11).

Convention throughout: `contrast` is the signal amplitude (renamed from `a`; cryoJAX renders
in contrast units, so `contrast = 1` is unit physical contrast). Only `contrast²/σ²` is
observable.

---

## 1. Model and whitened inner product

A single particle image, in the Fourier domain after whitening:

$$ Y(\vec k) = \text{contrast}\cdot \mu_s(\vec k) + n(\vec k), \qquad \mu_s = \hat C\,\hat P(R)\,V_s, $$

where `s ∈ {A, B}` is the true conformational state, `μ_s` is the CTF-projected template of
state `s` at pose `(R, t)`, and `n` is zero-mean Gaussian noise. Whitening means we measure
in units where the per-mode noise power is folded into the inner product:

$$ \langle a, b\rangle \equiv \sum_{\vec k} \frac{a(\vec k)\,\overline{b(\vec k)}}{\sigma^2(k)}, \qquad \lVert a\rVert^2 = \sum_{\vec k}\frac{|a(\vec k)|^2}{\sigma^2(k)}. $$

With this inner product the noise is effectively unit-variance per mode, so
`⟨n, u⟩` has variance `‖u‖²` for any fixed `u`. That single fact drives every moment below.

Two ingredients are used here but derived later: `σ²(k)`, the noise power spectrum, is taken
as given — its functional form is derived in §2 — and `contrast`, the signal amplitude, factors
out of the signal scale in §3. The discrimination ladder of §4–§9 needs only that they exist.

---

## 2. The noise model (shot noise and envelopes)

**Claim.**

$$ \sigma^2(k) = \frac{\text{shot}}{\mathrm{DQE}(k)\,E_{\text{rad}}(\text{dose})^2\,E_{\text{conf}}(\text{rmsf})^2}, \qquad \text{shot} = \frac{\text{box}^2}{4\,\text{dose}\,\text{pixel}^2}. $$

**Shot term — real space.** In contrast units the image intensity is `I = |\psi|^2 = 1 + 2C`
with `C` the contrast (weak-scattering linearization). A pixel of side `pixel` (Å) at exposure
`dose` (e⁻/Å²) collects `N_e = \text{dose}\cdot\text{pixel}^2` electrons. Poisson shot noise
gives relative intensity variance `1/N_e`, so with `I\approx 1`,

$$ \operatorname{Var}(I_{\text{pix}}) \approx \frac{1}{N_e} = \frac{1}{\text{dose}\,\text{pixel}^2}. $$

Because `C = (I-1)/2`, the **contrast** variance carries the factor of 4:

$$ \operatorname{Var}(C_{\text{pix}}) = \tfrac14\operatorname{Var}(I_{\text{pix}}) = \frac{1}{4\,\text{dose}\,\text{pixel}^2}. $$

**Shot term — per Fourier mode.** For a white real-space field of per-pixel variance `v` over
`N_{\text{pix}} = \text{box}^2` pixels, the *unnormalized* DFT
`n(\vec k) = \sum_{\vec x} n(\vec x)e^{-i\vec k\cdot\vec x}` has, by Parseval,
`\mathbb E|n(\vec k)|^2 = \sum_{\vec x} v = N_{\text{pix}}\,v = \text{box}^2\,v`. Hence

$$ \text{shot} = \text{box}^2\cdot\operatorname{Var}(C_{\text{pix}}) = \frac{\text{box}^2}{4\,\text{dose}\,\text{pixel}^2}. $$

So the `4` is the contrast-to-intensity Jacobian, `box²` is the unnormalized-DFT/Parseval
factor, and `dose·pixel²` is electrons per pixel. ∎

**Envelopes (denominator).** Three multiplicative factors degrade effective signal-to-noise
and so appear as `1/(\cdot)` on the noise:
- `DQE(k) < 1` — the detector records fewer effective electrons, `N_e \to \mathrm{DQE}\,N_e`,
  raising noise by `1/\mathrm{DQE}`.
- `E_{\text{rad}}(\text{dose})` — radiation-damage envelope, damps the **signal** amplitude.
- `E_{\text{conf}}(\text{rmsf}) = e^{-Bk^2/4}`, `B = 8\pi^2\,\text{rmsf}^2` — a **specimen**
  B-factor damping the signal amplitude.

**The "seam."** Because only `contrast\cdot\mu/\sigma` enters the SNR (§3), damping the signal
by an envelope `E` is *equivalent* to inflating the noise amplitude by `1/E` (power by
`1/E^2`). That is why signal envelopes can be "parked" in `σ²`. Two consequences to keep
straight:
- `E_{\text{conf}}` is a **specimen** property, not a collection property. The Figure-5
  drivers set it to the ordered-protein floor, a
  signal envelope `e^{-B_{\text{conf}}k²/4}` that by the seam above parks in `σ²`; the
  collection noise proper is `σ²(k) = shot/[DQE·E_rad²]`.
- **Note (reconciled 2026-09-21).** An earlier draft flagged a possible additive inelastic-ice term
  `σ²(k) \propto [1 + A^2 e^{-(k/k_{\text{ice}})^2}]/D` as appendix-vs-code drift. There is none: the
  code carries no such term (no `k_ice`/inelastic term in `src`), and the feasibility Supplementary
  Note now explicitly disclaims it, stating that the predictor does not model the forward-peaked
  inelastic background. Code, this derivation, and the appendix agree that `σ²` is the multiplicative
  shot-plus-envelopes form above.

---

## 3. Signal power, SNR, and why SNR is an output

**Claim.** The matched-filter SNR is linear in contrast and factors into contrast times a
pure collection/geometry factor:

$$ \mathrm{SNR}^2 = P = \text{contrast}^2\cdot\underbrace{\mathbb E_{\text{pose}}\lVert M\rVert^2}_{G^2}, \qquad \mathrm{SNR} = \text{contrast}\cdot G, \qquad G^2 = \mathbb E_{\text{pose}}\sum_{\vec k}\frac{|M(\vec k)|^2}{\sigma^2(k)}. $$

**Derivation.** The whitened matched-filter output power for the template `μ = contrast\cdot M`
is `P = \lVert\text{contrast}\,M\rVert^2 = \text{contrast}^2\lVert M\rVert^2`, averaged over the
pose distribution. `G^2` depends only on the structure spectrum `|M(k)|^2` and the collection
noise `σ²(k)` — no contrast. Hence `SNR = contrast·G`. ∎

**Why SNR is an *output*, never a target.** Fix the collection `(dose, optics)` → fixes
`σ²(k)` → fixes `G`. Fix `contrast = 1` on the signal side. Then `SNR = G` *falls out*; it is
not dialed. Different molecules on one microscope sit at different `SNR` because `G` differs.
This is why the operating point is the collection, and `SNR`/`η` is a per-pair measurement.

**Contrast is a gauge, not a physical claim.** Only `contrast²/σ²` is observable, so `contrast`
and `σ²` are one degree of freedom. Setting `contrast = 1` puts that freedom entirely on the
noise axis (where `dose` lives and is *reported* by the instrument), which is why we read
`dose` from the report rather than fitting it — fitting both is non-identifiable, and fitting
an effective `contrast` launders specimen heterogeneity into noise (usable only as a
cross-check). Real em-fits return contrast values well away from 1 precisely because real
data is not in cryoJAX contrast units.

**Relation to `d²`.** The detection signal is `M` (the template); the discrimination signal is
`Δμ = contrast\cdot\Delta M` (the difference map, §4–§5). Both carry `contrast²`, but they
answer different questions — "is a particle here?" vs "which state?" — so `SNR` and `d'²` are
distinct even though both scale as `contrast²`.

---

## 4. Per-particle known-pose discriminability

**Claim.** With the pose known, the log-likelihood ratio between states is *linear* in the
image and Gaussian under each class:

$$ \Lambda\mid A \sim \mathcal N(+D/2,\,D), \quad \Lambda\mid B \sim \mathcal N(-D/2,\,D), \quad D \equiv \lVert\Delta\mu\rVert^2, \quad \Delta\mu = \mu_A - \mu_B. $$

Its discriminability — the currency built from these moments in §5 — is the squared
difference-map norm `D`.

**Assumptions.** Gaussian noise; pose known; the two states enter only through their mean
images `μ_A, μ_B`.

**Derivation.** The Gaussian log-likelihood is `log p(Y|s) = -\tfrac12\lVert Y - \text{contrast}\,\mu_s\rVert^2 + \text{const}`.
Set `contrast = 1` for now (it reappears in §3). The LLR is

$$ \Lambda = \log\frac{p(Y|A)}{p(Y|B)} = -\tfrac12\lVert Y-\mu_A\rVert^2 + \tfrac12\lVert Y-\mu_B\rVert^2. $$

Expand both squares; the `‖Y‖²` terms cancel:

$$ \Lambda = \langle Y, \mu_A-\mu_B\rangle - \tfrac12\bigl(\lVert\mu_A\rVert^2 - \lVert\mu_B\rVert^2\bigr) = \langle Y - \mu_0,\ \Delta\mu\rangle, \qquad \mu_0 \equiv \tfrac12(\mu_A+\mu_B), $$

using `‖μ_A‖² − ‖μ_B‖² = ⟨μ_A−μ_B, μ_A+μ_B⟩ = ⟨Δμ, 2μ_0⟩`. So `Λ` is a single whitened
projection of the mean-subtracted image onto the difference map.

Now take moments. Under `A`, `Y = μ_A + n`, so `Y − μ_0 = \tfrac12\Delta\mu + n` and

$$ \Lambda = \tfrac12\lVert\Delta\mu\rVert^2 + \langle n, \Delta\mu\rangle \ \Rightarrow\ \mathbb E[\Lambda|A] = +\tfrac12 D, \quad \operatorname{Var}(\Lambda|A) = \lVert\Delta\mu\rVert^2 = D. $$

By symmetry `E[Λ|B] = −D/2`, `Var(Λ|B) = D`. ∎

**Checkpoint.** `D = ‖Δμ‖² = Σ_k |Δμ(k)|²/σ²(k)`. Since `Δμ = contrast·(M_A − M_B)`
carries one factor of `contrast`, `D ∝ contrast²`. This is the exact statement behind
"`d² ∝ contrast²`."

---

## 5. The currency `d'² ≡ gap²/within` is the state MLE's Fisher information

**Claim.** The variance-normalized deflection

$$ d'^2 \;\equiv\; \frac{\text{gap}^2}{\text{within}}, \qquad \text{gap} = \mathbb E[\Lambda|A] - \mathbb E[\Lambda|B], \qquad \text{within} = \text{within-class variance of }\Lambda, $$

is not an arbitrary summary of the first two moments. It **is the Fisher information about the
state carried by `Λ`** — equivalently the inverse CRLB variance of the state MLE read from `Λ`
— so `gap²/within` is *derived* from the information the estimator carries, and the classical
detectability index and the affine invariance both fall out of that.

**Derivation (deflection = Fisher information of the statistic).** Index the family by a scalar
state coordinate `θ` with `A` at `θ = +½`, `B` at `θ = −½` (spacing `Δθ = 1`; `A, B` are the
two ends of the conformational coordinate). Treat the sufficient statistic `Λ` as the data we
estimate `θ` from, and match a Gaussian to its first two moments, `Λ|θ ∼ N(m(θ),\,v(θ))`. For a
Gaussian family with `θ`-independent variance the Fisher information is the standard `m'(θ)²/v`
(the score is `∂_θ\log p = (Λ-m)m'/v`, whose variance is `m'^2/v`). With the known-pose moments
of §4, `m(θ) = Dθ` is **linear** and `v = D` is **constant**, so no small-motion approximation
is needed and

$$ I_\Lambda(\theta) = \frac{m'(\theta)^2}{v} = \frac{(\text{gap}/\Delta\theta)^2}{\text{within}} = \frac{\text{gap}^2}{\text{within}} = d'^2 \qquad(\Delta\theta = 1). $$

By the CRLB the state MLE from `Λ` then has asymptotic variance `within/\text{gap}^2 = 1/d'^2`:
`d'^2` *is* the estimator's information. Maximizing the deflection
`(∂_\theta\mathbb E[T])^2/\operatorname{Var}(T)` over all statistics `T` returns the *full*
Fisher information (Cauchy–Schwarz; the maximizer is the score), and at known pose `Λ` is
affinely the score. Writing the known-pose Fisher information about `z` as
`I_{zz} \equiv \lVert\partial_z\mu\rVert^2 = \lVert\Delta\mu\rVert^2` (the Gaussian-model Fisher
information `I_{θθ'} = ⟨∂_θμ, ∂_θ'μ⟩`, reused in §8), all three coincide:

$$ d'^2 = I_\Lambda = I_{zz} = D = \lVert\Delta\mu\rVert^2. $$

∎

**The per-particle discriminability is the difference-map norm.** Evaluating the currency on
§4's moments directly, `gap = E[Λ|A] − E[Λ|B] = D` and `within = Var(Λ|A) = D`, so

$$ d'^2 = \frac{\text{gap}^2}{\text{within}} = \frac{D^2}{D} = D = \lVert\Delta\mu\rVert^2. $$

The per-particle known-pose discriminability *is* the squared difference-map norm.

**Operational meaning (why it is the right currency).** Because `d'` indexes the Fisher
information, it also indexes performance: for equal-variance Gaussian scores the entire ROC and
the Bayes error depend on the pair only through `d' = \text{gap}/\sqrt{\text{within}}`, with
equal-prior per-particle error `P_err = Φ(−d/2)` (appendix Note 1, Bhattacharyya exponent
`d²/8`). Two problems with the same `d'` are equally hard.

**Affine invariance is a corollary.** Fisher information about `θ` is unchanged by an affine
reparametrization of the *statistic* `Λ ↦ αΛ + β` (`gap ↦ α\,\text{gap}`, `within ↦ α²\,\text{within}`).
That is exactly why `gap²/within` — and not the calibration-dependent mean gap — is reported on
the uncalibrated marginal LLR (§6 works the invariance out explicitly).

**Flag (exact at known pose; a lower bound under marginalization).** The identity
`deflection = Fisher information` is *exact* only when the score is linear in the statistic —
the linear-mean, constant-variance known-pose model. For the marginal (unknown-pose) `Λ` the
mean is nonlinear in `θ` and the variance state-dependent, so the reported `gap²/within` is the
*deflection coefficient* — the information of the moment-matched Gaussian — which under-reads
relative to the full marginal information. This is the same degradation as the `d'^2
≤ d'^2_{\text{CRLB}} ≤ I_{zz}` ordering flagged in §8, seen from the estimator side.

---

## 6. The three identities of `d²`, and where the third fails

**Claim.** For the *exact, calibrated* LLR, three quantities coincide:
(i) the mean gap, (ii) the Jeffreys divergence, (iii) the variance-normalized deflection
`gap²/within`; and in the small-motion limit all reduce to the Fisher information. For the
*marginal* (unknown-pose) LLR the third identity breaks, which is why the affine-invariant
`gap²/within` is the reliable estimator.

**(i) = (ii): mean gap = Jeffreys.** If `Λ` is the exact marginal-law ratio
`Λ = \log(P_A/P_B)`, then `E[Λ|A] = \int P_A \log(P_A/P_B) = D_{\mathrm{KL}}(P_A\Vert P_B)`,
and `−E[Λ|B] = D_{\mathrm{KL}}(P_B\Vert P_A)`. So

$$ \text{gap} = D_{\mathrm{KL}}(P_A\Vert P_B) + D_{\mathrm{KL}}(P_B\Vert P_A) = \text{Jeffreys divergence}. $$

**(iii): variance-normalized deflection.** Under the equal-covariance Gaussian of §4
(`Var(Λ|A) = Var(Λ|B) = D` and `gap = D`), we have `gap²/within = D²/D = D = gap`. So the
mean gap and the variance-normalized deflection agree — *when* the equal-covariance Gaussian
holds.

**Small-motion limit → Fisher.** Let the states differ by `δ` in the conformational
coordinate. Second-order Taylor of the KL divergence gives
`D_{\mathrm{KL}}(P_z\Vert P_{z+\delta}) = \tfrac12\,\delta^\top \mathbf I(z)\,\delta + O(\delta^3)`,
so `d² = δ^\top \mathbf I\,δ` — the Fisher–Rao metric — which for a scalar coordinate reduces
to `I_{zz}\,\delta^2`, the known-pose ceiling of §8.

**Where (iii) fails, and the resolution.** For the *marginal* ratio (pose integrated out),
`Λ` is nonlinear in `Y`, so the equal-covariance Gaussian no longer holds: integrating a
broad pose posterior inflates the variance, `Var(Λ) > gap`, and any variance-normalized index
reads low. Worse, in practice the marginal `Λ` is computed with an imperfect calibration, so
its *scale* is arbitrary: under an affine reparametrization `Λ \mapsto \alpha\Lambda + \beta`,

$$ \text{gap} \mapsto \alpha\,\text{gap}, \qquad \text{within} \mapsto \alpha^2\,\text{within}, \qquad \frac{\text{gap}^2}{\text{within}} \mapsto \frac{\alpha^2\text{gap}^2}{\alpha^2\text{within}} = \frac{\text{gap}^2}{\text{within}}. $$

The mean gap alone scales with `α` (calibration-dependent); `gap²/within` is **affine-invariant**
(calibration-free). That is the whole reason the settled currency is `gap²/within` rather than
the mean gap: on the uncalibrated marginal, only the invariant reading is trustworthy.

**Reporting.** The pose-marginal discriminability is reported as `gap²/within`,
`d'^2 = g^2/\operatorname{Var}(\Lambda)` on the converged bank (`d'^2(K)` on a bank of `K`
orientations), with the mean gap kept only as a stored diagnostic.

---

## 7. Known-pose *pool* discriminability `d'^2_\text{known}`

**Claim.**

$$ d'^2_{\text{known}} = \frac{\mathbb E[D]^2}{\mathbb E[D] + \operatorname{Var}[D]/4}. $$

**Assumptions.** Every particle's pose is known (so §4 holds per particle), but poses vary
across the pool, making `D = D(\text{pose})` a random variable with mean `E[D]`, variance
`Var[D]`.

**Derivation.** The pooled `Λ` is a two-stage draw: first the pose fixes `D`, then
`Λ \mid (\text{class}, D)` is the Gaussian of §4. Apply the laws of total expectation and
total variance.

*Pooled gap:*

$$ \mathbb E[\Lambda|A] = \mathbb E_D\bigl[\mathbb E(\Lambda|A,D)\bigr] = \mathbb E_D[D/2] = \tfrac12\mathbb E[D], \qquad \mathbb E[\Lambda|B] = -\tfrac12\mathbb E[D], $$
$$ \Rightarrow\quad \text{gap} = \mathbb E[D]. $$

*Pooled within-class variance* (class `A`; `B` identical by symmetry):

$$ \text{within} = \operatorname{Var}(\Lambda|A) = \underbrace{\mathbb E_D\bigl[\operatorname{Var}(\Lambda|A,D)\bigr]}_{\mathbb E_D[D]} + \underbrace{\operatorname{Var}_D\bigl[\mathbb E(\Lambda|A,D)\bigr]}_{\operatorname{Var}_D[D/2]} = \mathbb E[D] + \frac{\operatorname{Var}[D]}{4}. $$

Therefore `d'^2_{\text{known}} = \text{gap}^2/\text{within} = \mathbb E[D]^2/(\mathbb E[D] + \operatorname{Var}[D]/4)`. ∎

**Checkpoints.**
- *Homogeneous pool* (`Var[D] = 0`): `d'^2_\text{known} = E[D]²/E[D] = E[D]` — recovers the single-value
  answer of §4.
- *Heterogeneous pool* (`Var[D] > 0`): denominator inflates, so `d'^2_\text{known} < E[D]`. The pool's
  discriminability sits **below** the mean per-particle signal.
- The `/4` is precisely `Var(D/2)`: the class means `±D/2` wobble pose-to-pose, and the law
  of total variance dumps that wobble into `within`.

**Interpretation.** This departure of `within` from `gap` is *purely pose spread* — the
known-pose LLR is exactly Gaussian per particle, so there is no nonlinearity here. The
*marginal* rung (§8, unknown pose) adds a **second** inflation from the nonlinear marginal
ratio. The §7 `d'^2_\text{known}` (the pool deflection) isolates the first, and a retention taken against
it would isolate the second, the marginalization penalty.

**Flag (code mismatch, 2026-10).** The *shipped* survey retention (`kc_convergence.py`, appendix Note
`[[note:data]]`) divides the marginal pool deflection by `d2_known_pose = E[D] = I_zz` — the plain
pose-mean of `rungs/local.py`, not this §7 pool deflection — so its denominator carries no `Var[D]/4`
pose-spread term and the ratio is a marginal-deflection over a mean known-pose Fisher, not a
deflection-over-deflection. The appendix's `d'^2_known = I_zz` therefore matches the code; the §7
pool-deflection object is the distinct `_known_pose_rung` of `pipeline/chain.py`, not the retention
denominator. Reconcile by either renaming one of the two or routing the retention through the pool form.

**Provenance note.** The derivation above is independent of the code; a numerical cross-check against the known-pose rung is a separate
validation.

---

## 8. The pose-marginalization (CRLB) rung

**Claim.** With pose *unknown but locally estimated*, the conformational information is the
difference map with its pose-tangent components removed:

$$ d'^2_{\text{CRLB}} = I_{z\mid\phi} = I_{zz} - \mathbf I_{z\phi}\mathbf I_{\phi\phi}^{-1}\mathbf I_{\phi z} = \bigl\lVert \Delta\mu^{\perp}\bigr\rVert^2, $$

where `⊥` denotes the component orthogonal to the pose+shift tangent space (5 tangents:
3 rotation + 2 shift).

**Derivation.** As in §5, the Gaussian-model Fisher information between mean-image parameters is
the whitened inner product of the sensitivities,
`I_{\theta\theta'} = \langle \partial_\theta\mu, \partial_{\theta'}\mu\rangle`. With
`θ = (z, φ)` (conformation, pose):

$$ I_{zz} = \lVert\partial_z\mu\rVert^2 = \lVert\Delta\mu\rVert^2, \qquad \mathbf I_{z\phi} = \langle\partial_z\mu, \partial_\phi\mu\rangle, \qquad \mathbf I_{\phi\phi} = \langle\partial_\phi\mu, \partial_\phi\mu\rangle, $$

(`∂_z μ = Δμ` because the conformational coordinate interpolates `V_A ↔ V_B`). With pose a
nuisance parameter, the CRLB on the conformational variance is the inverse Schur complement,
so the available information is the Schur complement `I_{z|φ}`. For a Gram matrix of vectors
this equals the squared residual after projecting `∂_z μ` onto the pose-tangent subspace
`T_φ = \operatorname{span}\{\partial_\phi\mu\}`:

$$ I_{zz} - \mathbf I_{z\phi}\mathbf I_{\phi\phi}^{-1}\mathbf I_{\phi z} = \lVert\partial_z\mu - \operatorname{Proj}_{T_\phi}\partial_z\mu\rVert^2 = \lVert\Delta\mu^\perp\rVert^2. $$

∎ (This is the standard identity `‖a‖² − a^\top B(B^\top B)^{-1}B^\top a = ‖a_\perp‖²`.)

**Flag.** The *ordering* `d'^2 \le d'^2_{\text{CRLB}} \le I_{zz}` is a physical
expectation (removing information — marginalizing pose, then finite signal — can only lower
it), not an identity; the middle step is a bound, not an equality, because the marginal is a
curved-manifold distance, not a linear projection.

---

## 8b. The bank-convergence law extends from `d'^2` to the marginal LLR

**Claim.** The pose-bank deficit of the per-particle marginal log-ratio `Λ` — and so of its mean, its
gap and the pool deflection — follows the SAME `K/(K+K^*)` law as `d'^2` (§ Figure-5 convergence;
`statistics.convergence`), with the SAME `K^*`, **to leading order in the conformational difference**.
The scale that resolves the full pose does not enter; only the difference map's angular scale does.

**Setup.** For particle `i`, on a `K`-node bank,

$$ F_s^K = \log\sum_c w_c\,e^{\ell_s(i,c)}, \qquad \Lambda_i = F_A^K - F_B^K, \qquad \ell_s(\phi) = a\,\mathrm{Re}\langle Y_i, M_{s,\phi}\rangle_W - \tfrac12 a^2\lVert M_s\rVert_W^2. $$

The norm term is pose-independent, so the only `φ`-dependence is the cross term
`c_s(φ) = a\,Re⟨Y_i, M_{s,φ}⟩_W`.

**The difference splits off the common pose dependence.** The templates differ by `ΔM = M_A - M_B`, so
`c_A = c̄ + δ/2`, `c_B = c̄ - δ/2` with `δ(φ) = a\,Re⟨Y_i, ΔM_φ⟩_W`. The common part `c̄` (the whole
molecule, radius `R`) decorrelates under rotation on `θ_R ∼ 1/(k_rms R)`; the difference `δ` (extent
`ρ`) on `θ_ρ ∼ 1/(k_rms ρ)`. Since `R ≥ ρ`, `θ_R ≤ θ_ρ`: the pose posterior is SHARPER than the
difference is wide.

**The common pose posterior cancels.** Let `p(φ) ∝ w\,e^{ℓ̄}` be the posterior under the common part.
Its normalization factors out of both free energies, leaving a sum of ODD cumulants of `δ` under `p`
(the even ones, including the posterior-width term `κ_2`, are identical in both states and cancel):

$$ \Lambda = \log\langle e^{+\delta/2}\rangle_p - \log\langle e^{-\delta/2}\rangle_p = \langle\delta\rangle_p + \tfrac{1}{24}\kappa_3(\delta) + \cdots $$

To leading order in the change, `Λ_i ≈ ⟨δ⟩_p = ∫ p(φ)\,δ(φ)\,dφ`.

**Why the deficit is the difference's, not the pose's.** The bank deficit of `Λ` is the quadrature
error of `∫ p δ`. `p` is a sharp peak (width `θ_R`) at the true pose, so it samples `δ` there; a coarse
grid (`δ_K > θ_R`) misresolves `p` and samples at the nearest node, offset by `~δ_K` — but `δ` is
smooth on `θ_ρ ≥ θ_R`, so the value incurred changes only on the DIFFERENCE's scale `θ_ρ`. The error
is set by `θ_ρ`, giving `K^* ∼ (k_rms ρ)^3` of eq. (`kstar`) and, through the surface step in `Δμ`
(`γ = d+1`), the exponent `p = 1`. So `Λ(K)`, `gap(K)` and `d'^2(K)` share one `(·)K/(K+K^*)`. This is
the discrete echo of §8's `‖Δμ^⊥‖²`: the pose directions drop out of the DIFFERENCE even though they
dominate each free energy alone. ∎

**Flag.** The neglected term is `κ_3(δ)/24`; when the change is large (a big `δ`, or a high SNR that
makes `δ` effectively large) the odd cumulants reintroduce the pose scale `θ_R`, and the LLR then needs
a finer bank than `d'^2`. The leading-order result holds for the subtle changes this method targets;
whether a given pair is in that regime is read off its ladder — the `Λ`, gap and deflection ladders
collapsing onto one `K^*` is the test, which `deploy/orc/_cas9_probe.sh` measures (the probe is committed
but its outputs were not generated/committed as of 2026-10, so there is no real-pair result to cite yet).

---

## 9. From discriminability to particle count

**Claim.** Per particle, the Fisher information about the population fraction `p` runs
between a discriminability-limited and a counting-limited regime, and is bounded below by the
mean-score information:

$$ I_p(d'^2) \xrightarrow{d'^2\to 0} d'^2, \qquad I_p(d'^2)\xrightarrow{d'^2\to\infty} \frac{1}{p(1-p)}, \qquad \frac{1}{I_p} \le \frac{1}{d'^2} + p(1-p), $$

and `N = \text{criterion}/I_p` particles reach a target error on `p`.

**Setup.** The population is a mixture: fraction `p` in state `A`, `1−p` in `B`. Each particle
yields `Λ_i ∼ N(±d'^2/2,\ d'^2)` per §4 (write `u ≡ d'^2`). The per-particle density is
`f_{\text{mix}} = p f_A + (1-p) f_B`. The Fisher information about `p` is

$$ I_p = \int \frac{(f_A - f_B)^2}{f_{\text{mix}}}\,d\lambda. $$

**A useful simplification.** For these two Gaussians the LLR of the *densities* is `λ` itself:

$$ \log\frac{f_A(\lambda)}{f_B(\lambda)} = \frac{-(\lambda - u/2)^2 + (\lambda + u/2)^2}{2u} = \lambda \quad\Rightarrow\quad f_A = f_B\,e^{\lambda}. $$

Hence `f_A − f_B = f_B(e^\lambda − 1)`, `f_{\text{mix}} = f_B(p e^\lambda + 1 - p)`, and

$$ I_p = \int f_B\,\frac{(e^\lambda-1)^2}{p e^\lambda + 1 - p}\,d\lambda = \mathbb E_B\!\left[\frac{(e^\Lambda-1)^2}{p e^\Lambda + 1 - p}\right], \quad \Lambda\sim\mathcal N(-u/2,\,u). $$

**Large-`u` limit (counting).** The two Gaussians separate; on `A`'s support `f_{\text{mix}}\approx p f_A`,
on `B`'s support `\approx (1-p) f_B`, so

$$ I_p \to \frac1p\int f_A + \frac{1}{1-p}\int f_B = \frac1p + \frac1{1-p} = \frac{1}{p(1-p)}. $$

The ceiling is the Bernoulli counting information — you can read each particle's state, so all
that remains is the uncertainty in the proportion.

**Small-`u` limit (discriminability).** With `u` small, `Λ` is `O(\sqrt u)`, so
`e^\Lambda - 1 \approx \Lambda`, denominator `\approx 1`, and

$$ I_p = \mathbb E_B[\Lambda^2] + O(u^2), \qquad \mathbb E_B[\Lambda^2] = \operatorname{Var}_B(\Lambda) + \mathbb E_B[\Lambda]^2 = u + \tfrac{u^2}{4} \ \Rightarrow\ I_p = u + O(u^2) \to d'^2. $$

So a weak particle carries exactly its discriminability. (The retained `u²/4` is *not* `I_p`'s
full second-order coefficient — the `e^Λ-1 ≈ Λ` and denominator `≈ 1` steps already dropped other
`O(u²)` terms; only the leading `I_p → d'^2` is claimed.) ∎

**The mean-score variance (exact) and the bound.** Read the fraction from the mean score. Under
the mixture, `\mathbb E[\Lambda] = p\,u/2 - (1-p)\,u/2 = (p - \tfrac12)\,u`, and by total variance
`\operatorname{Var}[\Lambda] = u + p(1-p)\,u^2` (within-state `u` plus the between term). Inverting
the mean, `\hat p = \bar\Lambda/u + \tfrac12`, so

$$ \sigma_p^2 = \operatorname{Var}(\hat p) = \frac{\operatorname{Var}[\Lambda]}{N\,u^2} = \frac{p(1-p) + 1/d'^2}{N}, $$

*exactly*, for the equal-covariance model. By Cramér–Rao the maximum-likelihood fraction does at
least as well, `1/(N I_p) \le σ_p^2`, so this is an **upper bound** on the ML variance, i.e.
`1/I_p \le 1/d'^2 + p(1-p)`. It is met in both limits: it reproduces `I_p\to d'^2` (overlapping
states) and `I_p\to 1/[p(1-p)]` (separated states) derived above. The `fisher_from_deflection` map
evaluates the Gaussian integral `∫(f_A-f_B)^2/\text{mix}` of this section by quadrature
(`statistics/information.py`); between the limits it gives the smaller ML count.

**Count.** `Var(\hat p) = 1/(N\,I_p)`, so to reach standard error `\sigma_p`,

$$ N = \frac{1}{\sigma_p^2\,I_p} = \frac{\text{criterion}}{I_p(d'^2)}, \qquad \text{criterion} = 1/\sigma_p^2. $$

At `p = 0.5` the counting floor `p(1-p) = 1/4` is at its **maximum**, so `I_p` is smallest and —
for a fixed absolute target error `σ_p` — the required count is **largest**; `N` falls toward the
population extremes. So `p = 0.5` is the **worst-case (most conservative)** operating point for
absolute precision, not the best. (Under a fixed *relative* target `σ_p/p` the conclusion flips,
because pinning down a rare population is expensive — so which operating point is "favorable"
depends on which error is held fixed.)

**Local predictor.** Combining §8's capped per-pose CRLB with the count map `I_p` above, and
averaging over the Haar measure, gives the pool-free predictor of the converged marginal,

$$ I_{\text{local}} = \mathbb E_R\bigl[\,I_p(d'^2_\perp(R))\,\bigr], $$

with `d'^2_\perp(R) = \lVert\Delta\mu^\perp(R)\rVert^2` the per-pose CRLB of §8.

---

## 9b. The fraction MLE: `ρ = ⟨σ(Λ + logit p)⟩`

**Claim.** The maximum-likelihood estimate of the population fraction `p` is the fixed point of

$$ \rho = \Bigl\langle \operatorname{logit}^{-1}\!\bigl(\Lambda_i + \operatorname{logit}p\bigr)\Bigr\rangle, \qquad \hat p = \rho\big|_{\rho = p}, $$

where `logit^{-1} = σ` is the logistic sigmoid. The bracketed term is the per-particle posterior
responsibility, and the equation is simultaneously the E-step, the MLE score condition, and the
EM iteration.

**(a) The responsibility is a sigmoid of the LLR.** By Bayes, with prior `p` and per-particle
likelihoods `f_A = P(Y_i|A)`, `f_B = P(Y_i|B)`,

$$ r_i \equiv P(A\mid Y_i) = \frac{p f_A}{p f_A + (1-p) f_B} = \frac{1}{1 + e^{-(\Lambda_i + \operatorname{logit}p)}} = \sigma(\Lambda_i + \operatorname{logit}p), $$

using `Λ_i = log(f_A/f_B)` and `logit p = log(p/(1-p))`. So `ρ = ⟨r_i⟩` is the mean posterior
responsibility. (In the unknown-pose case `f_A, f_B` are the *marginal* likelihoods and `Λ_i` the
marginal LLR; the form is unchanged.)

**(b) It is the MLE score equation.** The pool log-likelihood is
`ℓ(p) = Σ_i log[p f_A^{(i)} + (1-p) f_B^{(i)}]`. Stationarity:

$$ \ell'(p) = \sum_i \frac{f_A^{(i)} - f_B^{(i)}}{p f_A^{(i)} + (1-p) f_B^{(i)}} = 0. $$

Rewrite the summand through `r_i`: since `f_A/\text{mix} = r_i/p` and `f_B/\text{mix} = (1-r_i)/(1-p)`,

$$ \frac{f_A - f_B}{\text{mix}} = \frac{r_i}{p} - \frac{1-r_i}{1-p}, \qquad \sum_i\!\left[\frac{r_i}{p} - \frac{1-r_i}{1-p}\right] = 0. $$

Multiplying through by `p(1-p)` gives `(1-p)\sum r_i = p\sum(1-r_i) = p(N - \sum r_i)`, hence
`\sum_i r_i = pN` and

$$ \hat p = \frac1N\sum_i r_i = \langle r_i\rangle \quad\Longleftrightarrow\quad \rho = p. $$

The MLE is the value at which the mean responsibility reproduces the prior. ∎

**(c) It is the EM iteration.** The E-step computes `r_i = σ(Λ_i + logit p^{(t)})`; the M-step for a
mixing proportion sets `p^{(t+1)} = ⟨r_i⟩`. So the iterate is

$$ p^{(t+1)} = \bigl\langle \sigma(\Lambda_i + \operatorname{logit}p^{(t)})\bigr\rangle, $$

whose fixed point is exactly the score-equation solution of (b). The sigmoid-average *is* the EM
update; "mean responsibility" (appendix Note D's fraction step) is not a heuristic but the MLE.

**(d) Precision ties back to §9.** The Fisher information of this estimator is
`I_p = ∫(f_A - f_B)^2/\text{mix}` — the quantity of §9 — so `Var(\hat p) = 1/(N I_p)` and
`N = criterion/I_p`. The fraction MLE `ρ` is precisely the estimator whose precision §9 turns into
a particle count.

**(e) The error, directly from the responsibilities.** The per-particle score of (b) is already in
hand: from `f_A/\text{mix} = r/p` and `f_B/\text{mix} = (1-r)/(1-p)`,

$$ \partial_p \log f_{\text{mix}} = \frac{f_A - f_B}{f_{\text{mix}}} = \frac{r}{p} - \frac{1-r}{1-p} = \frac{r - p}{p(1-p)}. $$

The information is its variance, and since `\mathbb E_{\text{mix}}[r] = p` at the truth (the score
equation of (b)),

$$ I_p = \mathbb E_{\text{mix}}\!\bigl[(\partial_p\log f_{\text{mix}})^2\bigr] = \frac{\operatorname{Var}_{\text{mix}}(r)}{[p(1-p)]^2}, \qquad \sigma_p = \frac{p(1-p)}{\sqrt{N\,\operatorname{Var}_{\text{mix}}(r)}}. $$

This is §9's `I_p = ∫(f_A-f_B)^2/\text{mix}` rewritten as a **variance of responsibilities** — the
same information, but a plug-in over the scored particles, needing no density estimate and free of
the tail sensitivity of the ratio `(f_A-f_B)^2/\text{mix}`. For the calibrated Gaussian of §4 it
reproduces `fisher_from_deflection(d'^2)`. ∎

**(f) The calibration gate `\mathbb E[r] = p`.** Every closed form for the count — §9's integral,
the Gaussian `fisher_from_deflection` map, and the responsibility variance of (e) — assumes
`r = σ(Λ + \operatorname{logit}p)` is the *true* posterior, i.e. that `Λ` is the calibrated LLR.
The one observable that reports whether it is holds is

$$ \mathbb E_{\text{mix}}[r] \stackrel{?}{=} p. $$

When it holds the three forms coincide and `σ_p` of (e) is exact. When it fails — as it does for an
imperfectly calibrated *marginal* `Λ`, where `\mathbb E[r] \ne p` — `r` is not the posterior,
`\operatorname{Var}(r)` conflates signal with miscalibration and the plug-in **over-reads** the
information, while the MLE is simultaneously biased. The honest error is then the sandwich (robust)
variance of the misspecified MLE, or a direct repeat-measurement of `σ_p`; no naive Fisher form
applies. This is the estimator-side face of §5–§6: only the affine-invariant `gap²/within` is
calibration-free, and only a calibration-free *estimator* (the two-pool MLE, §9c) restores the
count on an uncalibrated score.

**Naming flag.** This `ρ` (fraction MLE / mean responsibility) is *not* `rho_global = I_measured/I_local`
(the marginalization efficiency of §8). Same letter, different quantities — they need distinct
symbols in the prose.

---

## 9c. Pool calibration and the one- and two-pool MLEs

**Claim.** The three fraction estimators are the *same* MLE score equation `⟨r_i⟩ = p` of §9b,
differing only in how the responsibility `r_i` is formed from the reference data on hand — zero,
one, or two pure pools. Each added pool removes more of the calibration assumption the bare score
of §9b(a) smuggles in.

**(0 pools) `fraction_no_pool`.** Take `r_i = σ(Λ_i + \operatorname{logit}p)` — the score *as if*
`Λ` were the calibrated LLR (§9b). It is unbiased exactly when the calibration gate
`\mathbb E_{\text{mix}}[r] = p` of §9b(f) holds. When it fails — the marginal `Λ` is on an arbitrary
scale, so `r` is not the posterior — the fixed point is **biased** (compressed toward ½ under
overdispersion, otherwise as the miscalibration runs), and with
no reference it cannot learn the true scale, so the bias is irreducible at this rung. (Overdispersion
`within > gap` is a Gaussian *symptom*, not the condition: a strongly separated pool can have
`within > gap` from pose spread yet still pass `\mathbb E[r] = p` — the gate, not the moment ratio,
is decisive.)

**(1 pool) `fraction_one_pool`.** Measure one component density `f_A` from a pure pool, and supply the
other by reflection. The score is the log-ratio `Λ = \log[P(Y|A)/P(Y|B)]`, which maps to `-Λ` when the
two states are exchanged; so under swap symmetry the unseen state's density is the measured one
reflected about the decision boundary `Λ = 0`,

$$ f_B(x) = f_A(-x). $$

Run the §9b EM for the mixing weight against `(f_A, f_B)` exactly as the two-pool route does. This is
the **reference MLE with the missing reference mirrored**: it keeps the two-pool route's
calibration-freedom for the shape it measures and adds only swap symmetry for the shape it does not.
Exact when the two states are exchangeable in their score distribution; it degrades where they are
not — and, unlike a fitted midpoint, reflecting about the fixed boundary `Λ = 0` needs no parameter, so
it cannot wander on a flat likelihood ridge. ∎

**(2 pools) `fraction_two_pool`.** Estimate both component densities `f_A, f_B` from the two pure
pools and run the §9b EM for the mixing weight against them:
`r_i = p\,f_A(Λ_i)/[p\,f_A(Λ_i) + (1-p)\,f_B(Λ_i)]`, `p^{(t+1)} = ⟨r_i⟩`. Because it uses the
measured densities rather than assuming `Λ` is the LLR, it is **nonparametric and
calibration-free**: it reads correctly at any scale and is unmoved by the miscalibration that
biases the 0-pool estimate. It is the reference MLE of §9b with the responsibility computed
honestly, and it is the recourse whenever the calibration gate of §9b(f) fails.

**Interpretation.** The ladder is one of *reference information*: 0/1/2 pools buy, respectively, no
calibration, one measured density plus its reflection, and both measured densities. On a calibrated score all three agree
with §9b; on an uncalibrated marginal score only the two-pool MLE recovers the fraction, and its
`σ_p` must then come from a repeat-measurement or the sandwich variance, not the plug-in of §9b(e).
So a template bank (calibrated by construction) is counted with §9b(e); a real pool is counted with
`fraction_two_pool` once §9b(f) shows its marginal `Λ` is not calibrated.

---

## 10. Absolute-scale anchor (`κ`)

**Claim.** `\kappa = \mathrm{SNR}_{\text{ref}}^2/\langle P\rangle_{\text{pool}}` converts a
predictor computed in gauge units into absolute particle counts.

**Derivation.** By §3 the pooled power `⟨P⟩_pool` is computed at `contrast = 1`, i.e. in gauge
units — not absolute, because `contrast = 1` is physical only jointly with `σ²`. Anchoring to
one measured reference collection with known `SNR_ref` fixes the unit:

$$ \mathrm{SNR}^2_{\text{absolute}} = \kappa\cdot P, \qquad \kappa = \frac{\mathrm{SNR}_{\text{ref}}^2}{\langle P\rangle_{\text{pool}}}. $$

This is one unit calibration for the whole panel — consistent with "SNR is an output": within a
run `SNR` still falls out of `(dose, optics, contrast=1)`; `κ` only sets the shared unit. Moving
dose/flexibility/defocus moves `⟨P⟩_pool`, and `κ` absorbs most of it, so a reader who moves the
plan without re-anchoring `κ` gets an error of the opposite sign.

**Scope.** `κ` belongs to the **absolute-count predictor** panel (Fig 4), which needs an
absolute scale. Figure 5 (retention `= d'^2/d'^2_\text{known}`, a ratio; `K^\ast`, a bank count) is
self-contained and needs no anchor.

---

## 11. The state-difference predictor (eq. 1) — mechanistic form, one fitted constant

**Flag: the exponents are theory, the amplitude is fitted.** Eq. 1 is the frozen mechanistic form

$$ d'^2\big|_{\text{ref}} = C \cdot M \cdot \frac{\mathrm{RMSD}^2}{1 + (\mathrm{RMSD}/r^*)^2}, $$

with `M` in kDa, stated at the reference dose of 40 e⁻/Å². The survey is collected at that single
dose and no dose term is fitted, so dose is the condition the form is stated at, not a variable of
it. Every
structural piece is derived from §4; only `C` is fitted, and the paper anchors even that via `κ`.

- **`M¹` and `RMSD²` (numerator).** `d'^2 = \lVert\Delta\mu\rVert^2 = \sum_k |\Delta M(k)|^2/\sigma^2(k)`
  (§4) is the whitened power of the difference density. In the small-displacement (linear-response)
  limit the difference density is a gradient times the displacement, `\Delta V \approx (\mathrm{RMSD}
  \cdot\nabla)V`, so its power is quadratic in the displacement and `d'^2 \propto \mathrm{RMSD}^2`. The
  moving mass enters *linearly*, not quadratically: the displaced scattering adds to the difference
  power **extensively** — each moved unit contributes its own `RMSD²` and the contributions sum in
  power, not amplitude, at the resolutions that carry the signal — so `d'^2 \propto M \cdot
  \mathrm{RMSD}^2`. (Density-linear-in-`M` would give `M²` power; the `M¹` is the incoherent sum, not a
  linear amplitude.) The free fit returns a mass exponent indistinguishable from 1, the empirical
  statement of that extensivity, so freezing `M¹` costs no fit quality.
- **`/(1+(\mathrm{RMSD}/r^*)^2)` (denominator).** The quadratic rise cannot continue indefinitely:
  once a moved atom's phase `2πk·u` completes about a full cycle at the spatial frequency where the whitened
  difference spectrum centers, moving it further adds no new difference power. A quadratic rise that
  saturates at a plateau *is* a `q=2` denominator, so "`\mathrm{RMSD}^2` and `q=2`" are one physical
  statement, not two knobs. `r^*` is that phase-overlap scale, set by the specimen disorder, pixel and
  envelope, imported from the model rather than fitted — not a map resolution.
- **Limits.** Small motion `d'^2 ≈ C\,M\,\mathrm{RMSD}^2`; at `\mathrm{RMSD}=r^*`, half-saturation;
  large motion `d'^2 → C\,M\,r^{*2}`, a **mass-limited ceiling**. A bare power law has no ceiling and would
  predict unbounded `d'^2`, which is unphysical.

**Amplitude.** Each pair's plateau `d'^2` is the `K/(K+K^*)` extrapolation of its converged pose-marginal
ladder (§ Figure-5 convergence), not the top achieved rung, which undershoots reach-dependently.

**Fit quality.** The frozen one-parameter form fits essentially as well as the free fit with `r^*`
and `q` released, so the data independently prefer the frozen constants. The fit statistics themselves
(`n`, `R²`, the released `r^*` and `q`) are reported in the manuscript from `results/eq1_mechanistic.json`,
not fixed here.

**Cross-check — why not a bare power law.** A bare power law in `M` and `RMSD` (optionally dose) has
*more* free parameters yet fits worse. The reason is intrinsic: the effective RMSD exponent runs from
above one at small motions to below one at large, which a single power cannot represent but the
saturating denominator produces by construction. The mechanistic form is therefore the *simpler* model
on the axis that matters (one fitted degree of freedom) and the better-fitting one; the comparison
statistics are in the manuscript.

---

## What is derived vs. what to verify

| Relation | Status |
|---|---|
| §2 shot `= box²/(4·dose·pixel²)` | derived, exact |
| §2 additive ice term | reconciled — no such term in code or appendix |
| §3 `SNR = contrast·G`, `d² ∝ contrast²` | derived, exact |
| §4 per-particle `Λ ∼ N(±D/2, D)`, `D = ‖Δμ‖²` | derived, exact |
| §5 currency `d'² ≡ gap²/within` = Fisher info of state MLE | derived; exact = full Fisher at known pose, deflection lower bound under marginalization |
| §6 three identities + affine invariance | derived, exact |
| §7 known-pose pool `d'²_known = E[D]²/(E[D]+Var[D]/4)` | derived, exact |
| §8 CRLB `= ‖Δμ^⊥‖²` (Schur = projection) | derived, exact |
| §8b marginal-LLR bank deficit shares `d'²`'s `K/(K+K^*)`, `p=1` | derived to leading order in the change; the common pose posterior cancels in the free-energy difference; `κ_3(δ)` term breaks it for large changes |
| §9 limits `I_p → d'²` and `→ 1/[p(1−p)]` | derived, exact |
| §9 mean-score variance `σ_p² = [p(1−p) + 1/d'²]/N` | derived, exact for the mean-score estimator (equal-covariance model); upper bound on the ML variance by Cramér–Rao, met in both limits |
| §9b fraction MLE `p̂ = ⟨σ(Λ + logit p)⟩` | derived, exact |
| §9b(e) direct error `σ_p = p(1−p)/√(N·Var(r))` | derived, exact (= §9 integral for calibrated `Λ`) |
| §9b(f) calibration gate `E[r] = p` | derived; a diagnostic, and the boundary of every closed form |
| §9c one-pool reference MLE, unseen density mirrored `f_B(x)=f_A(−x)` | derived, exact for exchangeable states |
| §9c two-pool reference MLE (calibration-free) | derived, exact |
| §10 `κ` anchor | derived, exact |
| §11 eq. 1 form (`M¹`, `RMSD²`, `q=2` saturation, `r*`) | derived; only amplitude `C` fitted (anchored via κ) |
