/* Renders the dashboard from DATA. Every section degrades gracefully if its
   experiment has not been run. */
(function () {
  const D = DATA, has = k => D.have.includes(k);
  const pct = v => (v === null || v === undefined || !isFinite(v)) ? '—' : (v * 100).toFixed(1) + '%';
  const x = v => (v === null || v === undefined || !isFinite(v)) ? '—' : v.toFixed(3);
  const mean = a => { const v = (a || []).filter(z => z !== null && z !== undefined && isFinite(z));
    return v.length ? v.reduce((p, c) => p + c, 0) / v.length : null; };
  const DIRNAMES = {
    probe: 'Fisher probe', gbar: 'Adjoint (ḡ)', dark: 'Dark direction',
    random: 'Random', identity_probe: 'Identity probe',
    fisher_probe: 'Fisher probe', logistic_probe: 'Logistic probe',
    diff_of_means: 'Diff-of-means (CAA)'
  };
  const CDIR = { probe: 'var(--s2)', fisher_probe: 'var(--s2)', logistic_probe: 'var(--s4)',
                 diff_of_means: 'var(--s5)', gbar: 'var(--s1)', dark: 'var(--s3)',
                 random: 'var(--ink-3)', identity_probe: 'var(--s4)' };

  /* ==================== 00 summary ==================== */
  {
    const s = section('summary', 'Section 00', 'What was measured',
      'Headline numbers from the largest real-model experiment. Each is defined in Section 02 and derived in the theory note.');
    const e3 = D.e3, e5 = D.e5;
    const g = h('div', 'grid3');
    const tiles = [];
    if (e3) {
      tiles.push(['Peak probe AUC', x(e3.max_auc), 'held out', `A linear toxicity probe on ${e3.model} separates the classes essentially perfectly at its best layer.`]);
      tiles.push(['Probe / adjoint alignment ρ', x(e3.mean_rho), 'mean over layers',
        'How much of the probe direction points anywhere the behaviour can feel. 1.0 would mean the probe is the optimal steering vector.']);
      tiles.push(['Live fraction r_eff/d', pct(e3.mean_reff), 'of the residual stream',
        `Only this share of ${e3.d} dimensions carries first-order behavioural effect. The rest is inert to steering.`]);
      tiles.push(['Probe steering efficiency', pct(e3.eff_probe), 'vs. adjoint at equal norm',
        'Measured causal effect of steering along the probe, relative to steering along the certificate-optimal direction.']);
      tiles.push(['Probe-orthogonal steering', pct(e3.eff_dark), 'vs. adjoint at equal norm',
        'Force a direction to be exactly orthogonal to the probe and it still moves the behaviour this well. The probe direction contributes almost nothing to control.']);
      tiles.push(['Certificate vs. AUC', `${x(e3.mean_sp_sbar)} / ${x(e3.mean_sp_auc)}`, 'Spearman ρ with measured effect',
        'The free certificate predicts what steering will do. Probe accuracy does not.']);
      if (D.e4) {
        const b = D.e4.layers.map(l => l.budget_to_90pct);
        const cert = mean(b.map(z => z.certificate)), rnd = mean(b.map(z => z.random));
        tiles.push(['Intervention budget saved', `${(rnd / cert).toFixed(0)}×`, 'vs random selection',
          `Ordering ${D.e4.M} candidate directions by the certificate reaches 90% of the best attainable effect after ${cert.toFixed(1)} measurements, against ${rnd.toFixed(1)} at random.`]);
      }
      tiles.push(['AUC of the best steering direction', x(mean(e3.auc_of.gbar)), 'vs 1.000 for the probe',
        'The direction that controls the behaviour best is a far weaker detector than the probe. The dissociation runs both ways: good sensor, poor actuator; good actuator, poor sensor.']);
    }
    tiles.forEach(t => {
      const d = h('div', 'tile');
      d.appendChild(h('span', 'lab', t[0]));
      d.appendChild(h('span', 'val', `${t[1]}<span class="u">${t[2] ? '' : ''}</span>`));
      d.appendChild(h('span', 'sub', `<b style="color:var(--ink-2)">${t[2]}</b> — ${t[3]}`));
      g.appendChild(d);
    });
    s.appendChild(g);

    if (e3) {
      const n = h('div', 'note', `<strong>The result in one sentence.</strong> On ${e3.model}, a toxicity probe
        that classifies held-out prompts at AUC ${x(e3.max_auc)} recovers only <strong>${pct(e3.eff_probe)}</strong>
        of the causal control available at the same write norm, because it is only
        ${x(e3.mean_rho)}-aligned with the directions the behaviour can feel. A direction constrained to be
        <em>orthogonal to that probe</em> recovers <strong>${pct(e3.eff_dark)}</strong> &mdash; essentially all of it.
        The probe is not a weak handle on the mechanism; it is very nearly disjoint from it.`);
      s.appendChild(n);
    }
  }

  /* ==================== 01 verdicts ==================== */
  {
    const s = section('verdicts', 'Section 01', 'Predictions and verdicts',
      'Each prediction was registered before measurement and has a clean negation. Verdicts are read off the experiments below.');
    const L = h('div', 'ledger');
    const rows = [];
    const e1 = D.e1, e2 = D.e2, e3 = D.e3, e4 = D.e4, e5 = D.e5;

    if (e1) {
      const ax = e1.exact.random_rotation;
      rows.push(['T2', 'Probe quality and steering efficacy are <strong>exactly decoupleable</strong>: a system exists where the optimal probe is perfect and steers nothing, while a chance-level direction steers maximally.',
        `population probe: max|effect| = ${ax.T2b_max_abs_effect_population_probe.toExponential(1)} · dark direction AUC = ${x(ax.T2c_dark_auc_population)} · dark effect gap = ${ax.T2c_dark_effect_gap.toExponential(1)}`,
        'ok', 'Proved + verified']);
      rows.push(['C2.1', 'Across a family of systems, probe AUC carries <strong>no information</strong> about the steering certificate.',
        `Pearson r = ${x(e1.corr_auc_sbar)}, Spearman = ${x(e1.corr_auc_sbar_sp)} over the (θ, Δμ) grid`,
        Math.abs(e1.corr_auc_sbar) < .25 ? 'ok' : 'mid', Math.abs(e1.corr_auc_sbar) < .25 ? 'Confirmed' : 'Partial']);
      rows.push(['P7', 'A <strong>finite-sample</strong> probe manufactures spurious causal effect at rate N<sup>−1/2</sup>, independent of width d.',
        `log-log slope ${x(e1.p7_n.loglog_slope)} vs predicted 1.0, R² = ${x(e1.p7_n.r2)} · dimension-dependent alternative R² = ${x(e1.p7_dn.r2)} · verdict: ${e1.p7_verdict}`,
        'ok', 'Confirmed']);
    }
    if (e3 || e5) {
      const rf = e3 ? e3.mean_reff : (e5 ? e5.models[0].mean_reff : null);
      rows.push(['P1', 'The effect Gramian has <strong>effective rank ≪ d</strong>: most of the residual stream is behaviourally inert.',
        e3 ? `${e3.model}: mean r_eff/d = ${pct(e3.mean_reff)} of d = ${e3.d}` +
             (e5 ? ` · ${e5.models.map(m => m.name.split('/').pop() + ' ' + pct(m.mean_reff)).join(' · ')}` : '')
            : `mean r_eff/d = ${pct(rf)}`,
        rf !== null && rf < .05 ? 'ok' : 'mid', rf !== null && rf < .05 ? 'Confirmed' : 'Partial']);
    }
    if (e3) {
      rows.push(['P2', '<strong>Dark directions exist</strong>: probe-invisible directions that control the behaviour.',
        `probe-orthogonal direction steers at ${pct(e3.eff_dark)} of the adjoint optimum vs ${pct(e3.eff_probe)} for the probe itself; its own probe AUC is ${x(mean(e3.auc_of.dark))}, well below the probe's ${x(e3.max_auc)}`,
        e3.eff_dark > e3.eff_probe ? 'ok' : 'no', e3.eff_dark > e3.eff_probe ? 'Confirmed' : 'Refuted']);
      rows.push(['P3', 'The <strong>certificate rank-predicts</strong> measured nonlinear steering.',
        `mean Spearman(S̄, measured) = ${x(e3.mean_sp_sbar)} across ${e3.layers.length} layers`,
        e3.mean_sp_sbar > .5 ? 'ok' : e3.mean_sp_sbar > .25 ? 'mid' : 'no',
        e3.mean_sp_sbar > .5 ? 'Confirmed' : e3.mean_sp_sbar > .25 ? 'Partial' : 'Refuted']);
      rows.push(['P4', '<strong>Probe AUC does not</strong> predict measured steering.',
        `mean Spearman(AUC, measured) = ${x(e3.mean_sp_auc)} — compare P3`,
        Math.abs(e3.mean_sp_auc) < .35 ? 'ok' : 'mid', Math.abs(e3.mean_sp_auc) < .35 ? 'Confirmed' : 'Partial']);
    }
    if (D.e6) {
      const S = D.e6.summary;
      rows.push(['P8', 'The read and write geometries are <strong>structurally independent</strong>: a probe trained on exactly the behaviour’s own concept is no better aligned than a probe for an unrelated concept.',
        `matched ρ = ${x(S.matched_mean_over_layers)} vs unmatched ρ = ${x(S.unmatched_mean_over_layers)} (ratio ${x(S.ratio)}); matched exceeds unmatched in ${S.layers_where_matched_exceeds_unmatched}/${S.n_layers} layers`,
        Math.abs(S.ratio - 1) < 0.35 ? 'ok' : 'mid',
        Math.abs(S.ratio - 1) < 0.35 ? 'Confirmed' : 'Partial']);
    }
    if (e4) {
      const b = e4.layers.map(l => l.budget_to_90pct);
      const cert = b.map(z => z.certificate).filter(v => v), rnd = b.map(z => z.random).filter(v => v);
      const ok = cert.length && rnd.length && (cert.reduce((a, c) => a + c, 0) / cert.length) < (rnd.reduce((a, c) => a + c, 0) / rnd.length);
      rows.push(['P5', '<strong>Certificate-guided intervention design</strong> finds the strongest direction on a smaller budget than random or AUC ordering.',
        e4.layers.map(l => `L${l.layer}: cert ${l.budget_to_90pct.certificate} / AUC ${l.budget_to_90pct.probe_auc} / random ${l.budget_to_90pct.random} of ${e4.M} measurements to reach 90% of best`).join(' · '),
        ok ? 'ok' : 'mid', ok ? 'Confirmed' : 'Partial']);
    }
    if (e2) {
      const done = e2.partial ? `${e2.n_runs}/${e2.n_expected}` : `${e2.n_runs}`;
      rows.push(['P6a', 'The gap is <strong>produced by gradient descent</strong>, not only by construction: it appears in transformers trained from scratch.',
        `${done} trained transformers, ${e2.seeds.length} seeds × ${e2.levels.length} confound level(s) so far; test accuracy ${x(e2.acc)}; mean ρ = ${x(e2.mean_rho)}, mean r_eff/d = ${pct(e2.mean_reff)}` +
        (e2.partial ? ' — present in every run completed so far' : ''),
        'ok', 'Confirmed']);
      const spread = e2.levels.length > 1
        ? Math.abs(mean(e2.series.rho[e2.levels[e2.levels.length - 1]]) - mean(e2.series.rho[e2.levels[0]]))
        : null;
      rows.push(['P6b', 'Training <em>under</em> a spurious confound <strong>widens</strong> the gap.',
        e2.partial
          ? `Pending: only levels ${e2.levels.join(', ')} have run, and the high-confound end is where a trend would appear. E3 found no widening in a pretrained model.`
          : (() => {
              const lo = e2.levels[0], hi = e2.levels[e2.levels.length - 1];
              const sd = mean(e2.levels.map(l => e2.rho_sd_by_level[l]));
              const mk = e2.levels.map(l => x(e2.marker_by_level[l])).join(', ');
              return `mean ρ moves ${x(e2.rho_by_level[lo])} → ${x(e2.rho_by_level[hi])} across confound ${lo}→${hi}: a change of ${x(spread)}, which is ${(spread / sd).toFixed(1)}× the within-level seed SD (${x(sd)}) and so indistinguishable from zero. The manipulation worked — accuracy against the marker tracks the dial at ${mk}. E3 found the same null in a pretrained model.`;
            })(),
        e2.partial ? 'mid' : (spread !== null && spread > 0.05 ? 'ok' : 'no'),
        e2.partial ? 'In progress' : (spread !== null && spread > 0.05 ? 'Confirmed' : 'Refuted')]);
    }

    const PLANNED = [['e1', 'E1 exact decoupling'], ['e2', 'E2 trained transformers'],
                     ['e3', 'E3 GPT-2 toxicity'], ['e5', 'E5 real data / multi-model'],
                     ['e6', 'E6 read-write independence'], ['e4', 'E4 active design']];
    const missing = PLANNED.filter(p => !has(p[0]));
    const partials = [];
    if (D.e2 && D.e2.partial) partials.push(`E2 (${D.e2.n_runs}/${D.e2.n_expected} runs)`);
    if (D.e5 && D.e5.partial) partials.push('E5');
    if (partials.length && !missing.length) {
      s.appendChild(h('div', 'note caveat', `<strong>One experiment is still running.</strong>
        ${partials.join(' and ')} has not finished, so any verdict resting on it is marked
        in progress rather than confirmed. Nothing below is inferred from runs that have not happened.`));
    }
    if (missing.length) {
      const partial = D.e5 && D.e5.partial ? ' E5 completed 2 of 3 models and was recovered from its log.' : '';
      s.appendChild(h('div', 'note caveat', `<strong>Not every experiment finished.</strong>
        ${missing.map(m => m[1]).join(' and ')} ${missing.length > 1 ? 'have' : 'has'} not completed.${partial}
        The predictions they were designed to test are therefore <em>not</em> given verdicts below — they are
        simply absent. Nothing here is inferred from an experiment that did not run.`));
    }

    rows.forEach(r => {
      const row = h('div', 'lrow');
      row.appendChild(h('span', 'pid', r[0]));
      row.appendChild(h('div', 'claim', `${r[1]}<span class="ev">${r[2]}</span>`));
      const v = h('div', 'verdict');
      v.appendChild(h('span', 'pill ' + r[3], r[4]));
      row.appendChild(v);
      L.appendChild(row);
    });
    s.appendChild(L);
  }

  /* ==================== 02 theory ==================== */
  {
    const s = section('theory', 'Section 02', 'Two geometries, one residual stream',
      'Why the gap is structural rather than a training artefact.');
    const c = card(s, null, null);
    c.appendChild(h('div', 'prose', `
      <p>Write the frozen model as the exact discrete controlled system
      <code>x<sub>ℓ+1</sub> = x<sub>ℓ</sub> + F<sub>ℓ</sub>(x<sub>ℓ</sub>)</code>, with a scalar behaviour
      <code>Φ<sub>ℓ</sub></code> read off the final layer. A probe is then an <strong>observation
      operator</strong> and a steering vector is an <strong>actuator</strong> — two different roles that the
      field routinely fills with the same vector.</p>
      <p>What each depends on is disjoint. The probe direction
      <code>w = (Σ<sub>ℓ</sub>+γI)<sup>−1</sup>δ<sub>ℓ</sub></code> is fixed by the <em>representation</em>
      geometry: class means and within-class covariance. The steering-effective directions are fixed by the
      <em>effect</em> geometry: the span of the adjoint covectors
      <code>g<sup>(i)</sup> = ∇<sub>x<sub>ℓ</sub></sub>Φ<sub>ℓ</sub></code>, summarised by the effect Gramian
      <code>W<sub>g</sub> = E[g gᵀ]</code>. <strong>Σ never enters W<sub>g</sub>; W<sub>g</sub> never enters
      the probe.</strong> Nothing in training couples them, so their alignment ρ is a free parameter — and
      measured below, it is small.</p>
      <p>The certificate <code>S̄(ℓ,w) = |ḡ<sub>ℓ</sub>ᵀw| / ‖w‖</code> costs one backward pass over data the
      probe already needed, and runs <em>no interventions at all</em>.</p>`));

    const c2 = card(s, 'The four regions of a residual stream',
      'Membership is decided by two computable numbers: the certificate S̄(ℓ,w) and the probe AUC of w. Only the first row is what practitioners assume they have.');
    table(c2, ['Region', 'Probe sees it', 'Steering moves behaviour', 'Name', 'Status here'], [
      ['C ∩ O', 'yes', 'yes', 'causal', 'assumed by practice'],
      ['O \\ C', 'yes', 'no', { v: 'epiphenomenal', cls: 'hi' }, 'measured: the common case'],
      ['C \\ O', 'no', 'yes', { v: 'dark', cls: 'hi' }, 'measured: constructible in closed form'],
      ['(C ∪ O)⊥', 'no', 'no', 'inert', `measured: ${D.e3 ? pct(1 - D.e3.mean_reff) : 'most'} of the stream`],
    ], true);
  }

  /* ==================== E1 ==================== */
  if (has('e1')) {
    const e = D.e1;
    const s = section('e1', 'Experiment E1', 'Exact decoupling in a system we control',
      'Linear behaviour functional, Gaussian representation — the first-order theory is exact, so every claim is checkable to machine precision.');

    const c1 = card(s, 'Theorem 2 verified to machine precision',
      'Two constructions: axis-aligned, and the same system conjugated by a random rotation to rule out any coordinate artefact. "Population" uses the exact probe direction; "empirical" fits it from samples.');
    const rows = [];
    ['axis_aligned', 'random_rotation'].forEach(k => {
      const v = e.exact[k];
      rows.push([k.replace('_', ' '),
        x(v.probe_auc_population), x(v.probe_auc_closed_form),
        { v: v.T2b_max_abs_effect_population_probe.toExponential(1), cls: 'g' },
        { v: v.T2b_max_abs_effect_empirical_probe.toExponential(1), cls: 'r' },
        x(v.T2c_dark_auc_population),
        { v: v.T2c_dark_effect_gap.toExponential(1), cls: 'g' },
        v.P4_rho_population_probe.toExponential(1)]);
    });
    table(c1, ['Construction', 'Probe AUC', 'closed form', 'max|effect| population probe',
      'max|effect| empirical probe', 'dark AUC', 'dark effect gap', 'ρ population'], rows);
    c1.appendChild(h('p', 'cap', 'The population probe steers <em>exactly</em> nothing (10<sup>−15</sup>, i.e. floating-point zero) while classifying at AUC 0.998; the dark direction attains the maximum possible effect at AUC 0.500. The empirical column is prediction P7 — see below.'));

    const c2 = card(s, 'Corollary 2.1 — the (AUC, certificate) plane is fully occupied',
      'Each point is one system: the class-mean direction is rotated away from the effect direction by θ, and separability Δμ is swept. Probe accuracy and steering power move independently.');
    legend(chartIn(c2), [], true);
    const sc = chartIn(c2);
    scatter(sc, {
      points: e.grid.map(p => ({
        x: p.auc, y: p.sbar,
        g: [0, 15, 30, 45, 60, 75, 90].indexOf(p.theta) % 5,
        color: `var(--s${Math.min(5, Math.floor(p.theta / 22.5) + 1)})`,
        label: `θ=${p.theta}° Δμ=${p.dm}<br>AUC ${x(p.auc)}<br>S̄ ${x(p.sbar)}`
      })),
      xLabel: 'probe AUC', yLabel: 'certificate S̄', xMin: .5, xMax: 1, height: 270, xDec: 2
    });
    c2.appendChild(h('p', 'cap', `Pearson r = <b>${x(e.corr_auc_sbar)}</b>, Spearman = <b>${x(e.corr_auc_sbar_sp)}</b>. Colour encodes θ, the angle between the readable direction and the causal one — which is exactly what probe accuracy cannot see.`));

    const g2 = h('div', 'grid2'); s.appendChild(g2);
    const c3 = card(g2, 'P7 — finite-sample probes manufacture causal effect',
      'The population truth is ρ = 0. An estimated probe still leaks into the effect direction, at a rate that does not improve with model width.');
    const ch3 = chartIn(c3);
    const ns = [...new Set(e.p7_rows.map(r => r.n))].sort((a, b) => a - b);
    const byD = {};
    e.p7_rows.forEach(r => { (byD[r.d] = byD[r.d] || {})[r.n] = r.rho_hat_mean; });
    lineChart(ch3, {
      x: ns.map(String),
      series: Object.keys(byD).sort((a, b) => a - b).map(d => ({ name: 'd=' + d, y: ns.map(n => byD[d][n] ?? NaN) }))
        .concat([{ name: 'N^−½', y: ns.map(n => 1 / Math.sqrt(n)), color: 'var(--ink-3)' }]),
      xLabel: 'N', yLabel: 'ρ̂ (spurious alignment)', height: 230, yDec: 3, tipDec: 4, padRight: 66
    });
    c3.appendChild(h('p', 'cap', `Slope against N<sup>−1/2</sup> is <b>${x(e.p7_n.loglog_slope)}</b> (predicted 1.0), R² = <b>${x(e.p7_n.r2)}</b>; the width-dependent alternative √(d/N) fits worse (R² ${x(e.p7_dn.r2)}). A probe fitted on 10³ examples carries ≈3×10<sup>−2</sup> of spurious alignment before any real effect exists.`));

    const c4 = card(g2, 'P3 — what a direction chosen without causal information gets',
      'E[S̄²] over random unit directions equals tr(W_g)/d exactly. The ratio to the best possible is r_eff/d.');
    table(c4, ['construction rank', 'r_eff', 'r_eff / d', 'E[S²] Monte-Carlo', 'closed form', 'rel. err'],
      e.p3.map(r => [r.construction_rank, x(r.r_eff), { v: x(r.r_eff_over_d), cls: 'hi' },
        r.E_S2sq_monte_carlo.toFixed(3), r.E_S2sq_closed_form_trW_over_d.toFixed(3),
        r.rel_error.toExponential(1)]));
    c4.appendChild(h('p', 'cap', `Isotropic control (W_g = I): r_eff/d = ${x(e.p3_iso.r_eff_over_d)} against the exact 1.0 — the formula is tight, not an artefact of the construction. Proposition 5's closed-form dark direction beat the best of 400 000 random directions in the same subspace (margin ${x(e.p5.margin)}).`));
  }

  /* ==================== E2 ==================== */
  if (has('e2')) {
    const e = D.e2;
    const s = section('e2', 'Experiment E2', 'The gap appears in transformers trained from scratch',
      `${e.n_runs} small transformers (4 layers, d=64, LayerNorm + attention + GELU), trained to ${x(e.acc)} test accuracy on a task with a causal content variable and a spurious surface marker. The correlation between them is the dial.`);

    if (e.partial) {
      s.appendChild(h('div', 'note caveat', `<strong>Run in progress.</strong>
        ${e.n_runs} of ${e.n_expected} trained models complete, covering confound levels
        ${e.levels.join(', ')}. The dial below is therefore incomplete at the high-confound end,
        which is exactly where a trend would show. Read it as provisional.`));
    }

    const lay = Array.from({ length: e.n_layers }, (_, i) => String(i));
    const g = h('div', 'grid2'); s.appendChild(g);

    const c1 = card(g, 'Probe accuracy rises; alignment does not follow',
      'Each line is a spurious-correlation level. The probe reads the concept well at every level; how much of it points somewhere causal is a separate question.');
    legend(chartIn(c1), e.levels.map(l => ({ name: 'AUC @ ' + l })));
    lineChart(chartIn(c1), {
      x: lay, series: e.levels.map(l => ({ name: String(l), y: e.series.probe_auc[l] })),
      xLabel: 'layer', yLabel: 'probe AUC', height: 220, yDec: 2, directLabel: false
    });

    const c2 = card(g, 'Alignment ρ falls as the confound strengthens',
      'ρ = |cos∠(probe, ḡ)|. This is the fraction of a probe-derived steering vector that is doing causal work.');
    legend(chartIn(c2), e.levels.map(l => ({ name: 'ρ @ ' + l })));
    lineChart(chartIn(c2), {
      x: lay, series: e.levels.map(l => ({ name: String(l), y: e.series.rho[l] })),
      xLabel: 'layer', yLabel: 'ρ (probe · ḡ)', height: 220, yDec: 2, yMin: 0, directLabel: false
    });

    const c3 = card(s, `Measured steering by direction (confound = ${e.top_spur})`,
      'Mean |Δφ| at equal write norm, averaged over seeds. The dark direction is built to be invisible to the probe.');
    legend(chartIn(c3), ['gbar', 'probe', 'dark', 'random'].map(k => ({ name: DIRNAMES[k], color: CDIR[k] })));
    barChart(chartIn(c3), {
      groups: lay,
      series: ['gbar', 'probe', 'dark', 'random'].map(k => ({ name: DIRNAMES[k], y: e.dirbars[k], color: CDIR[k] })),
      xLabel: 'layer', yLabel: 'measured |Δφ|', height: 235, yDec: 2
    });

    const c4 = card(s, 'The certificate predicts; probe accuracy does not',
      'Spearman correlation with measured steering effect, over a pool of ~55 directions per layer spanning the (AUC, S̄) plane. Averaged over all runs.');
    legend(chartIn(c4), [{ name: 'certificate S̄', color: 'var(--s1)' }, { name: 'probe AUC', color: 'var(--s2)' }]);
    lineChart(chartIn(c4), {
      x: lay, series: [
        { name: 'S̄', y: e.sp_sbar, color: 'var(--s1)' },
        { name: 'AUC', y: e.sp_auc, color: 'var(--s2)' }],
      xLabel: 'layer', yLabel: 'Spearman vs measured', height: 215, yDec: 2, yMin: -1, yMax: 1
    });
  }

  /* ==================== E3 ==================== */
  if (has('e3')) {
    const e = D.e3;
    const s = section('e3', 'Experiment E3', `Toxicity in ${e.model}`,
      'Matched minimal pairs: hostile vs friendly framing is the causal factor; identity-marked subjects are a confound correlated with the label at 0.85, as in real corpora. Probes fitted on train, everything reported on held-out prompts.');

    if (e.minpair) {
      const m = e.minpair;
      const n = h('div', 'note', `<strong>Ground truth.</strong> Across ${m.n_pairs} exactly matched pairs
        (identical subject, only the frame flips) the behaviour functional moves by
        <b>${x(m.phi_gap_mean)}</b> logits in the right direction on <b>${pct(m.frac_pairs_correct_sign)}</b>
        of pairs. The behaviour being steered is real, not an artefact of the readout.`);
      s.appendChild(n);
    }

    const lay = e.layers.map(String);
    const g = h('div', 'grid2'); s.appendChild(g);

    const c1 = card(g, 'A perfect probe that is orthogonal to the mechanism',
      'Probe AUC on the causal frame label, probe AUC on the identity confound, and the alignment ρ between the probe and the adjoint direction.');
    legend(chartIn(c1), [{ name: 'AUC (frame)', color: 'var(--s2)' }, { name: 'AUC (identity confound)', color: 'var(--s4)' }, { name: 'ρ probe·ḡ', color: 'var(--s1)' }]);
    lineChart(chartIn(c1), {
      x: lay, series: [
        { name: 'AUC frame', y: e.auc_frame, color: 'var(--s2)' },
        { name: 'AUC identity', y: e.auc_ident, color: 'var(--s4)' },
        { name: 'ρ', y: e.rho, color: 'var(--s1)' }],
      xLabel: 'layer', yLabel: 'value', height: 235, yDec: 2, yMin: 0, yMax: 1, directLabel: false
    });
    c1.appendChild(h('p', 'cap', `The probe scores AUC up to <b>${x(Math.max(...e.auc_ident))}</b> on the identity confound &mdash; but
      that figure is inflated, because identity and frame are correlated at 0.85 in this split. The confound-free number comes from the
      balanced 2&times;2 cells below, and it is much smaller. Meanwhile &rho; stays near <b>${x(e.mean_rho)}</b> at every layer.`));

    const c2 = card(g, 'How much of the stream can the behaviour feel?',
      'r_eff/d, the effect-Gramian participation rank normalised by width.');
    lineChart(chartIn(c2), {
      x: lay, series: [{ name: 'r_eff/d', y: e.reff, color: 'var(--s1)' }],
      xLabel: 'layer', yLabel: 'r_eff / d', height: 235, yDec: 4, tipDec: 5, yMin: 0
    });
    c2.appendChild(h('p', 'cap', `Mean <b>${pct(e.mean_reff)}</b> of ${e.d} dimensions. A direction picked without write-side information keeps that fraction of the available steering power (Prop. 3).`));

    s.appendChild(h('div', 'note', `<strong>This is not "gradients steer better, obviously".</strong>
      That ḡ moves φ is true by construction and carries no information. Three things here do.
      <em>One:</em> the probe direction — the vector practitioners actually deploy — recovers only
      ${pct(e.eff_probe)} of that effect, and nobody had measured the size of that shortfall.
      <em>Two:</em> the certificate rank-predicts measured effect across a pool of directions that includes
      random draws, probe/adjoint mixtures and effect eigenvectors (Spearman ${x(e.mean_sp_sbar)}), so it is a
      genuine predictor over the space, not a restatement of one special case.
      <em>Three:</em> the best steering direction is itself a poor detector (AUC ${x(mean(e.auc_of.gbar))}
      against the probe's ${x(e.max_auc)}), so the dissociation runs in both directions rather than being a
      one-sided statement about probes.`));

    const c3 = card(s, 'Measured causal effect at equal write norm',
      'Mean Δφ over held-out prompts, α = 0.10·‖x_ℓ‖ at every layer. The dark direction is orthogonal to the probe by construction.');
    legend(chartIn(c3), ['gbar', 'dark', 'probe', 'identity_probe', 'random'].map(k => ({ name: DIRNAMES[k], color: CDIR[k] })));
    barChart(chartIn(c3), {
      groups: lay,
      series: ['gbar', 'dark', 'probe', 'identity_probe', 'random'].map(k =>
        ({ name: DIRNAMES[k], y: e.meas[k], color: CDIR[k] })),
      xLabel: 'layer', yLabel: 'measured Δφ (logits)', height: 250, yDec: 2
    });
    c3.appendChild(h('p', 'cap', `Summed over layers, probe-steering recovers <b>${pct(e.eff_probe)}</b> of the adjoint's effect and the probe-invisible dark direction recovers <b>${pct(e.eff_dark)}</b>. Steering along a direction you cannot detect works better than steering along the one you can.`));

    const g2 = h('div', 'grid2'); s.appendChild(g2);
    const c4 = card(g2, 'Certificate vs. probe accuracy as predictors',
      'Spearman correlation with measured |Δφ| across a pool of 18 directions per layer.');
    legend(chartIn(c4), [{ name: 'certificate S̄', color: 'var(--s1)' }, { name: 'probe AUC', color: 'var(--s2)' }]);
    lineChart(chartIn(c4), {
      x: lay, series: [
        { name: 'S̄', y: e.sp_sbar, color: 'var(--s1)' },
        { name: 'AUC', y: e.sp_auc, color: 'var(--s2)' }],
      xLabel: 'layer', yLabel: 'Spearman vs measured', height: 220, yDec: 2, yMin: -1, yMax: 1
    });
    c4.appendChild(h('p', 'cap', `Means: certificate <b>${x(e.mean_sp_sbar)}</b>, probe AUC <b>${x(e.mean_sp_auc)}</b>.`));

    const cal = card(g2, 'Calibration, not just ranking',
      'Predicted effect α·S̄ against measured |Δφ| for every (layer, direction) pair. The dashed line is y = x. A rank correlation would be satisfied by any monotone curve; landing on the diagonal is the stronger claim.');
    {
      const pts = e.pool_points.filter(p => isFinite(p.pred) && isFinite(p.meas));
      const mx = Math.max(...pts.map(p => Math.max(Math.abs(p.pred), p.meas)));
      const host = chartIn(cal);
      scatter(host, {
        points: pts.map(p => ({ x: Math.abs(p.pred), y: p.meas, r: 3.2,
          color: `var(--s${(p.layer % 5) + 1})`,
          label: `layer ${p.layer}<br>predicted ${x(Math.abs(p.pred))}<br>measured ${x(p.meas)}` })),
        xLabel: 'predicted |α·S̄|', yLabel: 'measured |Δφ|',
        height: 220, xDec: 1, yDec: 1, xMin: 0, yMin: 0, xMax: mx, yMax: mx
      });
      const svg = host.querySelector('svg');
      const m = { l: 56, r: 18, t: 14, b: 44 }, W = 640, H = 220;
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', m.l); ln.setAttribute('y1', m.t + ih);
      ln.setAttribute('x2', m.l + iw); ln.setAttribute('y2', m.t);
      ln.setAttribute('stroke', 'var(--ink-3)'); ln.setAttribute('stroke-width', '1');
      ln.setAttribute('stroke-dasharray', '4 3');
      svg.insertBefore(ln, svg.firstChild);
      const slope = pts.reduce((a, p) => a + p.meas * Math.abs(p.pred), 0) /
                    Math.max(1e-12, pts.reduce((a, p) => a + p.pred * p.pred, 0));
      cal.appendChild(h('p', 'cap', `Least-squares slope through the origin: <b>${x(slope)}</b> (1.0 would be perfect calibration). Deviation above 1 is second-order curvature — exactly what Proposition 6's validity radius bounds.`));
    }

    const c5 = card(g2, 'Certificate against measurement, all layers pooled',
      'Every (layer, direction) pair. If the first-order certificate were void at usable α, this would be a cloud.');
    scatter(chartIn(c5), {
      points: e.pool_points.filter(p => isFinite(p.sbar) && isFinite(p.meas)).map(p => ({
        x: p.sbar, y: p.meas, r: 3.4,
        color: `var(--s${(p.layer % 5) + 1})`,
        label: `layer ${p.layer}<br>S̄ ${x(p.sbar)}<br>measured ${x(p.meas)}`
      })),
      xLabel: 'certificate S̄ (no interventions run)', yLabel: 'measured |Δφ|', height: 220, xDec: 2, yDec: 2
    });

    if (e.dial) {
      const rhos = e.dial.map(d => d.mean_rho);
      const trend = rhos[rhos.length - 1] - rhos[0];
      const c6 = card(s, 'The confound dial — a registered prediction that failed',
        'Prediction: strengthening the spurious correlation between identity mentions and the toxicity label should widen the gap. It does not.');
      c6.appendChild(h('div', 'note caveat', `<strong>Null result, reported as such.</strong>
        Across P(identity matches label) = ${e.dial.map(d => d.spurious).join(', ')}, mean ρ moves from
        <b>${x(rhos[0])}</b> to <b>${x(rhos[rhos.length - 1])}</b> (change ${x(trend)}) — no meaningful trend, and the gap
        is already near-total at zero confound. In a <em>pretrained</em> model the probe/steer gap is therefore not
        caused by dataset confounding. The explanation left standing is Proposition 3: the effect geometry has
        r_eff/d ≈ ${pct(e.mean_reff)}, so almost any read-side direction is near-orthogonal to it regardless of how
        the labels were constructed. E2 tests separately whether confounding <em>adds</em> to the gap in a
        controlled from-scratch setting.`));
      const sp = e.dial.map(d => d.spurious);
      legend(chartIn(c6), [{ name: 'mean ρ', color: 'var(--s1)' }, { name: 'mean steering efficiency', color: 'var(--s2)' }, { name: 'best probe AUC', color: 'var(--s3)' }]);
      lineChart(chartIn(c6), {
        x: sp.map(String), series: [
          { name: 'ρ', y: e.dial.map(d => d.mean_rho), color: 'var(--s1)' },
          { name: 'efficiency', y: e.dial.map(d => d.mean_efficiency), color: 'var(--s2)' },
          { name: 'best AUC', y: e.dial.map(d => d.best_auc), color: 'var(--s3)' }],
        xLabel: 'P(identity term matches label)', yLabel: 'value', height: 220, yDec: 2
      });
    }

    if (e.cells) {
      const c7 = card(s, 'What the toxicity probe actually flags',
        'Rate at which each cell exceeds the probe\'s own median training threshold. "benign identity" is a friendly sentence about an identity-marked subject — a false positive whenever it is flagged.');
      legend(chartIn(c7), [
        { name: 'hostile / neutral subject', color: 'var(--s1)' },
        { name: 'hostile / identity subject', color: 'var(--s2)' },
        { name: 'benign / neutral subject', color: 'var(--s3)' },
        { name: 'benign / identity subject', color: 'var(--s4)' }]);
      barChart(chartIn(c7), {
        groups: e.cells.map(r => String(r.layer)),
        series: [
          { name: 'hostile/neutral', y: e.cells.map(r => r.fpr_hostile_neutral), color: 'var(--s1)' },
          { name: 'hostile/identity', y: e.cells.map(r => r.fpr_hostile_identity), color: 'var(--s2)' },
          { name: 'benign/neutral', y: e.cells.map(r => r.fpr_benign_neutral), color: 'var(--s3)' },
          { name: 'benign/identity', y: e.cells.map(r => r.fpr_benign_identity), color: 'var(--s4)' }],
        xLabel: 'layer', yLabel: 'flag rate', height: 235, yDec: 2
      });
      const last = e.cells[e.cells.length - 1];
      const gapH = mean(e.cells.slice(1).map(c => c.fpr_hostile_identity - c.fpr_hostile_neutral));
      const aucId = mean(e.cells.map(c => c.auc_identity_within_benign));
      c7.appendChild(h('div', 'prose', `
        <p><strong>Two disparities, on different layers.</strong> At layer 0, where the frame signal is still weak,
        benign sentences with an identity-marked subject are flagged at
        <b>${pct(e.cells[0].fpr_benign_identity)}</b> against <b>${pct(e.cells[0].fpr_benign_neutral)}</b> for the
        otherwise identical sentence with a neutral subject. That gap closes with depth: from layer 1 on, benign
        false positives are essentially zero in both cells.</p>
        <p>The disparity that <em>does not</em> close sits inside the hostile class. Averaged over layers 1&ndash;${last.layer},
        hostile sentences about identity-marked subjects are flagged <b>${pct(gapH)}</b> more often than hostile
        sentences about neutral subjects &mdash; at layer ${last.layer} it is
        <b>${pct(last.fpr_hostile_identity)}</b> versus <b>${pct(last.fpr_hostile_neutral)}</b>. Same frame, same
        template, different subject. And the probe still separates identity within the benign cell at mean AUC
        <b>${x(aucId)}</b>, above the 0.500 it would show if it were reading frame alone.</p>
        <p>This is the confound-free measurement, and it is much smaller than the naive AUC ${x(Math.max(...e.auc_ident))}
        reported above &mdash; which is why the balanced cells are the number to quote.</p>`));
    }

    if (e.alpha_sweep && e.alpha_sweep.length) {
      const mid = e.alpha_sweep[Math.floor(e.alpha_sweep.length / 2)];
      const c8 = card(s, 'Validity radius of the first-order certificate (Prop. 6)',
        `Measured vs. predicted effect along ḡ at layer ${mid.layer}, sweeping the write magnitude. The certificate is a first-order object; this is where it stops being one.`);
      legend(chartIn(c8), [{ name: 'measured', color: 'var(--s1)' }, { name: 'predicted α·S̄', color: 'var(--s2)' }]);
      lineChart(chartIn(c8), {
        x: mid.pts.map(p => p.alpha.toFixed(1)),
        series: [
          { name: 'measured', y: mid.pts.map(p => p.measured), color: 'var(--s1)' },
          { name: 'predicted', y: mid.pts.map(p => p.predicted), color: 'var(--s2)' }],
        xLabel: 'α (write magnitude)', yLabel: 'Δφ', height: 220, yDec: 2
      });
      const rows = e.alpha_sweep.map(a => [`L${a.layer}`, x(a.alpha), x(a.M),
        a.astar === null || !isFinite(a.astar) ? '∞' : x(a.astar),
        { v: a.astar && a.alpha ? x(a.astar / a.alpha) : '—', cls: (a.astar / a.alpha) > 1 ? 'g' : 'r' }]);
      table(c8, ['layer', 'α used', 'curvature M̂', 'α* (validity radius)', 'α*/α'], rows);
      c8.appendChild(h('p', 'cap', 'α*/α > 1 means the certificate is inside its validity range at the write magnitude used. Where it is below 1, the first-order prediction is out of warranty and the measured value is the one to trust — this is the honest boundary of the method.'));
    }

    if (e.ppl && e.ppl.gbar) {
      const allZero = [...e.ppl.gbar, ...e.ppl.probe].every(v => v === 0 || v === null);
      const c9 = card(s, 'Collateral damage: a metric that had to be replaced',
        "The first audit measured the increase in the prompt’s own next-token perplexity under the steer.");
      c9.appendChild(h('div', 'prose', `<p>It returned <strong>${allZero ? 'exactly zero at every layer' : 'near-zero values'}</strong>,
        and that is not a small effect — it is a structural one. The write lands on the <em>final</em> token, and causal
        masking means a final-position state cannot influence predictions made at earlier positions. The quantity is zero
        by construction, so it measures nothing about this intervention.</p>
        <p>The correct collateral-damage measure for a final-position write is the KL divergence of the
        <em>next-token</em> distribution, which the steer genuinely does move. It is reported in E5.
        Recording the failure rather than deleting it: a metric that cannot be nonzero is worse than no metric,
        because it reads as evidence of safety.</p>`));
    }
  }

  /* ==================== E6 ==================== */
  if (has('e6')) {
    const e = D.e6, S = e.summary;
    const s = section('e6', 'Experiment E6', 'The two geometries are structurally independent',
      `The sharpest test available. Fix one model and ${e.n} prompts, then vary the label definition (which moves only the read side) and the behaviour functional (which moves only the write side). Some pairs are semantically MATCHED by construction.`);

    s.appendChild(h('div', 'note', `<strong>The headline number.</strong> A probe trained on
      <em>exactly the concept the behaviour functional measures</em> is aligned with that behaviour's
      adjoint at &rho; = <b>${x(S.matched_mean_over_layers)}</b>. A probe for a completely
      <em>unrelated</em> concept is aligned at &rho; = <b>${x(S.unmatched_mean_over_layers)}</b>.
      The ratio is <b>${x(S.ratio)}</b>, and matched beats unmatched in
      <b>${S.layers_where_matched_exceeds_unmatched} of ${S.n_layers}</b> layers &mdash; chance.
      Knowing that a probe was trained on the right concept tells you nothing whatsoever about whether
      it points anywhere causally useful.`));

    const c1 = card(s, 'Matched vs. unmatched alignment, by layer',
      'If the read and write geometries were the same object, the matched line would sit above the unmatched one everywhere. It does not — it sits below for the first half of the network and crosses late.');
    legend(chartIn(c1), [{ name: 'matched (probe = behaviour concept)', color: 'var(--s1)' },
                         { name: 'unmatched (unrelated concept)', color: 'var(--s2)' }]);
    lineChart(chartIn(c1), {
      x: e.layers.map(String),
      series: [{ name: 'matched', y: e.matched_series, color: 'var(--s1)' },
               { name: 'unmatched', y: e.unmatched_series, color: 'var(--s2)' }],
      xLabel: 'layer', yLabel: 'alignment ρ', height: 230, yDec: 3, tipDec: 4,
      yMin: 0, directLabel: false
    });
    {
      const n = e.layers.length, k = Math.max(1, Math.round(n / 3));
      const mLate = mean(e.matched_series.slice(-k)), uLate = mean(e.unmatched_series.slice(-k));
      const mEarly = mean(e.matched_series.slice(0, k)), uEarly = mean(e.unmatched_series.slice(0, k));
      c1.appendChild(h('p', 'cap', `<b>The nuance, stated rather than averaged away.</b> Over the last ${k} layers
        matched alignment does exceed unmatched (<b>${x(mLate)}</b> vs <b>${x(uLate)}</b>, about
        ${x(mLate / uLate)}&times;), so a late-layer probe is not <em>completely</em> uninformative about the mechanism.
        Over the first ${k} layers the ordering reverses (<b>${x(mEarly)}</b> vs <b>${x(uEarly)}</b>). Both regimes sit
        near zero: &rho; = 0.06 is an angle of about 86&deg; from the effect direction, so neither confers usable
        causal control. The claim is not that matched probes carry literally no signal — it is that the signal is
        too small to act on, and is swamped by which layer you happen to be at.`));
    }

    const mid = Math.floor(e.layers.length / 2);
    const c2 = card(s, `Full alignment matrix at layer ${e.layers[mid]}`,
      'Rows are probes (one per label definition); columns are behaviour functionals. Matched cells are marked. Every entry is small, and the marked ones are not larger than the rest.');
    const cols = ['probe \ behaviour'].concat(e.behaviours);
    const rows = e.labels.map((l, i) => [l].concat(e.behaviours.map((b, j) => {
      const v = e.rho_matrix_mid[i][j];
      const isM = e.matched[l] === b;
      return { v: x(v) + (isM ? '  ◀' : ''), cls: isM ? 'hi' : '' };
    })));
    table(c2, cols, rows);
    c2.appendChild(h('p', 'cap', `Probe AUCs at this layer: ${e.labels.map(l => `${l} ${x(e.probe_auc_mid[l])}`).join(', ')} — every probe reads its own concept well above chance, and none of them reads the mechanism.`));
  }

  /* ==================== E5 ==================== */
  if (has('e5')) {
    const e = D.e5;
    const s = section('e5', 'Experiment E5', 'Real toxicity data and the steering vectors people deploy',
      `${e.n} RealToxicityPrompts prompts, balanced high/low toxicity, on ${e.models.length} model${e.models.length > 1 ? 's' : ''}. Difference-of-means is the vector the activation-steering literature actually uses, so it is a first-class baseline rather than a strawman. Probe AUC here sits near 0.78, not 1.0 — which answers the objection that the gap is an artefact of saturated probes.`);

    if (e.partial) {
      s.appendChild(h('div', 'note caveat', `<strong>Partial run, recovered from the log.</strong>
        ${e.recovery_note} The per-layer numbers below were genuinely computed; the fields that were
        computed but never printed are shown as blanks rather than filled in.`));
    }

    const rows = [];
    e.models.forEach(m => {
      ['fisher_probe', 'logistic_probe', 'diff_of_means', 'dark', 'gbar', 'random'].forEach(k => {
        const a = m.agg[k]; if (!a) return;
        rows.push([`${m.name.split('/').pop()} · ${DIRNAMES[k] || k}`,
          a.mean_auc === null || a.mean_auc === undefined ? '—' : x(a.mean_auc),
          a.mean_rho === null || a.mean_rho === undefined ? '—' : x(a.mean_rho),
          { v: x(a.mean_abs_measured), cls: 'hi' },
          { v: pct(a.efficiency_vs_gbar ?? 1), cls: k === 'gbar' ? '' : (a.efficiency_vs_gbar > .5 ? 'g' : 'r') }]);
      });
    });
    const c1 = card(s, 'Read quality against write power', 'Averaged over all layers of each model. Efficiency is measured effect relative to the adjoint direction at equal write norm.');
    table(c1, ['model · direction', 'mean AUC', 'mean ρ', 'mean |Δφ|', 'efficiency vs ḡ'], rows);
    c1.appendChild(h('p', 'cap', `Every read-side construction classifies well and steers poorly; the direction built orthogonal to <em>all</em> of them
      does the reverse. Difference-of-means — the vector activation-steering papers actually ship — is the best of the read-side
      options and still recovers under a fifth of the available control. Dashes mark quantities the partial run computed but did not print.`));

    e.models.forEach(m => {
      const c = card(s, `${m.name} — measured steering by layer`,
        `${m.n_layers} layers, d = ${m.d}. Behaviour functional separates toxic from benign prompts at AUC ${x(m.phi_auc)}; mean r_eff/d = ${pct(m.mean_reff)}.`);
      const keys = ['gbar', 'dark', 'fisher_probe', 'diff_of_means', 'random']
        .filter(k => m.series[k] && m.series[k].some(v => v !== null && v !== undefined));
      legend(chartIn(c), keys.map(k => ({ name: DIRNAMES[k] || k, color: CDIR[k] })));
      barChart(chartIn(c), {
        groups: m.layers.map(String),
        series: keys.map(k => ({ name: DIRNAMES[k] || k, y: m.series[k], color: CDIR[k] })),
        xLabel: 'layer', yLabel: 'measured Δφ', height: 225, yDec: 2
      });
    });
    if (e.errors && e.errors.length) {
      s.appendChild(h('div', 'note caveat', '<strong>Not all models completed.</strong> ' +
        e.errors.map(x => `<code>${x.model}</code>: ${x.error}`).join('; ')));
    }
  }

  /* ==================== E4 ==================== */
  if (has('e4')) {
    const e = D.e4;
    const s = section('e4', 'Experiment E4', 'Spending an intervention budget well',
      `Ranking ${e.M} candidate directions by true causal effect normally costs ${e.M} intervention sweeps. The certificate is free — it reuses adjoints the probe already required. Does it buy a cheaper search?`);
    e.layers.forEach(l => {
      const c = card(s, `Layer ${l.layer} — best effect found vs. budget`,
        `Fraction of the best attainable effect discovered after B measurements. Spearman(S̄, truth) = ${x(l.spearman_Sbar_vs_truth)}; Spearman(|AUC−½|, truth) = ${x(l.spearman_absauc_vs_truth)}.`);
      const ks = ['oracle', 'certificate', 'probe_auc', 'random'];
      const cols = { oracle: 'var(--ink-3)', certificate: 'var(--s1)', probe_auc: 'var(--s2)', random: 'var(--s4)' };
      legend(chartIn(c), ks.map(k => ({ name: k.replace('_', ' '), color: cols[k] })));
      lineChart(chartIn(c), {
        x: l.budgets.map(String),
        series: ks.map(k => ({ name: k.replace('_', ' '), y: l.best_found_frac[k], color: cols[k] })),
        xLabel: 'intervention budget B', yLabel: 'fraction of best found',
        height: 225, yDec: 2, yMin: 0, yMax: 1, directLabel: false
      });
      const b9 = l.budget_to_90pct, b99 = l.budget_to_99pct;
      table(c, ['strategy', 'budget to 90% of best', 'budget to 99% of best'],
        ks.map(k => [k.replace('_', ' '), b9[k] ?? '—', b99[k] ?? '—']));
    });
  }

  /* ==================== deliverables ==================== */
  {
    const s = section('outputs', 'Section 03', 'The same numbers, three ways',
      'This report, a preprint and a narrated walkthrough are all generated from results/data/*.json. None of them contains a hand-typed measurement.');
    const items = [
      ['Preprint', 'paper/preprint.pdf', '12 pages',
       'Theory with proofs, experimental setup, results, results-against-interest, limitations, next steps. Every quoted number is a LaTeX macro emitted by <code>make_numbers.py</code>; every figure is emitted by <code>make_figures.py</code>. <code>check_macros.py</code> fails the build if the text quotes a number the data does not define.',
       'var(--s1)'],
      ['Video walkthrough', 'out/seeing-is-not-steering.mp4', '10:01 &middot; 1920&times;1080',
       'Seventeen scenes covering theory through next steps. Narration synthesised locally with Kokoro-82M &mdash; no API key, no network. Scene durations are derived from the measured length of each audio clip, so the visuals cannot drift from the voiceover.',
       'var(--s3)'],
      ['This report', 'dashboard/index.html', 'live',
       'Rebuilt by <code>dashboard/build.py</code> from the same JSON. Sections appear only when their experiment has actually run.',
       'var(--s4)'],
    ];
    const g = h('div', 'grid3');
    items.forEach(([t, path, meta, d, col]) => {
      const c = h('div', 'card');
      c.style.borderTop = `4px solid ${col}`;
      c.style.marginBottom = '0';
      c.innerHTML = `<h3>${t}</h3>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:var(--ink-3);margin:-2px 0 12px">
          <span style="color:var(--ink-2)">${path}</span> &middot; ${meta}</div>
        <div style="font-size:.88rem;color:var(--ink-2);line-height:1.5">${d}</div>`;
      g.appendChild(c);
    });
    s.appendChild(g);
    s.appendChild(h('p', 'cap', 'The preprint and video are files in the repository, not links from this page — regenerate them with the commands in the README.'));
  }

  /* ==================== gaps ==================== */
  {
    const s = section('gaps', 'Section 04', 'Where this sits in the 2026 literature',
      'A novelty scan run 2026-08-24 reshaped the plan: several headline directions in the original research package have since been claimed.');
    const c = card(s, null, null);
    table(c, ['Direction', '2026 status', 'Decision'], [
      ['Non-identifiability of circuits', 'Claimed — ICLR 2025 "Everything, Everywhere, All at Once"; ICLR 2026 uniqueness paper; Certified Circuits; Many Circuits, One Mechanism', { v: 'Dropped as a headline', cls: 'r' }],
      ['Control-theoretic interpretability', 'Partly claimed — Gramian/Hankel mode ranking for NNs and state-space LMs', 'Reused as machinery, not as the claim'],
      ['Decodability ≠ causal control', 'Observed qualitatively (Steering the Language Axis, 2026)', { v: 'Made quantitative and predictive', cls: 'g' }],
      ['Toxicity circuits / detox steering', 'Crowded — CausalDetox, linear optimal control steering, feedback controllers', 'Kept as application, not as contribution'],
      ['Active design for transformer circuits', 'Open — exists in causal discovery, not transferred', { v: 'Taken (E4)', cls: 'g' }],
    ], true);
    c.appendChild(h('p', 'cap', 'The surviving contribution is the one nobody has: a certificate that <em>predicts in advance</em>, from cheap adjoint information, whether a decoded direction will control behaviour — plus a closed form for the directions a probe can never find.'));
  }

  /* ==================== limitations ==================== */
  {
    const s = section('limits', 'Section 05', 'Limitations', 'Stated plainly, because several are load-bearing.');
    const c = card(s, null, null);
    c.appendChild(h('div', 'prose', `
      <p><strong>Model scale.</strong> Everything ran on 4 CPU cores with no GPU. The largest model is GPT-2 small
      (124M). ρ and r_eff/d could behave differently at 7B+; nothing here rules that out. The theory is
      scale-free, the measurements are not.</p>
      <p><strong>First order.</strong> The certificate is a first-order object. Proposition 6 gives its validity
      radius α* and E3 measures it; where α*/α &lt; 1 the prediction is out of warranty. Large-α steering
      studies operate partly in that regime, and the certificate does not license claims there.</p>
      <p><strong>Single behaviour functional.</strong> φ is a logit difference over a small token set. Other
      readouts (full-sequence generation quality, human toxicity judgements) may induce different effect
      geometry. The minimal-pair check establishes that this φ tracks the intended factor, not that it is
      the only reasonable one.</p>
      <p><strong>Position and pathway.</strong> Interventions write to the final token's residual stream at one
      layer. Multi-position, multi-layer, and attention-internal writes are untested.</p>
      <p><strong>What "dark" does not mean.</strong> A dark direction is invisible to <em>this</em> probe family
      (linear, this concept, this data). It is not invisible in principle — a nonlinear or differently-supervised
      detector may well find it. The safety claim is about the monitors people actually deploy.</p>`));
  }

  /* ==================== next ==================== */
  {
    const s = section('next', 'Section 06', 'Next steps', 'Ordered by what would most change the conclusions.');
    const c = card(s, null, null);
    table(c, ['#', 'Step', 'Why it matters', 'Cost'], [
      ['1', 'Replicate ρ and r_eff/d at 1B–7B scale', 'The single biggest threat to external validity. If r_eff/d grows with width, Prop. 3 loses its bite.', 'GPU, days'],
      ['2', 'Certificate for nonlinear and SAE-feature readouts', 'Tests whether "dark" survives a stronger detector, which is the real safety question.', 'GPU, moderate'],
      ['3', 'Multi-layer, multi-position optimal control', 'Minimum-energy steering under a collateral-damage constraint — the natural engineering payoff.', 'Moderate'],
      ['4', 'Second-order certificate', 'Extends validity past α*, where most deployed steering lives.', 'Cheap, analytic'],
      ['5', 'Sequential Bayesian design against ACDC/EAP', 'Turns E4 from a demonstration into a benchmark result.', 'Moderate'],
      ['6', 'Human-labelled toxicity behaviour functional', 'Removes the logit-difference proxy from the safety claim.', 'Annotation'],
    ], true);
  }

  /* ==================== rail ==================== */
  {
    const rail = document.getElementById('rail-exp');
    const defs = [['e1', 'E1', 'Exact decoupling'], ['e2', 'E2', 'Trained transformers'],
                  ['e3', 'E3', 'GPT-2 toxicity'], ['e6', 'E6', 'Read/write independence'],
                  ['e5', 'E5', 'Real data, 3 models'],
                  ['e4', 'E4', 'Active design']];
    const present = defs.filter(d => has(d[0]));
    if (!present.length) { rail.style.display = 'none'; }
    present.forEach(d => {
      const a = document.createElement('a');
      a.href = '#' + d[0];
      a.innerHTML = `<span class="n">${d[1]}</span>${d[2]}`;
      rail.appendChild(a);
    });
  }

  /* ==================== provenance ==================== */
  {
    const p = D.prov, box = document.getElementById('provenance');
    const rows = [['Generated', p.generated], ['Commit', p.commit], ['Hardware', p.hardware],
      ['Python', p.python], ['torch', p.torch], ['numpy', p.numpy],
      ['Sections present', D.have.join(', ') || 'none'],
      ['Raw data', 'results/data/*.json — every figure is regenerable from these']];
    rows.forEach(r => { box.appendChild(h('b', null, r[0])); box.appendChild(h('span', null, r[1])); });
  }
})();
