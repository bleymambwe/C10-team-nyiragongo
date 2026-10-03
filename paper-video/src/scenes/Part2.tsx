import React from "react";
import { Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Frame, Reveal } from "../components/Frame";
import { BarsV, CountUp, Legend, LineChartV, StatCard } from "../components/Charts";
import { C, F } from "../theme";
import facts from "../facts.json";

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

/* ------------------------------------------------------------------ 06 */
export const S06Leakage: React.FC = () => {
  const rows = [200, 500, 1000, 4000, 16000, 64000];
  const theory = rows.map((n) => 1 / Math.sqrt(n));
  const measured = [0.0715, 0.0327, 0.0284, 0.0181, 0.0048, 0.0031];
  return (
    <Frame eyebrow="Proposition 7 · finite-sample leakage" title="Estimated probes manufacture causal effect">
      <Legend
        items={[
          { name: "measured ρ̂", color: C.s2 },
          { name: "theory: N^(−1/2)", color: C.ink3 },
        ]}
      />
      <LineChartV
        x={rows}
        series={[
          { name: "ρ̂", y: measured, color: C.s2 },
          { name: "N^−½", y: theory, color: C.ink3 },
        ]}
        yLabel="spurious alignment"
        xLabel="training examples N"
        height={300}
        dec={3}
        yMin={0}
        delay={1.0}
      />
      <div style={{ display: "flex", gap: 30, marginTop: 22 }}>
        <Reveal delay={3.6} style={{ flex: 1 }}>
          <StatCard label="Measured slope" accent={C.s3} sub="against a predicted 1.0">
            <CountUp to={facts.p7.loglog_slope} dec={3} delay={3.7} color={C.s3} size={96} />
          </StatCard>
        </Reveal>
        <Reveal delay={4.6} style={{ flex: 1 }}>
          <StatCard label="R²" accent={C.s3} sub="dimension-free law wins">
            <CountUp to={facts.p7.r2} dec={3} delay={4.7} color={C.s3} size={96} />
          </StatCard>
        </Reveal>
        <Reveal delay={5.6} style={{ flex: 2 }}>
          <StatCard label="What is absent" accent={C.warn} sub="">
            <div style={{ fontSize: 36, color: C.ink, lineHeight: 1.3, marginTop: 4 }}>
              No dependence on model width. A wider residual stream does{" "}
              <em style={{ color: C.warn }}>not</em> dilute the artefact.
            </div>
          </StatCard>
        </Reveal>
      </div>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 07 */
export const S07LowRank: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const live = facts.e3.mean_reff; // ~0.0018
  const DOTS = 24 * 12;
  const nLive = Math.max(1, Math.round(DOTS * live * 40)); // visually scaled, labelled below
  const grow = interpolate(frame, [1.2 * fps, 3.2 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <Frame eyebrow="Proposition 3 · effective rank" title="Almost none of the stream is live">
      <div style={{ display: "flex", gap: 54, alignItems: "center" }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(24, 1fr)",
            gap: 9,
            flex: 1,
            padding: 26,
            background: C.surface,
            border: `2px solid ${C.rule}`,
            borderRadius: 16,
          }}
        >
          {Array.from({ length: DOTS }).map((_, i) => {
            const isLive = i % Math.floor(DOTS / nLive) === 0 && i / DOTS < grow * 1.05;
            return (
              <div
                key={i}
                style={{
                  width: 20,
                  height: 20,
                  borderRadius: 4,
                  background: isLive ? C.s1 : C.surfaceAlt,
                  border: `1px solid ${isLive ? C.s1 : C.rule}`,
                  opacity: interpolate(frame, [0.4 * fps, 1.4 * fps], [0, 1], {
                    extrapolateLeft: "clamp",
                    extrapolateRight: "clamp",
                  }),
                }}
              />
            );
          })}
        </div>

        <div style={{ width: 620 }}>
          <Reveal delay={3.4}>
            <div style={{ fontFamily: F.mono, fontSize: 26, letterSpacing: "0.14em", color: C.ink3 }}>
              LIVE FRACTION r_eff / d
            </div>
            <CountUp to={live * 100} dec={2} suffix="%" delay={3.5} color={C.s1} size={128} />
            <div style={{ fontSize: 34, color: C.ink2, marginTop: 16, lineHeight: 1.4 }}>
              of a 768-dimensional residual stream carries any first-order behavioural effect.
            </div>
          </Reveal>
          <Reveal delay={5.6} style={{ marginTop: 34 }}>
            <div
              style={{
                background: C.surface,
                border: `2px solid ${C.rule}`,
                borderLeft: `6px solid ${C.warn}`,
                borderRadius: 14,
                padding: "26px 30px",
                fontSize: 34,
                color: C.ink,
                lineHeight: 1.4,
              }}
            >
              A direction chosen without write-side information keeps exactly this fraction of
              the available steering power.
            </div>
          </Reveal>
        </div>
      </div>
      <div style={{ marginTop: 20, fontSize: 28, color: C.ink3, fontFamily: F.mono }}>
        grid is illustrative &mdash; live cells shown at 40× their true density to remain visible
      </div>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 08 */
export const S08Gpt2: React.FC = () => {
  const e = facts.e3;
  return (
    <Frame eyebrow="Experiment E3 · GPT-2" title="A perfect probe, orthogonal to the mechanism">
      <Legend
        items={[
          { name: "probe AUC (held out)", color: C.s2 },
          { name: "alignment ρ with the mechanism", color: C.s1 },
        ]}
      />
      <LineChartV
        x={e.layers}
        series={[
          { name: "AUC", y: e.auc, color: C.s2 },
          { name: "ρ", y: e.rho, color: C.s1 },
        ]}
        yLabel="value"
        xLabel="layer"
        height={270}
        yMin={0}
        yMax={1}
        dec={2}
        delay={0.6}
      />
      <div style={{ display: "flex", gap: 30, marginTop: 24 }}>
        <Reveal delay={3.4} style={{ flex: 1 }}>
          <StatCard label="Probe steering" accent={C.s2} sub="of available control, at equal norm">
            <CountUp to={e.eff_probe * 100} dec={1} suffix="%" delay={3.5} color={C.s2} size={100} />
          </StatCard>
        </Reveal>
        <Reveal delay={4.6} style={{ flex: 1 }}>
          <StatCard
            label="Probe-orthogonal steering"
            accent={C.s3}
            sub="forced orthogonal, loses nothing"
          >
            <CountUp to={e.eff_dark * 100} dec={1} suffix="%" delay={4.7} color={C.s3} size={100} />
          </StatCard>
        </Reveal>
        <Reveal delay={5.8} style={{ flex: 2 }}>
          <StatCard label="Reading" accent={C.accent} sub="">
            <div style={{ fontSize: 36, color: C.ink, lineHeight: 1.32, marginTop: 4 }}>
              The probe is not a weak handle on the mechanism. It is very nearly{" "}
              <em style={{ color: C.accent }}>disjoint</em> from it.
            </div>
          </StatCard>
        </Reveal>
      </div>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 09 */
export const S09NotCircular: React.FC = () => {
  const e = facts.e3;
  const items = [
    [
      "The shortfall had never been measured",
      `The probe direction is what practitioners deploy. It recovers ${pct(e.eff_probe)} of available control.`,
      C.s2,
    ],
    [
      "The certificate predicts over the whole space",
      `Spearman ${e.mean_sp_s.toFixed(2)} against measured effect across random draws, mixtures and eigenvectors — while probe AUC sits at ${e.mean_sp_a.toFixed(2)}.`,
      C.s1,
    ],
    [
      "The dissociation runs both ways",
      "The best steering direction is itself a poor detector, so this is not a one-sided statement about probes.",
      C.s3,
    ],
  ];
  return (
    <Frame
      eyebrow="The objection"
      title={'"Of course the gradient steers better than a probe."'}
    >
      <Reveal delay={0.4}>
        <div style={{ fontSize: 38, color: C.ink2, marginBottom: 22, maxWidth: 1520, lineHeight: 1.35 }}>
          True by construction, and it carries no information. Three things here do.
        </div>
      </Reveal>
      {items.map(([t, d, col], i) => (
        <Reveal key={t as string} delay={1.6 + i * 2.4} style={{ marginBottom: 18 }}>
          <div
            style={{
              display: "flex",
              gap: 30,
              background: C.surface,
              border: `2px solid ${C.rule}`,
              borderLeft: `6px solid ${col as string}`,
              borderRadius: 14,
              padding: "22px 32px",
            }}
          >
            <div
              style={{
                fontFamily: F.mono,
                fontSize: 44,
                color: col as string,
                fontWeight: 600,
                minWidth: 60,
              }}
            >
              {i + 1}
            </div>
            <div>
              <div style={{ fontSize: 41, color: C.ink, fontWeight: 600 }}>{t}</div>
              <div style={{ fontSize: 32, color: C.ink2, marginTop: 8, lineHeight: 1.38 }}>{d}</div>
            </div>
          </div>
        </Reveal>
      ))}
    </Frame>
  );
};

/* ------------------------------------------------------------------ 10 */
export const S10Independence: React.FC = () => {
  const e = facts.e6;
  return (
    <Frame eyebrow="Experiment E6 · the sharpest test" title="The two geometries are structurally independent">
      <Legend
        items={[
          { name: "matched — probe trained on the behaviour's own concept", color: C.s1 },
          { name: "unmatched — an unrelated concept", color: C.s2 },
        ]}
      />
      <LineChartV
        x={Array.from({ length: e.matched_series.length }, (_, i) => i)}
        series={[
          { name: "matched", y: e.matched_series, color: C.s1 },
          { name: "unmatched", y: e.unmatched_series, color: C.s2 },
        ]}
        yLabel="alignment ρ"
        xLabel="layer"
        height={330}
        yMin={0}
        dec={3}
        delay={0.8}
      />
      <div style={{ display: "flex", gap: 30, marginTop: 24 }}>
        <Reveal delay={3.6} style={{ flex: 1 }}>
          <StatCard label="Matched" accent={C.s1} sub="probe = the behaviour's concept">
            <CountUp to={e.m} dec={3} delay={3.7} color={C.s1} size={110} />
          </StatCard>
        </Reveal>
        <Reveal delay={4.5} style={{ flex: 1 }}>
          <StatCard label="Unmatched" accent={C.s2} sub="a completely unrelated concept">
            <CountUp to={e.u} dec={3} delay={4.6} color={C.s2} size={110} />
          </StatCard>
        </Reveal>
        <Reveal delay={5.4} style={{ flex: 2 }}>
          <StatCard
            label={`Ratio ${e.ratio.toFixed(2)} · matched wins ${e.wins}/${e.n} layers`}
            accent={C.warn}
            sub=""
          >
            <div style={{ fontSize: 40, color: C.ink, lineHeight: 1.32, marginTop: 4 }}>
              Training a probe on the <em style={{ color: C.warn }}>right concept</em> tells you
              nothing about whether it points anywhere causally useful.
            </div>
          </StatCard>
        </Reveal>
      </div>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 11 */
export const S11RealData: React.FC = () => {
  const ms = facts.e5;
  const names = ["Fisher probe", "Diff-of-means (CAA)", "Orthogonal to both"];
  const keys = ["fisher_probe", "diff_of_means", "dark"] as const;
  const cols = [C.s2, C.s5, C.s3];
  return (
    <Frame eyebrow="Experiment E5 · RealToxicityPrompts" title="Real data, and the vectors people actually ship">
      <Legend items={names.map((n, i) => ({ name: n, color: cols[i] }))} />
      <BarsV
        groups={ms.map((m) => m.name.split("/").pop() as string)}
        series={keys.map((k, i) => ({
          name: names[i],
          color: cols[i],
          y: ms.map((m) => (m.agg as never as Record<string, { efficiency_vs_gbar: number }>)[k].efficiency_vs_gbar * 100),
        }))}
        yLabel="% of available control"
        xLabel="model"
        height={330}
        dec={0}
        delay={0.7}
      />
      <div style={{ display: "flex", gap: 30, marginTop: 26 }}>
        <Reveal delay={3.0} style={{ flex: 1 }}>
          <StatCard label="Probe accuracy here" accent={C.accent} sub="not saturated at 1.0 — so the gap is not an artefact of a maxed-out probe">
            <CountUp
              to={ms[0].agg.fisher_probe.mean_auc}
              dec={3}
              delay={3.1}
              color={C.accent}
              size={112}
            />
          </StatCard>
        </Reveal>
        <Reveal delay={4.2} style={{ flex: 2 }}>
          <StatCard label="The pattern replicates" accent={C.s3} sub="">
            <div style={{ fontSize: 38, color: C.ink, lineHeight: 1.34, marginTop: 4 }}>
              Every read-side construction classifies well and steers poorly. The direction built
              orthogonal to all of them does the reverse &mdash; on both models.
            </div>
          </StatCard>
        </Reveal>
      </div>
    </Frame>
  );
};
