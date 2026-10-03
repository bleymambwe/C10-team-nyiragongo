import React from "react";
import { Frame, Reveal } from "../components/Frame";
import { CountUp, Legend, LineChartV, StatCard } from "../components/Charts";
import { C } from "../theme";
import facts from "../facts.json";

/* ------------------------------------------------------------------ 12
   E2 -- transformers trained from scratch. The gap is produced by gradient
   descent, and a training-time confound does not widen it.                */
export const S12Trained: React.FC = () => {
  const e = facts.e2;
  const levels = e.levels.map((v) => v.toFixed(2));
  return (
    <Frame
      eyebrow="Experiment E2 &middot; trained from scratch"
      title="Gradient descent produces the gap on its own"
    >
      <Legend
        items={[
          { name: "alignment ρ", color: C.s1 },
          { name: "probe steering, as a fraction of available control", color: C.s2 },
        ]}
      />
      <LineChartV
        x={levels}
        series={[
          { name: "ρ", y: e.rho_by_level, color: C.s1 },
          {
            name: "probe",
            y: e.eff_by_level.map((v) => v / 100),
            color: C.s2,
          },
        ]}
        yLabel="value"
        xLabel="P(confound matches label)"
        height={300}
        yMin={0}
        yMax={0.25}
        dec={2}
        delay={0.8}
      />
      <div style={{ display: "flex", gap: 28, marginTop: 22 }}>
        <Reveal delay={3.2} style={{ flex: 1 }}>
          <StatCard
            label="At zero confounding"
            accent={C.s1}
            sub={`ρ = ${e.rho_by_level[0].toFixed(3)} with no confound present at all`}
          >
            <CountUp
              to={e.eff_by_level[0]}
              dec={1}
              suffix="%"
              delay={3.3}
              color={C.s1}
              size={100}
            />
          </StatCard>
        </Reveal>
        <Reveal delay={4.2} style={{ flex: 1 }}>
          <StatCard
            label="Models trained"
            accent={C.s3}
            sub={`${e.n_seeds} seeds × ${e.levels.length} confound levels, test accuracy ${e.acc.toFixed(3)}`}
          >
            <CountUp to={e.n_runs} dec={0} delay={4.3} color={C.s3} size={100} />
          </StatCard>
        </Reveal>
        <Reveal delay={5.2} style={{ flex: 2 }}>
          <StatCard label="Reading" accent={C.accent} sub="">
            <div style={{ fontSize: 36, color: C.ink, lineHeight: 1.32, marginTop: 4 }}>
              Not inherited from pretraining, and not specific to GPT&#8209;2. Train a
              transformer yourself and the gap is already there.
            </div>
          </StatCard>
        </Reveal>
      </div>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 13
   E4 -- the certificate as a search prior.                               */
export const S13Budget: React.FC = () => {
  const e = facts.e4;
  const mid = Math.floor(e.layers.length / 2);
  const cols: Record<string, string> = {
    oracle: C.ink3,
    certificate: C.s1,
    probe_auc: C.s2,
    random: C.s4,
  };
  const keys = ["oracle", "certificate", "probe_auc", "random"];
  return (
    <Frame
      eyebrow="Experiment E4 &middot; active design"
      title="Spending an intervention budget well"
    >
      <Legend
        items={keys.map((k) => ({ name: k.replace("_", " "), color: cols[k] }))}
      />
      <LineChartV
        x={e.budgets.map(String)}
        series={keys.map((k) => ({
          name: k.replace("_", " "),
          y: (e.curves as unknown as Record<string, number[][]>)[k][mid],
          color: cols[k],
        }))}
        yLabel="fraction found"
        xLabel={`intervention budget, of ${e.M} candidates`}
        height={300}
        yMin={0}
        yMax={1}
        dec={2}
        delay={0.8}
        drawSec={2.4}
        directLabel={false}
      />
      <div style={{ display: "flex", gap: 28, marginTop: 22 }}>
        <Reveal delay={3.4} style={{ flex: 1 }}>
          <StatCard
            label="Certificate"
            accent={C.s1}
            sub="measurements to reach 90% of the best effect"
          >
            <CountUp to={e.cert_mean} dec={1} delay={3.5} color={C.s1} size={104} />
          </StatCard>
        </Reveal>
        <Reveal delay={4.3} style={{ flex: 1 }}>
          <StatCard label="Random order" accent={C.s4} sub="same target, same pool">
            <CountUp to={e.rand_mean} dec={1} delay={4.4} color={C.s4} size={104} />
          </StatCard>
        </Reveal>
        <Reveal delay={5.2} style={{ flex: 2 }}>
          <StatCard
            label={`${e.speedup.toFixed(0)}× cheaper search`}
            accent={C.s3}
            sub=""
          >
            <div style={{ fontSize: 36, color: C.ink, lineHeight: 1.32, marginTop: 4 }}>
              And the prior costs nothing extra &mdash; it reuses adjoints the probe
              already required.
            </div>
          </StatCard>
        </Reveal>
      </div>
    </Frame>
  );
};
