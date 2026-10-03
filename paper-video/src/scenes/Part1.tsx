import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Frame, Reveal } from "../components/Frame";
import { CountUp, StatCard } from "../components/Charts";
import { C, F, LAYOUT } from "../theme";
import facts from "../facts.json";

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

/* ------------------------------------------------------------------ 00 */
export const S00Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill style={{ backgroundColor: C.ground, fontFamily: F.sans }}>
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(110% 80% at 50% 40%, rgba(58,96,130,0.26) 0%, rgba(13,17,20,0) 65%)",
        }}
      />
      <AbsoluteFill
        style={{
          justifyContent: "center",
          paddingLeft: LAYOUT.padX,
          paddingRight: LAYOUT.padX,
        }}
      >
        <Reveal i={0}>
          <div
            style={{
              fontFamily: F.mono,
              fontSize: 32,
              letterSpacing: "0.24em",
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 34,
            }}
          >
            Mechanistic interpretability &middot; measurement report
          </div>
        </Reveal>
        <Reveal i={1}>
          <div
            style={{
              fontSize: LAYOUT.h1,
              fontWeight: 700,
              letterSpacing: "-0.035em",
              lineHeight: 1.02,
              color: C.ink,
              maxWidth: 1500,
            }}
          >
            Seeing is not steering
          </div>
        </Reveal>
        <Reveal i={2}>
          <div
            style={{
              fontFamily: F.serif,
              fontSize: 52,
              lineHeight: 1.42,
              color: C.ink2,
              maxWidth: 1320,
              marginTop: 36,
            }}
          >
            Why linear probes find concepts that steering vectors cannot control &mdash;
            and what to measure instead.
          </div>
        </Reveal>
        <Reveal i={3}>
          <div
            style={{
              marginTop: 54,
              height: 5,
              width: interpolate(frame, [1.6 * fps, 3.4 * fps], [0, 620], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
                easing: Easing.bezier(0.16, 1, 0.3, 1),
              }),
              background: `linear-gradient(90deg, ${C.s1}, ${C.s3})`,
              borderRadius: 3,
            }}
          />
        </Reveal>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ 01 */
export const S01Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const step = (t: number) =>
    interpolate(frame, [t * fps, (t + 0.6) * fps], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.bezier(0.16, 1, 0.3, 1),
    });

  const Box: React.FC<{ label: string; sub: string; color: string; o: number }> = ({
    label,
    sub,
    color,
    o,
  }) => (
    <div
      style={{
        opacity: o,
        background: C.surface,
        border: `2px solid ${color}`,
        borderRadius: 16,
        padding: "34px 40px",
        flex: 1,
      }}
    >
      <div style={{ fontFamily: F.mono, fontSize: 30, color, letterSpacing: "0.12em" }}>
        {label}
      </div>
      <div style={{ fontSize: 40, color: C.ink, marginTop: 14, lineHeight: 1.3 }}>{sub}</div>
    </div>
  );

  return (
    <Frame eyebrow="The habit" title="One vector, two incompatible jobs">
      <div style={{ display: "flex", gap: 40, marginTop: 20 }}>
        <Box
          label="AS A SENSOR"
          sub="Train a probe until it detects the concept."
          color={C.s2}
          o={step(0.8)}
        />
        <Box
          label="AS AN ACTUATOR"
          sub="Add that same direction back and expect behaviour to move."
          color={C.s1}
          o={step(2.2)}
        />
      </div>

      <Reveal delay={4.2} style={{ marginTop: 58 }}>
        <div
          style={{
            fontFamily: F.serif,
            fontSize: 50,
            lineHeight: 1.45,
            color: C.ink,
            maxWidth: 1480,
          }}
        >
          Reading and writing are <em style={{ color: C.accent }}>different operations</em>.
          Nothing guarantees one vector is good at both.
        </div>
      </Reveal>

      <Reveal delay={5.8} style={{ marginTop: 40 }}>
        <div style={{ fontSize: 38, color: C.ink3, maxWidth: 1400, lineHeight: 1.45 }}>
          The field reports this gap after running the interventions. Nobody predicts it
          beforehand.
        </div>
      </Reveal>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 02 */
export const S02TwoGeometries: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const o = (t: number) =>
    interpolate(frame, [t * fps, (t + 0.7) * fps], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.bezier(0.16, 1, 0.3, 1),
    });

  const Side: React.FC<{
    tag: string;
    color: string;
    title: string;
    formula: string;
    depends: string;
    op: number;
  }> = ({ tag, color, title, formula, depends, op }) => (
    <div
      style={{
        opacity: op,
        flex: 1,
        background: C.surface,
        border: `2px solid ${C.rule}`,
        borderTop: `6px solid ${color}`,
        borderRadius: 16,
        padding: "34px 38px",
      }}
    >
      <div
        style={{
          fontFamily: F.mono,
          fontSize: 27,
          letterSpacing: "0.16em",
          color,
          textTransform: "uppercase",
        }}
      >
        {tag}
      </div>
      <div style={{ fontSize: 46, color: C.ink, marginTop: 14, fontWeight: 600 }}>{title}</div>
      <div
        style={{
          fontFamily: F.mono,
          fontSize: 40,
          color: C.ink,
          background: C.surfaceAlt,
          borderRadius: 10,
          padding: "20px 22px",
          marginTop: 22,
        }}
      >
        {formula}
      </div>
      <div style={{ fontSize: 34, color: C.ink2, marginTop: 20, lineHeight: 1.4 }}>
        {depends}
      </div>
    </div>
  );

  return (
    <Frame eyebrow="The mechanism" title="Two geometries, one residual stream">
      <div style={{ display: "flex", gap: 40 }}>
        <Side
          tag="Read side"
          color={C.s2}
          title="Representation geometry"
          formula="w = (Σ + γI)⁻¹ δ"
          depends="Class means and within-class covariance. This is what a probe fits."
          op={o(1.2)}
        />
        <Side
          tag="Write side"
          color={C.s1}
          title="Effect geometry"
          formula="W_g = E[ g gᵀ ],  g = ∇ₓ Φ"
          depends="Adjoint covectors carrying this layer to the behaviour. This is what steering moves."
          op={o(3.4)}
        />
      </div>

      <Reveal delay={6.2} style={{ marginTop: 46 }}>
        <div
          style={{
            background: "rgba(106,169,236,0.10)",
            borderLeft: `6px solid ${C.accent}`,
            borderRadius: "0 14px 14px 0",
            padding: "30px 36px",
          }}
        >
          <div style={{ fontSize: 44, color: C.ink, lineHeight: 1.4 }}>
            <span style={{ fontFamily: F.mono, color: C.s2 }}>Σ</span> never enters{" "}
            <span style={{ fontFamily: F.mono, color: C.s1 }}>W_g</span>.{" "}
            <span style={{ fontFamily: F.mono, color: C.s1 }}>W_g</span> never enters the probe.
          </div>
          <div style={{ fontSize: 36, color: C.ink2, marginTop: 14 }}>
            Disjoint information. Nothing in training couples them. Their alignment is a free
            parameter.
          </div>
        </div>
      </Reveal>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 03 */
export const S03Certificate: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <Frame eyebrow="The instrument" title="A certificate you can compute before steering">
      <Reveal delay={0.7}>
        <div
          style={{
            background: C.surface,
            border: `2px solid ${C.rule}`,
            borderRadius: 18,
            padding: "48px 56px",
            textAlign: "center",
          }}
        >
          <div
            style={{
              fontFamily: F.mono,
              fontSize: 78,
              color: C.ink,
              letterSpacing: "-0.01em",
            }}
          >
            S̄(ℓ, w) ={" "}
            <span style={{ color: C.accent }}>| ḡ<sub>ℓ</sub>ᵀ w |</span> / ‖w‖
          </div>
          <div style={{ fontSize: 34, color: C.ink3, marginTop: 24, fontFamily: F.mono }}>
            projection of the mean adjoint covector onto a candidate direction
          </div>
        </div>
      </Reveal>

      <div style={{ display: "flex", gap: 34, marginTop: 50 }}>
        {[
          ["One backward pass", "over data the probe already needed", C.s3],
          ["Zero interventions", "no forward sweeps, no steering runs", C.s1],
          ["Predicts the effect", "before any intervention is performed", C.s4],
        ].map(([a, b, col], i) => (
          <Reveal key={a} delay={2.6 + i * 0.8} style={{ flex: 1 }}>
            <div
              style={{
                background: C.surface,
                border: `2px solid ${C.rule}`,
                borderLeft: `6px solid ${col as string}`,
                borderRadius: 14,
                padding: "28px 32px",
                height: "100%",
              }}
            >
              <div style={{ fontSize: 42, color: C.ink, fontWeight: 600 }}>{a}</div>
              <div style={{ fontSize: 32, color: C.ink2, marginTop: 12, lineHeight: 1.4 }}>
                {b}
              </div>
            </div>
          </Reveal>
        ))}
      </div>

      <div
        style={{
          marginTop: "auto",
          fontSize: 34,
          color: C.ink3,
          fontFamily: F.mono,
          opacity: interpolate(frame, [5.4 * fps, 6.2 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        cost: N reverse-mode passes &nbsp;·&nbsp; interventions run: 0
      </div>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 04 */
export const S04Regions: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const cells = [
    { n: "CAUSAL", d: "probe sees it · steering works", c: C.s3, note: "what practice assumes" },
    { n: "EPIPHENOMENAL", d: "probe sees it · steering does nothing", c: C.s2, note: "measured: the common case" },
    { n: "DARK", d: "probe is blind · steering works", c: C.s1, note: "constructible in closed form" },
    { n: "INERT", d: "invisible · inconsequential", c: C.ink3, note: `measured: ${pct(1 - facts.e3.mean_reff, 1)} of the stream` },
  ];
  return (
    <Frame eyebrow="The partition" title="Four regions of a residual stream">
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 32,
          marginTop: 10,
        }}
      >
        {cells.map((c, i) => {
          const t0 = 1.0 + i * 1.15;
          return (
            <div
              key={c.n}
              style={{
                background: C.surface,
                border: `2px solid ${C.rule}`,
                borderTop: `6px solid ${c.c}`,
                borderRadius: 16,
                padding: "30px 36px",
                opacity: interpolate(frame, [t0 * fps, (t0 + 0.6) * fps], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                  easing: Easing.bezier(0.16, 1, 0.3, 1),
                }),
                scale: interpolate(frame, [t0 * fps, (t0 + 0.7) * fps], [0.94, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                  easing: Easing.bezier(0.16, 1, 0.3, 1),
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                } as any),
              }}
            >
              <div
                style={{
                  fontFamily: F.mono,
                  fontSize: 42,
                  color: c.c,
                  letterSpacing: "0.10em",
                  fontWeight: 600,
                }}
              >
                {c.n}
              </div>
              <div style={{ fontSize: 36, color: C.ink, marginTop: 12 }}>{c.d}</div>
              <div style={{ fontSize: 31, color: C.ink3, marginTop: 10, fontFamily: F.mono }}>
                {c.note}
              </div>
            </div>
          );
        })}
      </div>

      <Reveal delay={5.6} style={{ marginTop: 44 }}>
        <div style={{ fontSize: 40, color: C.ink2, lineHeight: 1.4, maxWidth: 1500 }}>
          Membership is decided by two computable numbers &mdash; the certificate, and the
          probe&rsquo;s accuracy.
        </div>
      </Reveal>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 05 */
export const S05Theorem: React.FC = () => {
  return (
    <Frame eyebrow="Theorem 2 · exact decoupling" title="They can be pulled apart completely">
      <div style={{ display: "flex", gap: 34, marginTop: 6 }}>
        <Reveal delay={0.9} style={{ flex: 1 }}>
          <StatCard label="Optimal probe" accent={C.s2} sub="classifies almost perfectly">
            <CountUp to={0.998} dec={3} delay={1.0} color={C.s2} size={130} />
          </StatCard>
        </Reveal>
        <Reveal delay={2.4} style={{ flex: 1 }}>
          <StatCard
            label="Its steering effect"
            accent={C.bad}
            sub="not small — exactly zero"
          >
            <div
              style={{
                fontFamily: F.mono,
                fontSize: 96,
                color: C.bad,
                fontWeight: 500,
                lineHeight: 1,
              }}
            >
              2×10⁻¹⁵
            </div>
          </StatCard>
        </Reveal>
        <Reveal delay={4.6} style={{ flex: 1 }}>
          <StatCard
            label="A chance-level direction"
            accent={C.s1}
            sub="AUC 0.500 — and it steers maximally"
          >
            <CountUp to={0.5} dec={3} delay={4.7} color={C.s1} size={130} />
          </StatCard>
        </Reveal>
      </div>

      <Reveal delay={7.0} style={{ marginTop: 48 }}>
        <div
          style={{
            background: "rgba(106,169,236,0.10)",
            borderLeft: `6px solid ${C.accent}`,
            borderRadius: "0 14px 14px 0",
            padding: "32px 38px",
          }}
        >
          <div style={{ fontSize: 46, color: C.ink, lineHeight: 1.38 }}>
            Corollary: no function of probe accuracy alone can estimate steering efficacy.
          </div>
        </div>
      </Reveal>

      <Reveal delay={9.0} style={{ marginTop: 34 }}>
        <div style={{ fontFamily: F.mono, fontSize: 36, color: C.ink3 }}>
          correlation across a swept family of systems:{" "}
          <span style={{ color: C.ink }}>r = {facts.e1_corr.toFixed(3)}</span>
        </div>
      </Reveal>
    </Frame>
  );
};
