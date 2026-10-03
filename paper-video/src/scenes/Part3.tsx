import React from "react";
import { AbsoluteFill } from "remotion";
import { Frame, Reveal } from "../components/Frame";
import { C, F, LAYOUT } from "../theme";
import facts from "../facts.json";

/* ------------------------------------------------------------------ 12 */
export const S12Failures: React.FC = () => {
  const items = [
    {
      tag: "A REGISTERED PREDICTION FAILED",
      t: "Confounding does not widen the gap",
      d: "Tested twice. In pretrained GPT‑2 the dial moved alignment not at all. So we controlled training itself: across 12 transformers trained from scratch, sweeping the confound 0.50→0.98 moved alignment by 0.004 — one fifth of the seed-to-seed variation.",
      w: "Two designs, one inheriting representations and one creating them. The cause is low-rank structure, not a flawed dataset. Cleaning your labels will not fix it.",
      c: C.bad,
    },
    {
      tag: "A PROPOSITION IS ONLY APPROXIMATE",
      t: "Probe-orthogonal ≠ undetectable",
      d: "Theory predicts such directions classify at exactly 0.500. In GPT-2 they classify around 0.56 — the Gaussian premise does not hold exactly in a real model.",
      w: "The steering half of the claim survives intact; the detectability half is weaker than stated.",
      c: C.warn,
    },
    {
      tag: "A METRIC WAS DISCARDED",
      t: "Prompt perplexity measures nothing here",
      d: "It returned identically zero at every layer. Causal masking means a write at the final token cannot affect earlier predictions.",
      w: "A metric that cannot be non-zero is worse than no metric — it reads as evidence of safety.",
      c: C.s5,
    },
  ];
  return (
    <Frame eyebrow="What did not work" title="Three results reported against interest" accent={C.warn}>
      <div style={{ display: "flex", gap: 26 }}>
        {items.map((it, i) => (
          <Reveal key={it.t} delay={0.8 + i * 3.6} style={{ flex: 1 }}>
            <div
              style={{
                background: C.surface,
                border: `2px solid ${C.rule}`,
                borderTop: `6px solid ${it.c}`,
                borderRadius: 16,
                padding: "22px 26px",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              <div
                style={{
                  fontFamily: F.mono,
                  fontSize: 25,
                  letterSpacing: "0.13em",
                  color: it.c,
                }}
              >
                {it.tag}
              </div>
              <div style={{ fontSize: 41, color: C.ink, fontWeight: 600, lineHeight: 1.22 }}>
                {it.t}
              </div>
              <div style={{ fontSize: 27, color: C.ink2, lineHeight: 1.38 }}>{it.d}</div>
              <div
                style={{
                  fontSize: 27,
                  color: C.ink,
                  lineHeight: 1.38,
                  marginTop: "auto",
                  paddingTop: 16,
                  borderTop: `1px solid ${C.rule}`,
                  fontStyle: "italic",
                }}
              >
                {it.w}
              </div>
            </div>
          </Reveal>
        ))}
      </div>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 13 */
export const S13Limitations: React.FC = () => {
  const items = [
    ["Model scale", "Four CPU cores, no GPU. Largest model is GPT-2 small. The theory is scale-free; the measurements are not."],
    ["One weak signal on scale", "Across four models spanning a 12× range of width, the absolute effect rank stays a small constant (1.6–4.6) rather than growing with d. If that holds, the gap widens with scale rather than closing. Four points is far too few to fit a trend."],
    ["First order", "The certificate is first-order. Its validity radius is computed per layer, and where the write magnitude exceeds it, the prediction is out of warranty."],
    ['What "dark" means', "Invisible to this probe family — linear, this concept, this data. Not a claim of invisibility in principle."],
  ];
  return (
    <Frame eyebrow="Limitations" title="Stated plainly, because they are load-bearing">
      {items.map(([t, d], i) => (
        <Reveal key={t} delay={0.8 + i * 2.2} style={{ marginBottom: 11 }}>
          <div
            style={{
              background: C.surface,
              border: `2px solid ${C.rule}`,
              borderLeft: `6px solid ${C.ink3}`,
              borderRadius: 14,
              padding: "16px 28px",
            }}
          >
            <div style={{ fontSize: 34, color: C.ink, fontWeight: 600 }}>{t}</div>
            <div style={{ fontSize: 28, color: C.ink2, marginTop: 6, lineHeight: 1.35 }}>{d}</div>
          </div>
        </Reveal>
      ))}
    </Frame>
  );
};

/* ------------------------------------------------------------------ 14 */
export const S14Safety: React.FC = () => {
  return (
    <Frame eyebrow="Consequence" title="What this means for monitoring" accent={C.bad}>
      <Reveal delay={0.7}>
        <div
          style={{
            background: "rgba(240,125,124,0.09)",
            border: `2px solid ${C.bad}`,
            borderRadius: 18,
            padding: "50px 56px",
          }}
        >
          <div style={{ fontSize: 56, color: C.ink, lineHeight: 1.32, maxWidth: 1450 }}>
            A monitor built from a toxicity probe is close to blind on the directions that most
            strongly drive toxic output.
          </div>
        </div>
      </Reveal>
      <Reveal delay={3.2} style={{ marginTop: 46 }}>
        <div style={{ fontSize: 48, color: C.ink2, lineHeight: 1.4, maxWidth: 1500 }}>
          Probe-based monitoring is therefore not, by itself, evidence of control.
        </div>
      </Reveal>
      <Reveal delay={5.0} style={{ marginTop: 36 }}>
        <div
          style={{
            fontFamily: F.mono,
            fontSize: 36,
            color: C.bad,
            letterSpacing: "0.04em",
          }}
        >
          Given the decoupling theorem, that is not a worry &mdash; it is a consequence.
        </div>
      </Reveal>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 15 */
export const S15Next: React.FC = () => {
  const steps = [
    ["1", "Replicate at 1B–7B parameters", "If the live fraction grows with width, the argument loses its force. The biggest threat to external validity.", C.s1],
    ["2", "Nonlinear and SAE readouts", "Tests whether dark directions survive a stronger detector. This is the real safety question.", C.s3],
    ["3", "Test the non-normality hypothesis", "The layer-to-layer Jacobian is a product of generically non-normal factors, which concentrates singular values. A plausible structural cause of the small effect rank — and measurable with pseudospectral diagnostics we have not computed.", C.s4],
    ["4", "A second-order certificate", "Extends validity past the first-order radius, where most deployed steering actually operates.", C.s5],
  ];
  return (
    <Frame eyebrow="Next steps" title="Ordered by what would most change the conclusions">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 26 }}>
        {steps.map(([n, t, d, col], i) => (
          <Reveal key={n as string} delay={0.8 + i * 1.9}>
            <div
              style={{
                background: C.surface,
                border: `2px solid ${C.rule}`,
                borderTop: `6px solid ${col as string}`,
                borderRadius: 16,
                padding: "28px 32px",
                height: "100%",
                display: "flex",
                gap: 24,
              }}
            >
              <div
                style={{
                  fontFamily: F.mono,
                  fontSize: 54,
                  color: col as string,
                  fontWeight: 600,
                  lineHeight: 1,
                }}
              >
                {n}
              </div>
              <div>
                <div style={{ fontSize: 40, color: C.ink, fontWeight: 600, lineHeight: 1.22 }}>
                  {t}
                </div>
                <div style={{ fontSize: 31, color: C.ink2, marginTop: 10, lineHeight: 1.42 }}>
                  {d}
                </div>
              </div>
            </div>
          </Reveal>
        ))}
      </div>
    </Frame>
  );
};

/* ------------------------------------------------------------------ 16 */
export const S16Close: React.FC = () => {
  const e = facts.e3;
  return (
    <AbsoluteFill style={{ backgroundColor: C.ground, fontFamily: F.sans }}>
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(110% 80% at 50% 45%, rgba(58,96,130,0.24) 0%, rgba(13,17,20,0) 66%)",
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
              fontSize: 30,
              letterSpacing: "0.22em",
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 30,
            }}
          >
            The claim in one sentence
          </div>
        </Reveal>
        <Reveal i={1}>
          <div
            style={{
              fontSize: 64,
              fontWeight: 600,
              letterSpacing: "-0.022em",
              lineHeight: 1.28,
              color: C.ink,
              maxWidth: 1560,
            }}
          >
            Reading a residual stream and writing to it are governed by different subspaces, that
            separation is measurable in advance from one backward pass, and probe accuracy carries
            essentially no information about causal control.
          </div>
        </Reveal>

        <div style={{ display: "flex", gap: 60, marginTop: 64 }}>
          {[
            ["ρ", e.mean_rho.toFixed(3), "probe / mechanism alignment"],
            ["r_eff/d", `${(e.mean_reff * 100).toFixed(2)}%`, "of the stream is live"],
            ["6.0%", "", "control recovered by the probe"],
          ].map(([a, b, c], i) => (
            <Reveal key={c} i={2 + i} style={{ flex: 1 }}>
              <div style={{ borderTop: `4px solid ${C.s1}`, paddingTop: 20 }}>
                <div
                  style={{
                    fontFamily: F.mono,
                    fontSize: 62,
                    color: C.ink,
                    fontWeight: 500,
                    lineHeight: 1.1,
                  }}
                >
                  {b || a}
                </div>
                <div style={{ fontSize: 28, color: C.ink3, marginTop: 10 }}>{c}</div>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal i={6} style={{ marginTop: 60 }}>
          <div style={{ fontFamily: F.mono, fontSize: 32, color: C.ink3 }}>
            Every number is reproducible from{" "}
            <span style={{ color: C.ink2 }}>results/data/*.json</span>
          </div>
        </Reveal>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
