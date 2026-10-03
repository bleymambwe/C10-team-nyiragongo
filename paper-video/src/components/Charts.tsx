import React from "react";
import { Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { C, F } from "../theme";

const nice = (lo: number, hi: number, n = 4) => {
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const s = Math.floor(lo / step) * step;
  const e = Math.ceil(hi / step) * step;
  const t: number[] = [];
  for (let v = s; v <= e + step * 1e-9; v += step) t.push(Number(v.toFixed(10)));
  return { lo: s, hi: e, ticks: t };
};

type Series = { name: string; y: number[]; color: string };

/** Multi-series line chart whose lines draw on over time. */
export const LineChartV: React.FC<{
  x: (string | number)[];
  series: Series[];
  width?: number;
  height?: number;
  yLabel?: string;
  xLabel?: string;
  yMin?: number;
  yMax?: number;
  dec?: number;
  delay?: number;
  drawSec?: number;
  directLabel?: boolean;
}> = ({
  x,
  series,
  width = 1640,
  height = 360,
  yLabel,
  xLabel,
  yMin,
  yMax,
  dec = 2,
  delay = 0.4,
  drawSec = 1.9,
  directLabel = true,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const m = { l: 158, r: 190, t: 26, b: 76 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const all = series.flatMap((s) => s.y).filter(Number.isFinite);
  const { lo, hi, ticks } = nice(
    yMin !== undefined ? yMin : Math.min(...all),
    yMax !== undefined ? yMax : Math.max(...all),
  );
  const X = (i: number) => m.l + (x.length < 2 ? iw / 2 : (i / (x.length - 1)) * iw);
  const Y = (v: number) => m.t + ih - ((v - lo) / (hi - lo)) * ih;

  const prog = interpolate(
    frame,
    [delay * fps, (delay + drawSec) * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.bezier(0.4, 0, 0.2, 1) },
  );

  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={m.l} x2={m.l + iw} y1={Y(t)} y2={Y(t)} stroke={C.rule} strokeWidth={1.5} />
          <text
            x={m.l - 20}
            y={Y(t) + 10}
            textAnchor="end"
            fill={C.ink3}
            fontFamily={F.mono}
            fontSize={28}
          >
            {t.toFixed(dec)}
          </text>
        </g>
      ))}
      <line
        x1={m.l}
        x2={m.l + iw}
        y1={m.t + ih}
        y2={m.t + ih}
        stroke={C.ruleStrong}
        strokeWidth={2}
      />
      {x.map((v, i) =>
        i % Math.max(1, Math.ceil(x.length / 10)) !== 0 && i !== x.length - 1 ? null : (
          <text
            key={i}
            x={X(i)}
            y={m.t + ih + 40}
            textAnchor="middle"
            fill={C.ink3}
            fontFamily={F.mono}
            fontSize={28}
          >
            {v}
          </text>
        ),
      )}
      {xLabel ? (
        <text
          x={m.l + iw / 2}
          y={height - 8}
          textAnchor="middle"
          fill={C.ink3}
          fontFamily={F.mono}
          fontSize={24}
          letterSpacing="0.14em"
        >
          {xLabel.toUpperCase()}
        </text>
      ) : null}
      {yLabel ? (
        <text
          x={20}
          y={m.t + ih / 2}
          textAnchor="middle"
          fill={C.ink3}
          fontFamily={F.mono}
          fontSize={24}
          letterSpacing="0.14em"
          transform={`rotate(-90 20 ${m.t + ih / 2})`}
        >
          {yLabel.toUpperCase()}
        </text>
      ) : null}

      {series.map((s, si) => {
        const pts = s.y.map((v, i) => [X(i), Y(v)] as const);
        const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join(" ");
        // approximate path length for the draw-on effect
        let len = 0;
        for (let i = 1; i < pts.length; i++)
          len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
        const shown = Math.floor(prog * (pts.length - 1));
        const last = pts[Math.max(0, shown)];
        return (
          <g key={s.name}>
            <path
              d={d}
              fill="none"
              stroke={s.color}
              strokeWidth={5}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray={len}
              strokeDashoffset={len * (1 - prog)}
            />
            {pts.map((p, i) =>
              i <= shown ? (
                <circle
                  key={i}
                  cx={p[0]}
                  cy={p[1]}
                  r={6.5}
                  fill={s.color}
                  stroke={C.ground}
                  strokeWidth={3}
                />
              ) : null,
            )}
            {directLabel && prog > 0.05 ? (
              <text
                x={last[0] + 22}
                y={last[1] + 10}
                fill={s.color}
                fontFamily={F.mono}
                fontSize={28}
                fontWeight={500}
                opacity={interpolate(prog, [0.05, 0.3], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                })}
              >
                {s.name}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
};

/** Grouped bars that grow from the baseline. */
export const BarsV: React.FC<{
  groups: (string | number)[];
  series: Series[];
  width?: number;
  height?: number;
  yLabel?: string;
  xLabel?: string;
  dec?: number;
  delay?: number;
}> = ({
  groups,
  series,
  width = 1640,
  height = 360,
  yLabel,
  xLabel,
  dec = 1,
  delay = 0.4,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const m = { l: 158, r: 40, t: 26, b: 76 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const all = series.flatMap((s) => s.y).filter(Number.isFinite);
  const { lo, hi, ticks } = nice(Math.min(0, ...all), Math.max(0, ...all));
  const Y = (v: number) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const gw = iw / groups.length;
  const bw = Math.max(6, (gw - 16) / series.length - 5);

  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={m.l} x2={m.l + iw} y1={Y(t)} y2={Y(t)} stroke={C.rule} strokeWidth={1.5} />
          <text
            x={m.l - 20}
            y={Y(t) + 10}
            textAnchor="end"
            fill={C.ink3}
            fontFamily={F.mono}
            fontSize={28}
          >
            {t.toFixed(dec)}
          </text>
        </g>
      ))}
      <line x1={m.l} x2={m.l + iw} y1={Y(0)} y2={Y(0)} stroke={C.ruleStrong} strokeWidth={2} />
      {groups.map((g, gi) => (
        <g key={gi}>
          <text
            x={m.l + gi * gw + gw / 2}
            y={m.t + ih + 40}
            textAnchor="middle"
            fill={C.ink3}
            fontFamily={F.mono}
            fontSize={28}
          >
            {g}
          </text>
          {series.map((s, si) => {
            const v = s.y[gi];
            if (!Number.isFinite(v)) return null;
            const t0 = delay * fps + gi * 0.045 * fps;
            const grow = interpolate(frame, [t0, t0 + 0.75 * fps], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            });
            const full = Y(v) - Y(0);
            const h = Math.abs(full) * grow;
            const yTop = full < 0 ? Y(0) - h : Y(0);
            return (
              <rect
                key={si}
                x={m.l + gi * gw + 8 + si * (bw + 5)}
                y={yTop}
                width={bw}
                height={Math.max(1, h)}
                rx={5}
                fill={s.color}
                stroke={C.ground}
                strokeWidth={2}
              />
            );
          })}
        </g>
      ))}
      {xLabel ? (
        <text
          x={m.l + iw / 2}
          y={height - 8}
          textAnchor="middle"
          fill={C.ink3}
          fontFamily={F.mono}
          fontSize={24}
          letterSpacing="0.14em"
        >
          {xLabel.toUpperCase()}
        </text>
      ) : null}
      {yLabel ? (
        <text
          x={20}
          y={m.t + ih / 2}
          textAnchor="middle"
          fill={C.ink3}
          fontFamily={F.mono}
          fontSize={24}
          letterSpacing="0.14em"
          transform={`rotate(-90 20 ${m.t + ih / 2})`}
        >
          {yLabel.toUpperCase()}
        </text>
      ) : null}
    </svg>
  );
};

export const Legend: React.FC<{ items: { name: string; color: string }[] }> = ({
  items,
}) => (
  <div style={{ display: "flex", gap: 40, marginBottom: 16, flexWrap: "wrap" }}>
    {items.map((it) => (
      <div key={it.name} style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div
          style={{ width: 30, height: 6, borderRadius: 3, background: it.color, flex: "none" }}
        />
        <span style={{ color: C.ink2, fontSize: 32, fontFamily: F.sans }}>{it.name}</span>
      </div>
    ))}
  </div>
);

/** A number that counts up, for headline statistics. */
export const CountUp: React.FC<{
  to: number;
  dec?: number;
  suffix?: string;
  delay?: number;
  dur?: number;
  color?: string;
  size?: number;
}> = ({ to, dec = 1, suffix = "", delay = 0.3, dur = 1.1, color = C.ink, size = 150 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = interpolate(frame, [delay * fps, (delay + dur) * fps], [0, to], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  return (
    <span
      style={{
        fontFamily: F.mono,
        fontSize: size,
        fontWeight: 500,
        color,
        letterSpacing: "-0.03em",
        fontVariantNumeric: "tabular-nums",
        lineHeight: 1,
      }}
    >
      {v.toFixed(dec)}
      {suffix}
    </span>
  );
};

export const StatCard: React.FC<{
  label: string;
  children: React.ReactNode;
  sub?: string;
  accent?: string;
}> = ({ label, children, sub, accent = C.accent }) => (
  <div
    style={{
      background: C.surface,
      border: `2px solid ${C.rule}`,
      borderTop: `5px solid ${accent}`,
      borderRadius: 14,
      padding: "22px 28px",
      display: "flex",
      flexDirection: "column",
      gap: 8,
      flex: 1,
      minWidth: 0,
    }}
  >
    <div
      style={{
        fontFamily: F.mono,
        fontSize: 27,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: C.ink3,
      }}
    >
      {label}
    </div>
    <div>{children}</div>
    {sub ? (
      <div style={{ fontSize: 31, color: C.ink2, lineHeight: 1.35 }}>{sub}</div>
    ) : null}
  </div>
);
