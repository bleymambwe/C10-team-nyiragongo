import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { C, F, LAYOUT } from "../theme";

// Shared scene chrome: ground, a hairline rule, an eyebrow label and a heading.
// Every scene sits inside this so the frame never jumps between cuts.
export const Frame: React.FC<{
  eyebrow?: string;
  title?: string;
  children?: React.ReactNode;
  accent?: string;
}> = ({ eyebrow, title, children, accent = C.accent }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: C.ground, fontFamily: F.sans }}>
      {/* faint vignette so the centre reads brighter than the edges */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(120% 90% at 50% 32%, rgba(58,86,110,0.20) 0%, rgba(13,17,20,0) 62%)",
        }}
      />

      <AbsoluteFill
        style={{
          paddingLeft: LAYOUT.padX,
          paddingRight: LAYOUT.padX,
          paddingTop: LAYOUT.padY,
          paddingBottom: LAYOUT.padY,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {eyebrow ? (
          <div
            style={{
              fontFamily: F.mono,
              fontSize: LAYOUT.eyebrow,
              letterSpacing: "0.20em",
              textTransform: "uppercase",
              color: accent,
              marginBottom: 22,
              opacity: interpolate(frame, [0, 0.5 * fps], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
                easing: Easing.bezier(0.16, 1, 0.3, 1),
              }),
              translate: interpolate(frame, [0, 0.6 * fps], ["-16px 0px", "0px 0px"], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
                easing: Easing.bezier(0.16, 1, 0.3, 1),
              }),
            }}
          >
            {eyebrow}
          </div>
        ) : null}

        {title ? (
          <div
            style={{
              fontSize: LAYOUT.h2,
              fontWeight: 600,
              letterSpacing: "-0.02em",
              lineHeight: 1.08,
              color: C.ink,
              maxWidth: 1500,
              marginBottom: 34,
              textWrap: "balance",
              opacity: interpolate(frame, [0.15 * fps, 0.8 * fps], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
                easing: Easing.bezier(0.16, 1, 0.3, 1),
              }),
              translate: interpolate(
                frame,
                [0.15 * fps, 0.9 * fps],
                ["0px 18px", "0px 0px"],
                {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                  easing: Easing.bezier(0.16, 1, 0.3, 1),
                },
              ),
            }}
          >
            {title}
          </div>
        ) : null}

        <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
          {children}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// Reveals children on a stagger. `i` is the item's position in the stagger.
export const Reveal: React.FC<{
  i?: number;
  delay?: number;
  children: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ i = 0, delay = 0, children, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t0 = delay * fps + i * 0.28 * fps;

  return (
    <div
      style={{
        ...style,
        opacity: interpolate(frame, [t0, t0 + 0.55 * fps], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
          easing: Easing.bezier(0.16, 1, 0.3, 1),
        }),
        translate: interpolate(
          frame,
          [t0, t0 + 0.7 * fps],
          ["0px 22px", "0px 0px"],
          {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          },
        ),
      }}
    >
      {children}
    </div>
  );
};
