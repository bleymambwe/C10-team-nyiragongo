// Visual identity shared with the HTML report, so the two read as one system.
// Dark ground throughout: this is an instrument panel, not a document.

export const C = {
  ground: "#0d1114",
  surface: "#161c21",
  surfaceAlt: "#1d252b",
  rule: "#2c353b",
  ruleStrong: "#414f57",
  ink: "#eef3f6",
  ink2: "#a7b8c1",
  ink3: "#74868f",
  accent: "#6aa9ec",
  s1: "#3987e5",
  s2: "#e2703a",
  s3: "#25b184",
  s4: "#d9a223",
  s5: "#9085e9",
  good: "#43c295",
  bad: "#f07d7c",
  warn: "#dfa93a",
} as const;

export const F = {
  sans: '"IBM Plex Sans", system-ui, sans-serif',
  mono: '"IBM Plex Mono", ui-monospace, monospace',
  serif: '"IBM Plex Serif", Georgia, serif',
} as const;

// 1920x1080. Safe area and type scale follow the video-layout guidance,
// scaled from the 1080-wide baseline.
export const LAYOUT = {
  W: 1920,
  H: 1080,
  padX: 140,
  padY: 108,
  eyebrow: 30,
  h1: 128,
  h2: 84,
  body: 46,
  small: 34,
  tiny: 26,
  stat: 150,
} as const;
