import React from "react";
import { AbsoluteFill, Sequence, staticFile } from "remotion";
import { Audio } from "@remotion/media";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";

import narration from "./narration.json";
import { C } from "./theme";
import { S00Title, S01Problem, S02TwoGeometries, S03Certificate, S04Regions, S05Theorem } from "./scenes/Part1";
import { S06Leakage, S07LowRank, S08Gpt2, S09NotCircular, S10Independence, S11RealData } from "./scenes/Part2";
import { S12Failures, S13Limitations, S14Safety, S15Next, S16Close } from "./scenes/Part3";
import { S12Trained, S13Budget } from "./scenes/Part4";

export const FPS = 30;
export const TRANSITION_FRAMES = 12;

const COMPONENTS: Record<string, React.FC> = {
  title: S00Title,
  problem: S01Problem,
  two_geometries: S02TwoGeometries,
  certificate: S03Certificate,
  regions: S04Regions,
  theorem: S05Theorem,
  leakage: S06Leakage,
  lowrank: S07LowRank,
  gpt2: S08Gpt2,
  notcircular: S09NotCircular,
  independence: S10Independence,
  realdata: S11RealData,
  trained: S12Trained,
  budget: S13Budget,
  failures: S12Failures,
  limitations: S13Limitations,
  safety: S14Safety,
  next: S15Next,
  close: S16Close,
};

/** Frames each scene occupies on the timeline, before transition overlap. */
export const sceneFrames = narration.scenes.map((s) =>
  Math.round((s.durationSec + 0.55) * FPS),
);

/**
 * A TransitionSeries overlaps neighbours, shortening the timeline by
 * TRANSITION_FRAMES per cut. The audio is placed on an independent track using
 * the same accounting, so narration stays locked to its scene.
 */
export const audioStartFrames = (() => {
  const starts: number[] = [];
  let t = 0;
  sceneFrames.forEach((f, i) => {
    starts.push(t);
    t += f - (i < sceneFrames.length - 1 ? TRANSITION_FRAMES : 0);
  });
  return starts;
})();

export const totalFrames =
  sceneFrames.reduce((a, b) => a + b, 0) -
  TRANSITION_FRAMES * (sceneFrames.length - 1);

export const PaperVideo: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: C.ground }}>
      <TransitionSeries>
        {narration.scenes.flatMap((s, i) => {
          const Comp = COMPONENTS[s.id];
          const seq = (
            <TransitionSeries.Sequence
              key={`s-${s.id}`}
              durationInFrames={sceneFrames[i]}
              name={`${String(i).padStart(2, "0")} ${s.id}`}
            >
              <Comp />
            </TransitionSeries.Sequence>
          );
          if (i === narration.scenes.length - 1) return [seq];
          return [
            seq,
            <TransitionSeries.Transition
              key={`t-${s.id}`}
              presentation={fade()}
              timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
            />,
          ];
        })}
      </TransitionSeries>

      {narration.scenes.map((s, i) => (
        <Sequence
          key={`a-${s.id}`}
          from={audioStartFrames[i]}
          durationInFrames={Math.round(s.durationSec * FPS) + 6}
          layout="none"
          name={`voice ${s.id}`}
        >
          <Audio src={staticFile(s.file)} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
