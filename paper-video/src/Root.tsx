import React from "react";
import { Composition, Folder } from "remotion";
import { loadFonts } from "./fonts";

import "./index.css";
import narration from "./narration.json";
import { FPS, PaperVideo, sceneFrames, totalFrames } from "./Video";
import { S00Title, S01Problem, S02TwoGeometries, S03Certificate, S04Regions, S05Theorem } from "./scenes/Part1";
import { S06Leakage, S07LowRank, S08Gpt2, S09NotCircular, S10Independence, S11RealData } from "./scenes/Part2";
import { S12Failures, S13Limitations, S14Safety, S15Next, S16Close } from "./scenes/Part3";
import { S12Trained, S13Budget } from "./scenes/Part4";

loadFonts();

const SCENES: Record<string, React.FC> = {
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

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="PaperVideo"
        component={PaperVideo}
        durationInFrames={totalFrames}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Folder name="Scenes">
        {narration.scenes.map((s, i) => (
          <Composition
            key={s.id}
            id={`${String(i).padStart(2, "0")}-${s.id.replace(/_/g, "-")}`}
            component={SCENES[s.id]}
            durationInFrames={sceneFrames[i]}
            fps={FPS}
            width={1920}
            height={1080}
          />
        ))}
      </Folder>
    </>
  );
};
