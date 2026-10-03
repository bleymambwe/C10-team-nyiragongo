import { loadFont } from "@remotion/fonts";
import { staticFile } from "remotion";

/**
 * Load IBM Plex from files in public/fonts rather than from Google Fonts.
 *
 * @remotion/google-fonts fetches at render time, and a transient
 * "NetworkError: A network error occurred." inside headless Chrome killed
 * whole render segments. Rendering now has no network dependency at all,
 * matching the rest of the project (Kokoro and the model weights are local
 * too).
 *
 * IBM Plex Sans ships as a variable font, so one file covers 400/600/700 --
 * the three weight entries below intentionally point at the same file.
 */
export const loadFonts = () =>
  Promise.all([
    loadFont({ family: "IBM Plex Sans", url: staticFile("fonts/IBMPlexSans-400.woff2"), weight: "400" }),
    loadFont({ family: "IBM Plex Sans", url: staticFile("fonts/IBMPlexSans-600.woff2"), weight: "600" }),
    loadFont({ family: "IBM Plex Sans", url: staticFile("fonts/IBMPlexSans-700.woff2"), weight: "700" }),
    loadFont({ family: "IBM Plex Mono", url: staticFile("fonts/IBMPlexMono-400.woff2"), weight: "400" }),
    loadFont({ family: "IBM Plex Mono", url: staticFile("fonts/IBMPlexMono-500.woff2"), weight: "500" }),
    loadFont({ family: "IBM Plex Mono", url: staticFile("fonts/IBMPlexMono-600.woff2"), weight: "600" }),
    loadFont({ family: "IBM Plex Serif", url: staticFile("fonts/IBMPlexSerif-400.woff2"), weight: "400" }),
  ]);
