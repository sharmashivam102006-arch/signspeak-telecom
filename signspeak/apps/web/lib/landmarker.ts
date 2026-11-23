/**
 * MediaPipe Tasks wrapper.
 *
 * Three landmarkers, not Holistic. Holistic is deprecated and its hand
 * tracking is noticeably worse; the Tasks API costs a little more CPU for
 * three graphs but gives better landmarks, which is the input to everything
 * downstream. Pay it.
 *
 * Handedness needs care. MediaPipe labels hands from the *image's* point of
 * view, and the self-view is mirrored so the user sees themselves the right
 * way round. So MediaPipe's "Left" is the user's right hand. Getting this
 * backwards swaps the dominant and non-dominant channels for every frame, and
 * the failure is subtle: the model still works a bit, just badly, and you lose
 * a week.
 */

import type { Landmarks, RawFrame } from "./features";

const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODELS = {
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
};

export interface LandmarkerOptions {
  /** Face mesh costs ~3ms/frame and only matters for signs distinguished by
   * non-manual markers. Off by default; turn it on once your vocabulary needs
   * it, and say in the report which condition the numbers came from. */
  useFace?: boolean;
  delegate?: "GPU" | "CPU";
}

export class MultiLandmarker {
  private hand: any;
  private pose: any;
  private face: any | null = null;

  private constructor(hand: any, pose: any, face: any | null) {
    this.hand = hand;
    this.pose = pose;
    this.face = face;
  }

  static async create(opts: LandmarkerOptions = {}): Promise<MultiLandmarker> {
    const { FilesetResolver, HandLandmarker, PoseLandmarker, FaceLandmarker } = await import(
      "@mediapipe/tasks-vision"
    );
    const fileset = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
    const delegate = opts.delegate ?? "GPU";

    const [hand, pose, face] = await Promise.all([
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS.hand, delegate },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }),
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS.pose, delegate },
        runningMode: "VIDEO",
        numPoses: 1,
      }),
      opts.useFace
        ? FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: MODELS.face, delegate },
            runningMode: "VIDEO",
            numFaces: 1,
          })
        : Promise.resolve(null),
    ]);

    return new MultiLandmarker(hand, pose, face);
  }

  detect(video: HTMLVideoElement, tMs: number): RawFrame {
    const h = this.hand.detectForVideo(video, tMs);
    const p = this.pose.detectForVideo(video, tMs);
    const f = this.face ? this.face.detectForVideo(video, tMs) : null;

    let leftHand: Landmarks | null = null;
    let rightHand: Landmarks | null = null;

    for (let i = 0; i < (h.landmarks?.length ?? 0); i++) {
      const lm = h.landmarks[i] as Landmarks;
      const label = h.handednesses?.[i]?.[0]?.categoryName;
      // Mirrored self-view: MediaPipe's "Left" is the user's right hand.
      if (label === "Left") rightHand = lm;
      else leftHand = lm;
    }

    return {
      pose: p.landmarks?.[0] ?? null,
      leftHand,
      rightHand,
      face: f?.faceLandmarks?.[0] ?? null,
    };
  }

  close() {
    this.hand?.close?.();
    this.pose?.close?.();
    this.face?.close?.();
  }
}
