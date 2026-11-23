/**
 * LiveKit wrapper.
 *
 * Two decisions worth knowing about.
 *
 * 1. Captions travel on the LiveKit data channel, not the app's WebSocket.
 *    The data channel already follows the same connection as the media, so a
 *    caption cannot arrive before the video it belongs to, and it keeps
 *    working if the API server is restarting. The WebSocket carries presence
 *    and clock sync, where ordering against media does not matter.
 *
 * 2. Captions go out RELIABLE, not lossy. LiveKit's lossy mode is right for
 *    data you resend constantly (cursor positions); a caption is sent once and
 *    is the only access one participant has to what was said. Dropping it is
 *    not a hiccup, it is the person missing a sentence.
 */

import type { Room, RemoteParticipant, RoomEvent as LKRoomEvent } from "livekit-client";

export interface CaptionMessage {
  kind: "caption";
  source: "sign" | "speech" | "text";
  text: string;
  displayName: string;
  confidence?: number;
  action?: "speak" | "confirm" | "reject";
  startedAtMs: number;
  isFinal: boolean;
  utteranceId?: string;
}

export type RoomEvent =
  | { type: "data"; identity: string; payload: CaptionMessage }
  | { type: "participant"; identity: string; joined: boolean }
  | { type: "remoteVideo"; identity: string; stream: MediaStream }
  | { type: "connectionQuality"; quality: string }
  | { type: "disconnected"; reason?: string };

export interface RoomClientOptions {
  url: string;
  token: string;
  identity: string;
  onEvent: (e: RoomEvent) => void;
}

export class RoomClient {
  private room: Room | null = null;
  readonly identity: string;

  constructor(private opts: RoomClientOptions) {
    this.identity = opts.identity;
  }

  async connect(publish: { audio: boolean; video: boolean }) {
    const {
      Room,
      RoomEvent: LK,
      Track,
    } = await import("livekit-client");

    const room = new Room({
      adaptiveStream: true,
      dynacast: true,
      videoCaptureDefaults: {
        resolution: { width: 1280, height: 720, frameRate: 30 },
        // Sign recognition needs the hands sharp and the motion smooth. 720p30
        // is the floor; adaptive streaming may drop the *transmitted* quality
        // on a bad link, but the LOCAL track stays full quality because
        // recognition runs on the local camera, not on what the network
        // delivered. That separation is why recognition survives a weak
        // connection.
      },
    });

    room.on(LK.DataReceived, (payload: Uint8Array, participant?: RemoteParticipant) => {
      try {
        const msg = JSON.parse(new TextDecoder().decode(payload));
        this.opts.onEvent({
          type: "data",
          identity: participant?.identity ?? "unknown",
          payload: msg,
        });
      } catch {
        /* ignore malformed */
      }
    });

    room.on(LK.ParticipantConnected, (p: RemoteParticipant) =>
      this.opts.onEvent({ type: "participant", identity: p.identity, joined: true }),
    );
    room.on(LK.ParticipantDisconnected, (p: RemoteParticipant) =>
      this.opts.onEvent({ type: "participant", identity: p.identity, joined: false }),
    );
    room.on(LK.TrackSubscribed, (track: any, _pub: any, participant: RemoteParticipant) => {
      if (track.kind === Track.Kind.Video) {
        this.opts.onEvent({
          type: "remoteVideo",
          identity: participant.identity,
          stream: new MediaStream([track.mediaStreamTrack]),
        });
      }
    });
    room.on(LK.Disconnected, (reason?: any) =>
      this.opts.onEvent({ type: "disconnected", reason: String(reason ?? "") }),
    );

    await room.connect(this.opts.url, this.opts.token);
    await room.localParticipant.setCameraEnabled(publish.video);
    await room.localParticipant.setMicrophoneEnabled(publish.audio);
    this.room = room;
  }

  localVideoStream(): MediaStream | null {
    const pub = this.room?.localParticipant.getTrackPublication(
      // @ts-expect-error runtime enum
      "camera",
    );
    const track = (pub as any)?.track?.mediaStreamTrack;
    return track ? new MediaStream([track]) : null;
  }

  async publishCaption(msg: CaptionMessage) {
    if (!this.room) return;
    const { DataPacket_Kind } = await import("livekit-client");
    await this.room.localParticipant.publishData(
      new TextEncoder().encode(JSON.stringify(msg)),
      // RELIABLE: a caption is sent once and is somebody's only access to what
      // was said. Never use LOSSY for this.
      { reliable: true } as any,
    );
  }

  async setMicrophone(on: boolean) {
    await this.room?.localParticipant.setMicrophoneEnabled(on);
  }

  async setCamera(on: boolean) {
    // Turning the camera off stops sign recognition too. The UI must say so -
    // silently stopping the one participant's ability to talk is the worst
    // failure mode this product has.
    await this.room?.localParticipant.setCameraEnabled(on);
  }

  /** Publish synthesised speech as a SEPARATE audio track, so the receiver can
   * label it as machine-spoken rather than pass it off as a human voice. */
  async publishSynthesizedAudio(track: MediaStreamTrack) {
    if (!this.room) return;
    await this.room.localParticipant.publishTrack(track, {
      name: "signspeak-tts",
      source: undefined,
    } as any);
  }

  disconnect() {
    this.room?.disconnect();
    this.room = null;
  }
}
