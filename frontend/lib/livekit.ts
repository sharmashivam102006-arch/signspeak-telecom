// LiveKit calling for SignSpeak Telecom.
//
// The call itself is LiveKit (livekit-client). The legacy P2P client
// (lib/webrtc.ts) was deleted 2026-11-18 — there is no P2P fallback.
// Media (camera/mic) is published to a LiveKit room with a
// server-issued token (GET /api/rooms/{id}/token); there is no P2P fallback
// for media. If the token endpoint says LiveKit is not configured, the
// caller must show that error — never silently fall back to fake calling.
//
// Track attachment: this module exposes the raw livekit-client Track
// objects. The UI attaches them to <video>/<audio> elements with
// track.attach(element) and detaches on unmount/leave. No fake video,
// no placeholder avatars pretending to be video, no simulated participants.
import {
  Room,
  RoomEvent,
  Track,
  createLocalTracks,
  VideoPresets,
  type LocalAudioTrack,
  type LocalTrack,
  type LocalVideoTrack,
  type Participant,
  type RemoteAudioTrack,
  type RemoteVideoTrack,
  type RemoteParticipant,
  type RemoteTrack,
  type RemoteTrackPublication,
  type RoomOptions,
} from "livekit-client";

export type ParticipantInfo = {
  identity: string;
  name: string;
  isLocal: boolean;
  isCameraEnabled: boolean;
  isMicrophoneEnabled: boolean;
  isSpeaking: boolean;
  hasVideo: boolean;
  hasAudio: boolean;
};

export type RemoteTrackInfo = {
  participant: RemoteParticipant;
  videoTrack: RemoteVideoTrack | null;
  audioTrack: RemoteAudioTrack | null;
};

export type LiveKitHandlers = {
  onLocalTracks: (video: LocalVideoTrack | null, audio: LocalAudioTrack | null) => void;
  onRemoteTracks: (info: RemoteTrackInfo) => void;
  onRemoteLeft: (identity: string) => void;
  onParticipantsChanged: (list: ParticipantInfo[]) => void;
  onConnectionState: (state: string) => void;
  onError: (msg: string) => void;
};

export type MediaAvailability = {
  camera: boolean;
  microphone: boolean;
  error: string;
};

/** Display name for a participant: LiveKit name, else the identity. */
export function displayName(p: Participant): string {
  return p.name || p.identity;
}

export class LiveKitCall {
  private room: Room | null = null;
  private localTracks: LocalTrack[] = [];
  private speaking = new Set<string>();
  private media: MediaAvailability = { camera: false, microphone: false, error: "" };

  constructor(private handlers: LiveKitHandlers) {}

  get isConnected(): boolean {
    return !!this.room && this.room.state === "connected";
  }

  get mediaAvailability(): MediaAvailability {
    return this.media;
  }

  getLocalParticipant(): Participant | null {
    return this.room?.localParticipant ?? null;
  }

  getLocalVideoTrack(): LocalVideoTrack | null {
    return (this.localTracks.find((t) => t.kind === Track.Kind.Video) as LocalVideoTrack | undefined) ?? null;
  }

  getLocalAudioTrack(): LocalAudioTrack | null {
    return (this.localTracks.find((t) => t.kind === Track.Kind.Audio) as LocalAudioTrack | undefined) ?? null;
  }

  /** A MediaStream built from the local published tracks (for MediaPipe + preview). */
  getLocalMediaStream(): MediaStream | null {
    const msTracks = this.localTracks
      .map((t) => t.mediaStreamTrack)
      .filter((t): t is MediaStreamTrack => !!t);
    return msTracks.length > 0 ? new MediaStream(msTracks) : null;
  }

  async connect(url: string, token: string): Promise<void> {
    const opts: RoomOptions = { adaptiveStream: true, dynacast: true };
    // Optional TURN relay (dev/sandbox only): when the environment blocks
    // UDP entirely, direct ICE cannot work and media must relay over TCP.
    // Set NEXT_PUBLIC_TURN_URL (e.g. turn:127.0.0.1:3478?transport=tcp)
    // plus NEXT_PUBLIC_TURN_USERNAME / NEXT_PUBLIC_TURN_CREDENTIAL.
    // When unset, LiveKit's own ICE servers from the JoinResponse are used.
    const turnUrl = process.env.NEXT_PUBLIC_TURN_URL;
    const turnUser = process.env.NEXT_PUBLIC_TURN_USERNAME;
    const turnPass = process.env.NEXT_PUBLIC_TURN_CREDENTIAL;
    const connectOpts =
      turnUrl && turnUser && turnPass
        ? {
            rtcConfig: {
              iceServers: [{ urls: turnUrl, username: turnUser, credential: turnPass }],
            },
          }
        : undefined;
    const room = new Room(opts);
    this.room = room;
    this.bindEvents(room);
    this.handlers.onConnectionState("Connecting…");
    await room.connect(url, token, connectOpts);
    this.handlers.onConnectionState("Connected");

    // Publish camera + mic. If getUserMedia is denied, join anyway with
    // whatever could be acquired and report the gap honestly — the call
    // must not crash because a device is unavailable.
    await this.acquireAndPublish();
    this.emitParticipants();
  }

  private async acquireAndPublish(): Promise<void> {
    const room = this.room;
    if (!room) return;
    let tracks: LocalTrack[] = [];
    try {
      tracks = await createLocalTracks({
        audio: true,
        video: { resolution: VideoPresets.h720 },
      });
    } catch (e) {
      // Full request failed (often a permission denial). Retry each
      // device separately so one denial does not kill the other.
      const msg = e instanceof Error ? e.message : String(e);
      this.media.error = msg;
      tracks = [];
      for (const kind of ["audio", "video"] as const) {
        try {
          const single = await createLocalTracks(
            kind === "audio" ? { audio: true } : { video: { resolution: VideoPresets.h720 } },
          );
          tracks.push(...single);
        } catch {
          // That device stays unavailable; the other may still work.
        }
      }
    }
    const hasVideo = tracks.some((t) => t.kind === Track.Kind.Video);
    const hasAudio = tracks.some((t) => t.kind === Track.Kind.Audio);
    if (!hasVideo || !hasAudio) {
      const missing = [
        !hasVideo ? "camera" : null,
        !hasAudio ? "microphone" : null,
      ]
        .filter(Boolean)
        .join(" and ");
      this.media = {
        camera: hasVideo,
        microphone: hasAudio,
        error:
          this.media.error ||
          `Could not access your ${missing}. Check browser permissions — you are still in the room, but the other participant cannot see/hear you on that device.`,
      };
      if (this.media.error) this.handlers.onError(this.media.error);
    } else {
      this.media = { camera: true, microphone: true, error: "" };
    }
    for (const track of tracks) {
      try {
        await room.localParticipant.publishTrack(track);
        this.localTracks.push(track);
      } catch (e) {
        this.handlers.onError(`Could not publish ${track.kind} track: ${e instanceof Error ? e.message : String(e)}`);
        track.stop();
      }
    }
    this.handlers.onLocalTracks(this.getLocalVideoTrack(), this.getLocalAudioTrack());
    this.emitParticipants();
  }

  /** Mute/unmute: enables/disables the published mic track in LiveKit. */
  async setMicrophoneEnabled(enabled: boolean): Promise<boolean> {
    if (!this.room) return false;
    await this.room.localParticipant.setMicrophoneEnabled(enabled);
    this.emitParticipants();
    return this.room.localParticipant.isMicrophoneEnabled;
  }

  /** Camera on/off: enables/disables the published camera track in LiveKit. */
  async setCameraEnabled(enabled: boolean): Promise<boolean> {
    if (!this.room) return false;
    await this.room.localParticipant.setCameraEnabled(enabled);
    // Re-acquiring the camera creates a fresh track: refresh our handles.
    this.refreshLocalTracks();
    this.handlers.onLocalTracks(this.getLocalVideoTrack(), this.getLocalAudioTrack());
    this.emitParticipants();
    return this.room.localParticipant.isCameraEnabled;
  }

  async toggleMicrophone(): Promise<boolean> {
    const current = this.room?.localParticipant.isMicrophoneEnabled ?? false;
    return this.setMicrophoneEnabled(!current);
  }

  async toggleCamera(): Promise<boolean> {
    const current = this.room?.localParticipant.isCameraEnabled ?? false;
    return this.setCameraEnabled(!current);
  }

  private refreshLocalTracks(): void {
    const room = this.room;
    if (!room) return;
    const pubs = [
      ...room.localParticipant.videoTrackPublications.values(),
      ...room.localParticipant.audioTrackPublications.values(),
    ];
    this.localTracks = pubs
      .map((p) => p.track)
      .filter((t): t is LocalTrack => !!t);
  }

  private remoteTrackInfo(p: RemoteParticipant): RemoteTrackInfo {
    let videoTrack: RemoteVideoTrack | null = null;
    let audioTrack: RemoteAudioTrack | null = null;
    p.videoTrackPublications.forEach((pub) => {
      if (pub.track && pub.kind === Track.Kind.Video) videoTrack = pub.track as RemoteVideoTrack;
    });
    p.audioTrackPublications.forEach((pub) => {
      if (pub.track && pub.kind === Track.Kind.Audio) audioTrack = pub.track as RemoteAudioTrack;
    });
    return { participant: p, videoTrack, audioTrack };
  }

  private snapshotParticipants(): ParticipantInfo[] {
    const room = this.room;
    if (!room) return [];
    const list: ParticipantInfo[] = [];
    const lp = room.localParticipant;
    list.push({
      identity: lp.identity,
      name: displayName(lp),
      isLocal: true,
      isCameraEnabled: lp.isCameraEnabled,
      isMicrophoneEnabled: lp.isMicrophoneEnabled,
      isSpeaking: this.speaking.has(lp.identity),
      hasVideo: lp.videoTrackPublications.size > 0,
      hasAudio: lp.audioTrackPublications.size > 0,
    });
    room.remoteParticipants.forEach((p) => {
      list.push({
        identity: p.identity,
        name: displayName(p),
        isLocal: false,
        isCameraEnabled: p.isCameraEnabled,
        isMicrophoneEnabled: p.isMicrophoneEnabled,
        isSpeaking: this.speaking.has(p.identity),
        hasVideo: p.videoTrackPublications.size > 0,
        hasAudio: p.audioTrackPublications.size > 0,
      });
    });
    return list;
  }

  private emitParticipants(): void {
    this.handlers.onParticipantsChanged(this.snapshotParticipants());
  }

  private bindEvents(room: Room): void {
    room.on(RoomEvent.ConnectionStateChanged, () => {
      const state = room.state;
      this.handlers.onConnectionState(
        state === "connected"
          ? "Connected"
          : state === "reconnecting"
            ? "Reconnecting…"
            : state === "disconnected"
              ? "Disconnected"
              : state,
      );
      this.emitParticipants();
    });
    room.on(RoomEvent.ParticipantConnected, () => this.emitParticipants());
    room.on(RoomEvent.ParticipantDisconnected, (p: RemoteParticipant) => {
      this.speaking.delete(p.identity);
      this.handlers.onRemoteLeft(p.identity);
      this.emitParticipants();
    });
    room.on(
      RoomEvent.TrackSubscribed,
      (track: RemoteTrack, _pub: RemoteTrackPublication, participant: RemoteParticipant) => {
        void track;
        this.handlers.onRemoteTracks(this.remoteTrackInfo(participant));
        this.emitParticipants();
      },
    );
    room.on(
      RoomEvent.TrackUnsubscribed,
      (_track: RemoteTrack, _pub: RemoteTrackPublication, participant: RemoteParticipant) => {
        this.handlers.onRemoteTracks(this.remoteTrackInfo(participant));
        this.emitParticipants();
      },
    );
    room.on(RoomEvent.TrackMuted, () => this.emitParticipants());
    room.on(RoomEvent.TrackUnmuted, () => this.emitParticipants());
    room.on(RoomEvent.LocalTrackPublished, () => {
      this.refreshLocalTracks();
      this.handlers.onLocalTracks(this.getLocalVideoTrack(), this.getLocalAudioTrack());
      this.emitParticipants();
    });
    room.on(RoomEvent.ActiveSpeakersChanged, (speakers: Participant[]) => {
      this.speaking = new Set(speakers.map((s) => s.identity));
      this.emitParticipants();
    });
    room.on(RoomEvent.Disconnected, () => {
      this.handlers.onConnectionState("Disconnected");
      this.emitParticipants();
    });
  }

  /** Leave the call: unpublish + stop local tracks, then disconnect. */
  async disconnect(): Promise<void> {
    const room = this.room;
    this.room = null;
    if (!room) return;
    for (const track of this.localTracks) {
      try {
        room.localParticipant.unpublishTrack(track);
      } catch {
        // Already unpublished; ignore.
      }
      track.stop();
    }
    this.localTracks = [];
    try {
      await room.disconnect();
    } catch {
      // Disconnect is best-effort on teardown.
    }
  }
}
