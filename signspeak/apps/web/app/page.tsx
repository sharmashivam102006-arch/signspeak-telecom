"use client";

/**
 * Landing: create or join a call.
 *
 * The modality choice is framed as "how do you want to communicate", not "are
 * you deaf". The system needs to know which channel to render; it does not
 * need to know anything about the person. That framing is also why the
 * database has no disability column - see apps/api/.../models.py.
 *
 * No sign-up wall. A Deaf user should not have to create an account before
 * they can take an urgent call.
 */

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { unlock as unlockTts } from "@/lib/tts";

type Modality = "sign" | "speech";

export default function Home() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [modality, setModality] = useState<Modality>("speech");
  const [hand, setHand] = useState<"left" | "right">("right");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const go = (room: string) =>
    router.push(
      `/call/${room}?name=${encodeURIComponent(name || "Guest")}&modality=${modality}&hand=${hand}`,
    );

  async function create() {
    setBusy(true);
    setError(null);
    try {
      // Browsers silently ignore speechSynthesis until a real user gesture.
      // Unlocking here, on a click, is why audio works later.
      unlockTts();
      const room = await api.createRoom({ display_name: name || "Guest", modality });
      go(room.room_name);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function join() {
    setBusy(true);
    setError(null);
    try {
      unlockTts();
      const room = code.trim().toUpperCase();
      await api.joinRoom(room, { display_name: name || "Guest", modality });
      go(room);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh max-w-2xl flex-col justify-center gap-8 px-6 py-12">
      <header>
        <h1 className="text-4xl font-semibold tracking-tight">SignSpeak Telecom</h1>
        <p className="mt-3 text-lg text-muted">
          One call, two ways of talking. Sign and it is spoken aloud; speak and it appears as
          captions.
        </p>
        <p className="mt-2 text-sm text-muted">
          This version recognises a small, fixed set of signs. It is not a sign-language
          translator, and it is not suitable for medical or legal conversations.
        </p>
      </header>

      <section className="rounded-xl border border-border bg-surface p-5">
        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">Your name</span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Guest"
            className="w-full rounded-lg border border-border bg-bg px-3 py-2 outline-none focus:border-accent"
          />
        </label>

        <fieldset className="mb-4">
          <legend className="mb-2 text-sm font-medium">How will you communicate?</legend>
          <div className="grid gap-2 sm:grid-cols-2">
            <Choice
              selected={modality === "sign"}
              onClick={() => setModality("sign")}
              title="I'll sign"
              body="Your camera reads your signs and speaks them for the other person."
            />
            <Choice
              selected={modality === "speech"}
              onClick={() => setModality("speech")}
              title="I'll speak"
              body="Your speech becomes live captions for the other person."
            />
          </div>
        </fieldset>

        {modality === "sign" && (
          <fieldset className="mb-4">
            <legend className="mb-2 text-sm font-medium">Dominant hand</legend>
            <div className="flex gap-2">
              {(["right", "left"] as const).map((h) => (
                <button
                  key={h}
                  onClick={() => setHand(h)}
                  aria-pressed={hand === h}
                  className={`rounded-lg border px-4 py-2 text-sm capitalize ${
                    hand === h ? "border-accent text-accent" : "border-border"
                  }`}
                >
                  {h}
                </button>
              ))}
            </div>
            <p className="mt-2 text-xs text-muted">
              Roughly one signer in ten signs left-dominant. Telling the system lets it read you
              correctly instead of treating your signs as unfamiliar ones.
            </p>
          </fieldset>
        )}

        {error && (
          <p role="alert" className="mb-3 rounded border border-low/40 bg-low/10 p-2 text-sm text-low">
            {error}
          </p>
        )}

        <div className="flex flex-col gap-3 sm:flex-row">
          <button
            onClick={create}
            disabled={busy}
            className="flex-1 rounded-lg bg-accent px-4 py-3 font-semibold text-bg disabled:opacity-50"
          >
            Start a call
          </button>
          <div className="flex flex-1 gap-2">
            <input
              value={code}
              onChange={(e) => setCode(e.target.value.toUpperCase())}
              placeholder="ROOM CODE"
              aria-label="Room code"
              className="w-full rounded-lg border border-border bg-bg px-3 py-3 tracking-widest outline-none focus:border-accent"
            />
            <button
              onClick={join}
              disabled={busy || code.trim().length < 4}
              className="rounded-lg border border-border px-4 py-3 font-semibold disabled:opacity-50"
            >
              Join
            </button>
          </div>
        </div>
      </section>

      <footer className="text-sm text-muted">
        <p>
          Sign recognition runs entirely in your browser. Camera frames and hand positions never
          leave your device — only the recognised word is sent to the other person. Speech
          recognition uses your browser&apos;s built-in service, which may send audio to its
          provider.
        </p>
      </footer>
    </main>
  );
}

function Choice({
  selected,
  onClick,
  title,
  body,
}: {
  selected: boolean;
  onClick: () => void;
  title: string;
  body: string;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={selected}
      className={`rounded-lg border p-3 text-left transition ${
        selected ? "border-accent bg-accent/10" : "border-border hover:border-muted"
      }`}
    >
      <span className="block font-medium">{title}</span>
      <span className="mt-1 block text-sm text-muted">{body}</span>
    </button>
  );
}
