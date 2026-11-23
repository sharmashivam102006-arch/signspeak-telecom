"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Phone, Plus, LogIn, Copy, Check, User, Settings as SettingsIcon, Activity, MessageSquareText, Clock } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Card, Badge } from "@/components/ui/card";
import { api, checkHealth, getToken, getUser } from "@/lib/api";
import { loadCalls, loadRecentTranscripts, loadSettings, saveSettings, Settings, CallRecord } from "@/lib/store";

/** Normalise a room code: trim, uppercase, drop an optional "SS-" prefix and dashes. */
function normaliseCode(raw: string): string {
  return raw.trim().toUpperCase().replace(/^SS-?/, "").replace(/[-\s]/g, "");
}

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<{ name?: string; email?: string } | null>(null);
  const [joinCode, setJoinCode] = useState("");
  const [joinError, setJoinError] = useState("");
  const [startError, setStartError] = useState("");
  const [copied, setCopied] = useState(false);
  const [settings, setSettings] = useState<Settings>(loadSettings());
  const [health, setHealth] = useState<{ status: string; whisper?: string; tts?: string; livekit_mode?: string } | null>(null);
  const [vocab, setVocab] = useState<string[] | null>(null);
  const [vocabError, setVocabError] = useState(false);
  const [calls, setCalls] = useState<CallRecord[]>([]);
  const [transcripts, setTranscripts] = useState<ReturnType<typeof loadRecentTranscripts>>([]);
  const [rooms, setRooms] = useState<{ id: string; code?: string; created_at?: string }[]>([]);

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); return; }
    setUser(getUser());
    setCalls(loadCalls());
    setTranscripts(loadRecentTranscripts());
    checkHealth().then(setHealth);
    api<{ vocabulary?: string[] }>("/api/ai/sign/vocabulary").then(d => setVocab(d.vocabulary ?? null)).catch(() => setVocabError(true));
    // GET /api/rooms returns a bare JSON array (with {rooms: [...]} kept as compat).
    api<unknown>("/api/rooms").then(d => setRooms(Array.isArray(d) ? d : (d as { rooms?: { id: string; code?: string }[] }).rooms ?? [])).catch(() => { /* backend may not list rooms yet */ });
  }, [router]);

  const updateSettings = (patch: Partial<Settings>) => {
    const next = { ...settings, ...patch };
    setSettings(next); saveSettings(next);
  };

  const startCall = async () => {
    // Create the room server-side first and use the returned code — a
    // locally invented id would point at a room that does not exist.
    setStartError("");
    try {
      const room = await api<{ code: string }>("/api/rooms", { method: "POST", body: JSON.stringify({ name: "SignSpeak Call" }) });
      if (!room.code) throw new Error("Server did not return a room code");
      router.push(`/room/${room.code}`);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : "Could not create a room — is the backend running?");
    }
  };
  const joinCall = async () => {
    const code = normaliseCode(joinCode);
    if (!code) return setJoinError("Enter a room code, e.g. ABC123 or SS-ABC123.");
    if (!/^[A-Z0-9]{6}$/.test(code)) return setJoinError("Room codes are 6 letters/digits — use the code from your invitation link (SS- prefix optional).");
    setJoinError("");
    try {
      // Validate against the backend first; only navigate on success.
      await api(`/api/rooms/${code}`);
      router.push(`/room/${code}`);
    } catch {
      setJoinError(`No room found for code ${code}. Check the code and try again.`);
    }
  };
  const copyLink = async (id: string) => {
    await navigator.clipboard.writeText(`${window.location.origin}/room/${id}`);
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  };

  if (!user) return <div className="min-h-screen bg-ink" />;
  return (
    <div className="min-h-screen bg-ink">
      <Navbar />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold sm:text-3xl">Welcome, {user.name || "there"}</h1>
            <p className="mt-1 text-sm text-muted">Start a call, or join one with a room code.</p>
          </div>
          <Badge tone={!health ? "slate" : health.status === "ok" ? "green" : "red"}><Activity size={12} /> {health ? (health.status === "ok" ? "Service online" : "Service offline") : "Checking…"}</Badge>
        </div>

        <div className="grid gap-5 lg:grid-cols-3">
          <div className="space-y-5 lg:col-span-2">
            <Card>
              <h2 className="font-semibold">Start or join a call</h2>
              <div className="mt-4 flex flex-wrap gap-3">
                <Button size="lg" onClick={startCall}><Plus size={18} /> Start New Call</Button>
                <div className="flex flex-1 gap-2">
                  <Input value={joinCode} onChange={e => setJoinCode(e.target.value)} placeholder="Room code, e.g. ABC123" onKeyDown={e => e.key === "Enter" && joinCall()} />
                  <Button variant="secondary" onClick={joinCall}><LogIn size={16} /> Join Call</Button>
                </div>
              </div>
              {startError && <p role="alert" className="mt-2 text-sm text-red-300">{startError}</p>}
              {joinError && <p role="alert" className="mt-2 text-sm text-red-300">{joinError}</p>}
              {rooms.length > 0 && (
                <div className="mt-4 space-y-2">
                  <p className="text-sm font-medium text-muted">Your rooms on the server</p>
                  {rooms.map(r => (
                    <div key={r.id} className="flex items-center justify-between rounded-lg border border-slate-800 px-3 py-2 text-sm">
                      <span className="flex items-center gap-2"><Phone size={14} className="text-cyan" /> {r.code ?? r.id}</span>
                      <span className="flex gap-1">
                        <Button size="sm" variant="ghost" onClick={() => copyLink(r.code ?? r.id)}>{copied ? <Check size={14} /> : <Copy size={14} />} Copy link</Button>
                        <Button size="sm" variant="outline" onClick={() => router.push(`/room/${r.code ?? r.id}`)}>Join</Button>
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </Card>

            <Card>
              <h2 className="flex items-center gap-2 font-semibold"><Clock size={16} className="text-cyan" /> Recent calls</h2>
              {calls.length === 0 ? (
                <p className="mt-3 text-sm text-muted">No calls yet. Calls you finish on this device appear here.</p>
              ) : (
                <ul className="mt-3 space-y-2 text-sm">
                  {calls.map((c, i) => (
                    <li key={i} className="flex justify-between rounded-lg border border-slate-800 px-3 py-2">
                      <span>Room {c.roomId}</span>
                      <span className="text-muted">{new Date(c.startedAt).toLocaleString()}{c.durationSec != null ? ` · ${Math.round(c.durationSec / 60)} min` : ""}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>

            <Card>
              <h2 className="flex items-center gap-2 font-semibold"><MessageSquareText size={16} className="text-violet-300" /> Recent transcripts</h2>
              {transcripts.length === 0 ? (
                <p className="mt-3 text-sm text-muted">No transcripts yet. They stay on this device unless you save them to the server.</p>
              ) : (
                <ul className="mt-3 space-y-2 text-sm">
                  {transcripts.map(t => (
                    <li key={t.roomId} className="flex justify-between rounded-lg border border-slate-800 px-3 py-2">
                      <span>Room {t.roomId} · {t.entries.length} message{t.entries.length === 1 ? "" : "s"}</span>
                      <Button size="sm" variant="ghost" onClick={() => router.push(`/room/${t.roomId}`)}>Open room</Button>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>

          <div className="space-y-5">
            <Card>
              <h2 className="flex items-center gap-2 font-semibold"><User size={16} className="text-cyan" /> Profile</h2>
              <p className="mt-3 text-sm"><span className="text-muted">Name:</span> {user.name || "—"}</p>
              <p className="mt-1 text-sm"><span className="text-muted">Account:</span> {!user.email || user.email.endsWith("@guest.signspeak.local") ? "Guest" : user.email}</p>
            </Card>

            <Card>
              <h2 className="flex items-center gap-2 font-semibold"><Activity size={16} className="text-success" /> Services</h2>
              <ul className="mt-3 space-y-2 text-sm">
                <li className="flex justify-between"><span>Server</span><Badge tone={!health ? "slate" : health.status === "ok" ? "green" : "red"}>{health ? (health.status === "ok" ? "Online" : "Offline") : "Checking…"}</Badge></li>
                <li className="flex justify-between"><span>Video calls</span><Badge tone={!health ? "slate" : health.livekit_mode === "livekit" ? "green" : "amber"}>{!health ? "Checking…" : health.livekit_mode === "livekit" ? "Configured" : "Not configured"}</Badge></li>
                <li className="flex justify-between"><span>Sign vocabulary</span>
                  {vocab ? <Badge tone="green">{vocab.length} signs</Badge> : vocabError ? <Badge tone="amber">Unavailable</Badge> : <Badge tone="slate">Checking…</Badge>}
                </li>
                <li className="flex justify-between"><span>Live captions</span>{!health ? <Badge tone="slate">Checking…</Badge> : health.whisper === "installed" ? <Badge tone="green">Server (Whisper)</Badge> : <Badge tone="cyan">Browser</Badge>}</li>
                <li className="flex justify-between"><span>Spoken signs</span>{!health ? <Badge tone="slate">Checking…</Badge> : health.tts && health.tts !== "not-installed" ? <Badge tone="green">Server voice</Badge> : <Badge tone="cyan">Browser</Badge>}</li>
              </ul>
              {vocab && <p className="mt-3 text-xs text-muted">Signs: {vocab.join(", ")}</p>}
              {vocabError && <p className="mt-3 text-xs text-muted">The server did not answer. Demo signs still work; refresh to check again.</p>}
            </Card>

            <Card>
              <h2 className="flex items-center gap-2 font-semibold"><SettingsIcon size={16} className="text-violet-300" /> Accessibility settings</h2>
              <div className="mt-4 space-y-4 text-sm">
                <div>
                  <Label>Caption size</Label>
                  <div className="mt-1 flex gap-1.5">
                    {(["small", "medium", "large"] as const).map(s => (
                      <Button key={s} size="sm" variant={settings.captionSize === s ? "primary" : "outline"} onClick={() => updateSettings({ captionSize: s })}>{s}</Button>
                    ))}
                  </div>
                </div>
                <label className="flex items-center justify-between gap-2">
                  <span>High contrast captions</span>
                  <input type="checkbox" checked={settings.highContrast} onChange={e => updateSettings({ highContrast: e.target.checked })} className="h-4 w-4 accent-cyan" />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span>Mute generated speech (TTS)</span>
                  <input type="checkbox" checked={settings.ttsMuted} onChange={e => updateSettings({ ttsMuted: e.target.checked })} className="h-4 w-4 accent-cyan" />
                </label>
                <div>
                  <Label>TTS volume: {Math.round(settings.ttsVolume * 100)}%</Label>
                  <input type="range" min={0} max={1} step={0.1} value={settings.ttsVolume} onChange={e => updateSettings({ ttsVolume: Number(e.target.value) })} className="w-full accent-cyan" />
                </div>
                <div>
                  <Label>Sign language</Label>
                  <select value={settings.language} onChange={e => updateSettings({ language: e.target.value })} className="w-full rounded-xl border border-slate-700 bg-panel px-3 py-2 text-sm">
                    <option value="en-IN">English (India)</option>
                    <option value="en-US">English (US)</option>
                    <option value="hi-IN">Hindi</option>
                  </select>
                </div>
                <div>
                  <Label>Accept sign above: {settings.signThresholdHigh}% confidence</Label>
                  <input type="range" min={50} max={99} value={settings.signThresholdHigh} onChange={e => updateSettings({ signThresholdHigh: Number(e.target.value) })} className="w-full accent-success" />
                  <Label>Ask to repeat below: {settings.signThresholdLow}% confidence</Label>
                  <input type="range" min={30} max={90} value={settings.signThresholdLow} onChange={e => updateSettings({ signThresholdLow: Number(e.target.value) })} className="w-full accent-amber-400" />
                </div>
                <p className="text-xs text-muted">Settings are saved in this browser and apply in the call room.</p>
              </div>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
