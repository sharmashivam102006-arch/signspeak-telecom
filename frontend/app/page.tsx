import Link from "next/link";
import { ArrowRight, Hand, Mic, Volume2, Captions, Shield, Zap, Globe, Users, Check, ChevronDown, MessageSquareText, AudioLines, Languages, Eye, HeartHandshake, Sparkles } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { LogoMark } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { Card, Badge } from "@/components/ui/card";

function SectionTitle({ kicker, title, sub }: { kicker: string; title: string; sub?: string }) {
  return (
    <div className="mx-auto mb-10 max-w-2xl text-center">
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-cyan">{kicker}</p>
      <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">{title}</h2>
      {sub && <p className="mt-3 text-muted">{sub}</p>}
    </div>
  );
}

export default function Landing() {
  return (
    <div className="bg-ink">
      <Navbar />
      {/* HERO */}
      <section className="mx-auto max-w-7xl px-4 pb-16 pt-16 sm:px-6 sm:pt-24">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div className="fade-up">
            <Badge tone="cyan"><Sparkles size={12} /> AI-powered accessible calling</Badge>
            <h1 className="mt-5 text-5xl font-bold leading-[1.05] tracking-tight sm:text-6xl">
              Communication<br />Without <span className="text-cyan">Barriers.</span>
            </h1>
            <p className="mt-5 text-lg text-muted">One Call. Two Languages. No Communication Barrier.</p>
            <p className="mt-3 text-slate-300">
              SignSpeak Telecom puts an AI communication layer inside the video call itself. One person signs, the other hears speech. One person speaks, the other reads live captions — in the same call, in real time.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/signup"><Button size="lg">Start Communicating <ArrowRight size={18} /></Button></Link>
              <a href="#how-it-works"><Button size="lg" variant="outline">Explore How It Works</Button></a>
            </div>
            <p className="mt-4 text-sm text-muted">No card required · Guest demo available · Works in your browser</p>
          </div>
          {/* Illustrative call preview — a mock, not a live call. */}
          <div className="fade-up rounded-3xl border border-slate-800 bg-panel p-4 shadow-2xl shadow-cyan/5">
            <div className="mb-3 flex items-center justify-between">
              <span className="flex items-center gap-2 text-sm font-medium">Illustration — simulated call · Room SS-4F2K</span>
              <Badge tone="slate">Illustration</Badge>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="relative aspect-video overflow-hidden rounded-2xl bg-gradient-to-br from-violet-900/60 to-slate-900">
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <Hand size={44} className="text-violet-300" />
                  <p className="mt-2 text-sm font-medium">Asha · signing</p>
                  <p className="text-xs text-muted">Hand tracking active</p>
                </div>
                <span className="absolute left-2 top-2 rounded-md bg-black/60 px-2 py-0.5 text-[11px]">Asha — Sign language</span>
                <span className="absolute bottom-2 left-2 rounded-md bg-violet/80 px-2 py-0.5 text-[11px]">Sign recognised: “Thank You” · 92%</span>
              </div>
              <div className="relative aspect-video overflow-hidden rounded-2xl bg-gradient-to-br from-cyan-900/50 to-slate-900">
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <Mic size={44} className="text-cyan" />
                  <p className="mt-2 text-sm font-medium">Rohan · speaking</p>
                  <p className="text-xs text-muted">Speech recognition active</p>
                </div>
                <span className="absolute left-2 top-2 rounded-md bg-black/60 px-2 py-0.5 text-[11px]">Rohan — Spoken language</span>
                <span className="absolute bottom-2 left-2 rounded-md bg-cyan/80 px-2 py-0.5 text-[11px] text-ink">Speaking… “You’re welcome!”</span>
              </div>
            </div>
            <div className="mt-3 rounded-xl bg-black/70 px-4 py-3 text-center">
              <p className="text-[11px] uppercase tracking-wider text-cyan">Live captions · Speech → Text</p>
              <p className="text-base font-medium">“You’re welcome — shall we meet at 5?”</p>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-muted">
              <span className="flex items-center gap-1.5"><Volume2 size={14} className="text-violet-300" /> AI voice: “Thank you” playing for Rohan</span>
              <span>12:04 elapsed</span>
            </div>
          </div>
        </div>
        <div className="mt-14 flex justify-center"><ChevronDown className="animate-bounce text-muted" /></div>
      </section>

      {/* PROBLEM / SOLUTION */}
      <section className="border-t border-slate-800/70 py-16">
        <div className="mx-auto grid max-w-7xl gap-6 px-4 sm:px-6 md:grid-cols-2">
          <Card>
            <h3 className="text-xl font-semibold text-red-300">The problem</h3>
            <p className="mt-3 text-muted">Deaf and hearing people still can&apos;t hold a natural video call together. One side signs, the other speaks — and today&apos;s tools force them to switch apps, rely on a human interpreter, or simply not call at all.</p>
          </Card>
          <Card className="border-cyan/30">
            <h3 className="text-xl font-semibold text-cyan">The SignSpeak solution</h3>
            <p className="mt-3 text-muted">Both directions live inside one call: sign language is recognised and spoken aloud, and speech is transcribed into live captions — with confidence scores, corrections, and a shared transcript.</p>
          </Card>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how-it-works" className="border-t border-slate-800/70 py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <SectionTitle kicker="How it works" title="Two pipelines. One call." sub="Both directions run at the same time, so neither person has to wait their turn to be understood." />
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <Badge tone="violet">Pipeline A · Sign → Text → Speech</Badge>
              <h3 className="mt-3 text-xl font-semibold">Signing becomes speech</h3>
              <ol className="mt-4 space-y-3 text-sm text-slate-300">
                {[["Webcam captures the signer", "Video frames are processed in the browser with OpenCV-style preprocessing."],
                  ["MediaPipe extracts hand landmarks", "21 landmarks per hand, normalised and buffered as a 30-frame sequence."],
                  ["Temporal model recognises the sign", "A PyTorch LSTM/GRU sequence classifier returns a sign and a confidence score."],
                  ["Text is spoken aloud", "Text-to-speech plays the recognised sign for the hearing participant — with mute and volume controls."]].map(([t, d], i) => (
                  <li key={t} className="flex gap-3"><span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet/20 text-xs font-bold text-violet-300">{i + 1}</span><span><strong>{t}.</strong> <span className="text-muted">{d}</span></span></li>
                ))}
              </ol>
            </Card>
            <Card>
              <Badge tone="cyan">Pipeline B · Speech → Text → Captions</Badge>
              <h3 className="mt-3 text-xl font-semibold">Speech becomes captions</h3>
              <ol className="mt-4 space-y-3 text-sm text-slate-300">
                {[["Microphone captures speech", "Audio is captured alongside the call, without blocking video."],
                  ["Whisper-family ASR transcribes", "Streaming-capable speech recognition produces interim and final results."],
                  ["Live captions appear in the call", "The Deaf participant reads what is being said as it is said."],
                  ["Captions join the transcript", "Every caption is timestamped, labelled by direction, and exportable."]].map(([t, d], i) => (
                  <li key={t} className="flex gap-3"><span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan/20 text-xs font-bold text-cyan">{i + 1}</span><span><strong>{t}.</strong> <span className="text-muted">{d}</span></span></li>
                ))}
              </ol>
            </Card>
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section id="features" className="border-t border-slate-800/70 py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <SectionTitle kicker="Features" title="Built for real conversations" />
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {[
              [Hand, "Sign recognition", "MediaPipe hand landmarks + a temporal PyTorch model, with smoothing and duplicate suppression."],
              [Captions, "Live captions", "Speech-to-text captions inside the call, with interim and final results."],
              [Volume2, "Sign-to-speech", "Recognised signs are spoken naturally, with mute, volume and voice controls."],
              [Check, "Confidence & correction", "High-confidence results flow through; uncertain ones ask you to accept, correct or repeat."],
              [MessageSquareText, "Shared transcript", "Both directions in one scrollable, copyable, exportable transcript."],
              [Shield, "Privacy by default", "No recording by default. Transcripts are stored only with consent."],
              [Zap, "Low-latency calls", "WebRTC media stays separate from AI processing, so a slow model never kills the call."],
              [Eye, "Accessibility settings", "Caption size, high-contrast mode and TTS voice — yours to tune. Recognition thresholds are server settings (SIGN_THRESHOLD_HIGH / SIGN_THRESHOLD_LOW)."],
              [Users, "Guest demo mode", "Try a call without registering. Demo predictions are always labelled Demo."],
            ].map(([Icon, t, d]) => {
              const I = Icon as typeof Hand;
              return (
                <Card key={t as string}>
                  <I size={22} className="text-cyan" />
                  <h3 className="mt-3 font-semibold">{t as string}</h3>
                  <p className="mt-1.5 text-sm text-muted">{d as string}</p>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* TECHNOLOGY */}
      <section id="technology" className="border-t border-slate-800/70 py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <SectionTitle kicker="Technology" title="A serious stack, honestly labelled" sub="Real inference where it exists, clearly-labelled demo where it doesn't — never fake AI results presented as real." />
          <div className="flex flex-wrap justify-center gap-2.5">
            {["Next.js 14", "React + TypeScript", "Tailwind CSS", "FastAPI (Python)", "LiveKit", "MediaPipe hand landmarks", "PyTorch LSTM/GRU", "Whisper ASR", "Piper TTS", "SQLite"].map(t => <Badge key={t} tone="slate">{t}</Badge>)}
          </div>
          <div className="mx-auto mt-10 grid max-w-4xl gap-5 sm:grid-cols-3">
            <Card><AudioLines size={20} className="text-cyan" /><h3 className="mt-2 font-semibold">Computer vision</h3><p className="mt-1 text-sm text-muted">OpenCV + MediaPipe hand landmarks (21 per hand), normalised into temporal sequences.</p></Card>
            <Card><Languages size={20} className="text-violet-300" /><h3 className="mt-2 font-semibold">Deep learning</h3><p className="mt-1 text-sm text-muted">PyTorch sequence classification with confidence scoring and configurable thresholds.</p></Card>
            <Card><Globe size={20} className="text-cyan" /><h3 className="mt-2 font-semibold">Real-time media</h3><p className="mt-1 text-sm text-muted">LiveKit real-time video and audio, with WebSocket signalling through the FastAPI backend.</p></Card>
          </div>
        </div>
      </section>

      {/* FUTURE VISION */}
      <section id="about" className="border-t border-slate-800/70 py-20">
        <div className="mx-auto max-w-4xl px-4 text-center sm:px-6">
          <SectionTitle kicker="Future vision" title="Beyond one call" />
          <p className="text-muted">More signs, more languages, group calls, and on-device inference — so accessible communication works anywhere, for anyone, without an interpreter in the middle. SignSpeak Telecom starts with six signs and one honest promise: what the AI knows, it shows; what it doesn&apos;t, it asks.</p>
          <div className="mt-8 flex items-center justify-center gap-2 text-sm text-muted"><HeartHandshake size={18} className="text-cyan" /> Built for Deaf, hard-of-hearing and hearing people — together.</div>
          <div className="mt-10"><Link href="/signup"><Button size="lg">Start Communicating <ArrowRight size={18} /></Button></Link></div>
        </div>
      </section>

      <footer className="border-t border-slate-800/70 py-10">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 sm:flex-row sm:px-6">
          <div className="flex items-center gap-2"><LogoMark size={28} /><span className="text-sm font-semibold">SignSpeak Telecom</span><span className="text-xs text-muted">— One Call. Two Languages. No Communication Barrier.</span></div>
          <p className="text-xs text-muted">© 2026 SignSpeak Telecom. No video or audio is recorded by default.</p>
        </div>
      </footer>
    </div>
  );
}
