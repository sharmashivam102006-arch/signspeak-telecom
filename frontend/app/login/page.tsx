"use client";
import { Suspense, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { LogIn, Sparkles } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { login, guestLogin, setSession, ApiError } from "@/lib/api";

/** Safe post-login destination: only a same-site relative path is
 *  honoured ("/room/ABC123"); anything else falls back to /dashboard,
 *  so a crafted ?next= can never redirect off-site. */
function safeNext(raw: string | null): string {
  if (raw && raw.startsWith("/") && !raw.startsWith("//")) return raw;
  return "/dashboard";
}

// The form reads ?next= via useSearchParams, which Next.js requires to
// sit inside a <Suspense> boundary at build time (see LoginPage below).
function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const next = safeNext(searchParams.get("next"));
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError("");
    if (!/^\S+@\S+\.\S+$/.test(email)) return setError("Enter a valid email address.");
    if (password.length < 6) return setError("Password must be at least 6 characters.");
    setBusy(true);
    try {
      const r = await login(email, password);
      setSession(r.access_token, r.user ?? { email });
      router.push(next);
    } catch (err) { setError(err instanceof ApiError ? err.message : "Login failed. Is the backend running at http://localhost:8000?"); }
    finally { setBusy(false); }
  };
  const guest = async () => {
    setError(""); setBusy(true);
    try {
      const r = await guestLogin();
      setSession(r.access_token, r.user ?? { name: "Guest" });
      router.push(next);
    } catch { setError("Guest login needs the backend (/api/auth/guest). Start the backend, or sign up instead."); }
    finally { setBusy(false); }
  };

  return (
    <main className="mx-auto flex max-w-md flex-col px-4 py-16">
      <Card>
        <h1 className="text-2xl font-bold">Welcome back</h1>
        <p className="mt-1 text-sm text-muted">Log in to start or join a SignSpeak call.</p>
        <form onSubmit={submit} className="mt-6 space-y-4">
          <div><Label htmlFor="email">Email</Label><Input id="email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required /></div>
          <div><Label htmlFor="password">Password</Label><Input id="password" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" required /></div>
          {error && <p role="alert" className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">{error}</p>}
          <Button type="submit" className="w-full" disabled={busy}><LogIn size={16} /> {busy ? "Logging in…" : "Log in"}</Button>
        </form>
        <Button variant="outline" className="mt-3 w-full" onClick={guest} disabled={busy}><Sparkles size={16} /> Continue as guest (demo)</Button>
        <p className="mt-4 text-center text-sm text-muted">No account? <Link href="/signup" className="text-cyan hover:underline">Sign up</Link></p>
      </Card>
    </main>
  );
}

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-ink">
      <style>{`button:focus-visible, a:focus-visible, input:focus-visible { outline: 2px solid #22D3EE; outline-offset: 2px; }`}</style>
      <Navbar />
      <Suspense>
        <LoginForm />
      </Suspense>
    </div>
  );
}
