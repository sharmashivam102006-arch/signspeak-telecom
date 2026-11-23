"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { UserPlus } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { register, setSession, ApiError } from "@/lib/api";

export default function SignupPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError("");
    if (name.trim().length < 2) return setError("Enter your name.");
    if (!/^\S+@\S+\.\S+$/.test(email)) return setError("Enter a valid email address.");
    if (password.length < 8) return setError("Password must be at least 8 characters.");
    if (password !== confirm) return setError("Passwords do not match.");
    setBusy(true);
    try {
      const r = await register(name.trim(), email, password);
      setSession(r.access_token, r.user ?? { name: name.trim(), email });
      router.push("/dashboard");
    } catch (err) { setError(err instanceof ApiError ? err.message : "Sign up failed. Is the backend running at http://localhost:8000?"); }
    finally { setBusy(false); }
  };

  return (
    <div className="min-h-screen bg-ink">
      <Navbar />
      <main className="mx-auto flex max-w-md flex-col px-4 py-16">
        <Card>
          <h1 className="text-2xl font-bold">Create your account</h1>
          <p className="mt-1 text-sm text-muted">Join SignSpeak Telecom — free to start.</p>
          <form onSubmit={submit} className="mt-6 space-y-4">
            <div><Label htmlFor="name">Full name</Label><Input id="name" value={name} onChange={e => setName(e.target.value)} placeholder="Asha Sharma" required /></div>
            <div><Label htmlFor="email">Email</Label><Input id="email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required /></div>
            <div><Label htmlFor="password">Password</Label><Input id="password" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="At least 8 characters" required /></div>
            <div><Label htmlFor="confirm">Confirm password</Label><Input id="confirm" type="password" value={confirm} onChange={e => setConfirm(e.target.value)} placeholder="Repeat your password" required /></div>
            {error && <p role="alert" className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">{error}</p>}
            <Button type="submit" className="w-full" disabled={busy}><UserPlus size={16} /> {busy ? "Creating account…" : "Create account"}</Button>
          </form>
          <p className="mt-4 text-center text-sm text-muted">Already have an account? <Link href="/login" className="text-cyan hover:underline">Log in</Link></p>
        </Card>
      </main>
    </div>
  );
}
