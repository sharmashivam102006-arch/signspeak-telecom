"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { LogOut } from "lucide-react";
import { Logo } from "./Logo";
import { Button } from "./ui/button";
import { clearSession, getUser } from "@/lib/api";

export function Navbar() {
  const router = useRouter();
  const [user, setUser] = useState<{ name?: string } | null>(null);
  useEffect(() => { setUser(getUser()); }, []);
  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-ink/85 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        <Logo />
        <nav className="hidden items-center gap-7 text-sm text-slate-300 md:flex">
          <a href="/#features" className="hover:text-cyan">Features</a>
          <a href="/#how-it-works" className="hover:text-cyan">How It Works</a>
          <a href="/#technology" className="hover:text-cyan">Technology</a>
          <a href="/#about" className="hover:text-cyan">About</a>
        </nav>
        <div className="flex items-center gap-2">
          {user ? (
            <>
              <Button variant="ghost" size="sm" onClick={() => router.push("/dashboard")}>Dashboard</Button>
              <Button variant="outline" size="sm" onClick={() => { clearSession(); setUser(null); router.push("/"); }}>
                <LogOut size={15} /> Log out
              </Button>
            </>
          ) : (
            <>
              <Link href="/login"><Button variant="ghost" size="sm">Login</Button></Link>
              <Link href="/signup"><Button size="sm">Start Communicating</Button></Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
