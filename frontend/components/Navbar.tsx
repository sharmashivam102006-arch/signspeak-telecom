"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { LogOut } from "lucide-react";
import { Logo } from "./Logo";
import { Button } from "./ui/button";
import { clearSession, getUser } from "@/lib/api";

export function Navbar() {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<{ name?: string } | null>(null);
  useEffect(() => { setUser(getUser()); }, []);
  // Marketing links only belong on the landing page; inside the app
  // (dashboard, call) they are noise.
  const onLanding = pathname === "/";
  const inApp = pathname?.startsWith("/dashboard") || pathname?.startsWith("/room");
  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-ink/85 backdrop-blur">
      <div className={`mx-auto flex items-center justify-between px-4 sm:px-6 ${inApp ? "max-w-7xl py-2.5" : "max-w-7xl py-3"}`}>
        <Logo />
        {onLanding && (
          <nav className="hidden items-center gap-7 text-sm text-slate-300 md:flex">
            <a href="/#features" className="hover:text-cyan">Features</a>
            <a href="/#how-it-works" className="hover:text-cyan">How It Works</a>
            <a href="/#technology" className="hover:text-cyan">Technology</a>
            <a href="/#about" className="hover:text-cyan">About</a>
          </nav>
        )}
        <div className="flex items-center gap-2">
          {user ? (
            <>
              {user.name && (
                <span className="hidden max-w-[10rem] truncate text-sm text-muted sm:inline">{user.name}</span>
              )}
              {pathname !== "/dashboard" && (
                <Button variant="ghost" size="sm" onClick={() => router.push("/dashboard")}>Dashboard</Button>
              )}
              <Button variant="outline" size="sm" onClick={() => { clearSession(); setUser(null); router.push("/"); }}>
                <LogOut size={15} /> Log out
              </Button>
            </>
          ) : (
            <>
              <Link href="/login"><Button variant="ghost" size="sm">Login</Button></Link>
              <Link href="/signup"><Button size="sm"><span className="sm:hidden">Sign up</span><span className="hidden sm:inline">Start Communicating</span></Button></Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
