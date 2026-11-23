export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("signspeak_token");
}
export function getUser(): { id?: string; name?: string; email?: string } | null {
  if (typeof window === "undefined") return null;
  try { return JSON.parse(localStorage.getItem("signspeak_user") || "null"); } catch { return null; }
}
export function setSession(token: string, user: unknown) {
  localStorage.setItem("signspeak_token", token);
  localStorage.setItem("signspeak_user", JSON.stringify(user ?? {}));
}
export function clearSession() {
  localStorage.removeItem("signspeak_token");
  localStorage.removeItem("signspeak_user");
}

export class ApiError extends Error { constructor(public status: number, message: string) { super(message); } }

export async function api<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const token = getToken();
  const res = await fetch(`${API_URL}${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(opts.headers || {}),
    },
  });
  const text = await res.text();
  let data: unknown = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const msg = (data as { detail?: string; message?: string })?.detail
      || (data as { detail?: string; message?: string })?.message
      || `Request failed (${res.status})`;
    throw new ApiError(res.status, typeof msg === "string" ? msg : `Request failed (${res.status})`);
  }
  return data as T;
}

export type AuthResponse = { access_token: string; token_type?: string; user?: { id: string; name: string; email: string } };
export async function login(email: string, password: string) {
  return api<AuthResponse>("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
}
export async function register(name: string, email: string, password: string) {
  return api<AuthResponse>("/api/auth/register", { method: "POST", body: JSON.stringify({ name, email, password }) });
}
export async function guestLogin() {
  return api<AuthResponse>("/api/auth/guest", { method: "POST", body: JSON.stringify({}) });
}
export async function checkHealth(): Promise<{ status: string; services?: Record<string, string> }> {
  try { return await api<{ status: string }>("/health"); }
  catch { return { status: "unavailable" }; }
}
