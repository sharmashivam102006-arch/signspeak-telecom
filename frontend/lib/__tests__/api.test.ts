// @vitest-environment jsdom
// Unit tests for frontend/lib/api.ts — the typed fetch client every page
// uses: auth-header handling, error mapping to ApiError, and session
// helpers. fetch is stubbed; no network is touched.
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { api, ApiError, getToken, setSession, clearSession } from "../api";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api()", () => {
  it("returns parsed JSON on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ ok: true })));
    await expect(api<{ ok: boolean }>("/health")).resolves.toEqual({ ok: true });
  });

  it("throws ApiError with the server detail message on failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "Not found" }, 404)));
    const err = await api("/api/rooms/nope").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(404);
    expect((err as ApiError).message).toBe("Not found");
  });

  it("falls back to a generic message when the error body has no detail", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("boom", { status: 500 })));
    const err = await api("/health").catch((e) => e);
    expect((err as ApiError).status).toBe(500);
    expect((err as ApiError).message).toBe("Request failed (500)");
  });

  it("sends no Authorization header without a session", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);
    await api("/health");
    const headers = fetchMock.mock.calls[0][1].headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });

  it("sends the Bearer token when a session exists", async () => {
    setSession("tok-123", { id: "u1", name: "Test" });
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);
    await api("/api/rooms");
    const headers = fetchMock.mock.calls[0][1].headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer tok-123");
  });
});

describe("session helpers", () => {
  it("setSession / getToken / clearSession round-trip through localStorage", () => {
    expect(getToken()).toBeNull();
    setSession("abc", { id: "u1" });
    expect(getToken()).toBe("abc");
    expect(localStorage.getItem("signspeak_user")).toContain("u1");
    clearSession();
    expect(getToken()).toBeNull();
    expect(localStorage.getItem("signspeak_user")).toBeNull();
  });
});
