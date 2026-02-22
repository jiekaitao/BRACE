import { getApiBase } from "./api";

const TOKEN_KEY = "iw_auth_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): HeadersInit {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export interface AuthUser {
  user_id: string;
  username: string;
  injury_profile?: Record<string, unknown> | null;
  risk_modifiers?: Record<string, unknown> | null;
  research_guidelines?: Record<string, unknown> | null;
}

export interface AuthResponse {
  user_id: string;
  username: string;
  token: string;
}

export async function register(username: string): Promise<AuthResponse> {
  const res = await fetch(`${getApiBase()}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Registration failed" }));
    throw new Error(err.detail || "Registration failed");
  }
  const data = await res.json();
  setToken(data.token);
  return data;
}

export async function login(username: string): Promise<AuthResponse> {
  const res = await fetch(`${getApiBase()}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Login failed" }));
    throw new Error(err.detail || "Login failed");
  }
  const data = await res.json();
  setToken(data.token);
  return data;
}

export async function logout(): Promise<void> {
  try {
    await fetch(`${getApiBase()}/api/auth/logout`, {
      method: "POST",
      headers: authHeaders(),
    });
  } finally {
    clearToken();
  }
}

export async function getMe(): Promise<AuthUser | null> {
  const token = getToken();
  if (!token) return null;
  const res = await fetch(`${getApiBase()}/api/auth/me`, {
    headers: authHeaders(),
  });
  if (!res.ok) {
    clearToken();
    return null;
  }
  return res.json();
}

export async function getProfile(): Promise<{ injury_profile: Record<string, unknown> | null; risk_modifiers: Record<string, unknown> | null }> {
  const res = await fetch(`${getApiBase()}/api/auth/profile`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Failed to get profile");
  return res.json();
}

export async function updateProfile(data: { injury_profile?: Record<string, unknown> | null; risk_modifiers?: Record<string, unknown> | null }): Promise<void> {
  const res = await fetch(`${getApiBase()}/api/auth/profile`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to update profile");
}
