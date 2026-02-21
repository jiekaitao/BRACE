/**
 * Returns the HTTP(S) base URL for the backend API.
 *
 * When accessed via HTTPS (e.g. Tailscale serve), the backend is proxied on
 * port 8443.  Over plain HTTP (local dev) it's on port 8000.
 *
 * The build-time env var NEXT_PUBLIC_API_URL takes precedence if set.
 */
export function getApiBase(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof window === "undefined") {
    return "http://localhost:8001";
  }
  // Client-side: Proxy through Next.js rewrite to bypass Mixed Content limits
  return "/api/backend";
}

/**
 * Returns the WebSocket base URL for the backend.
 *
 * Same port logic as getApiBase() but with ws:/wss: protocol.
 */
export function getWsBase(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }
  if (typeof window === "undefined") {
    return "ws://localhost:8001";
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  // Client-side: Proxy through Next.js rewrite to bypass mixed content limits
  return `${protocol}//${window.location.host}/api/backend`;
}
