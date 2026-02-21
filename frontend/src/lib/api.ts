/**
 * Returns the HTTP(S) base URL for the backend API.
 *
 * Routing logic:
 * - braceml.com        → https://ws.braceml.com
 * - HTTPS (Tailscale)  → https://<hostname>:8443
 * - HTTP  (local dev)  → http://<hostname>:8001
 *
 * The build-time env var NEXT_PUBLIC_API_URL takes precedence if set.
 */
export function getApiBase(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof window === "undefined") {
    return "http://localhost:8000";
  }
  if (window.location.hostname === "braceml.com") {
    return "https://ws.braceml.com";
  }
  const isSecure = window.location.protocol === "https:";
  const port = isSecure ? 8443 : 8001;
  const protocol = isSecure ? "https:" : "http:";
  return `${protocol}//${window.location.hostname}:${port}`;
}

/**
 * Returns the WebSocket base URL for the backend.
 *
 * Routing logic:
 * - braceml.com        → wss://ws.braceml.com
 * - HTTPS (Tailscale)  → wss://<hostname>:8443
 * - HTTP  (local dev)  → ws://<hostname>:8001
 *
 * The build-time env var NEXT_PUBLIC_WS_URL takes precedence if set.
 */
export function getWsBase(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }
  if (typeof window === "undefined") {
    return "ws://localhost:8001";
  }
  if (window.location.hostname === "braceml.com") {
    return "wss://ws.braceml.com";
  }
  const isSecure = window.location.protocol === "https:";
  const port = isSecure ? 8443 : 8001;
  const protocol = isSecure ? "wss:" : "ws:";
  return `${protocol}//${window.location.hostname}:${port}`;
}
