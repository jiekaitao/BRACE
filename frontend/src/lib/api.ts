/**
 * Returns the HTTP(S) base URL for the backend API.
 *
 * Routing logic:
 * - braceml.com               → https://ws.braceml.com
 * - HTTPS (Tailscale / Caddy) → same origin (Caddy routes /api/* to backend)
 * - HTTP  (local dev)         → http://<hostname>:8001
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
  if (window.location.hostname === "braceml.com") {
    return "https://ws.braceml.com";
  }
  const isSecure = window.location.protocol === "https:";
  if (isSecure) {
    // Behind Caddy reverse proxy — /api/* routes to backend on same origin
    return "";
  }
  return `http://${window.location.hostname}:8001`;
}

/**
 * Returns the WebSocket base URL for the backend.
 *
 * Routing logic:
 * - braceml.com               → wss://ws.braceml.com
 * - HTTPS (Tailscale / Caddy) → same origin (Caddy routes /ws/* to backend)
 * - HTTP  (local dev)         → ws://<hostname>:8001
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
  if (isSecure) {
    // Behind Caddy reverse proxy — /ws/* routes to backend on same origin
    return `wss://${window.location.hostname}`;
  }
  return `ws://${window.location.hostname}:8001`;
}
