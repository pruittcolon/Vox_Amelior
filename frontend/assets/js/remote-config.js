/**
 * Remote Configuration for File System Access
 * 
 * This script automatically configures the API base URL when the application
 * is opened via the file:// protocol (e.g., on a mounted network share).
 * 
 * It intercepts window.fetch to prepend the server IP.
 */
(function () {
    // Configuration
    const SERVER_IP = '100.68.213.84'; // Tailscale IP of the Nemo Server
    const SERVER_PORT = '8000';
    const API_BASE = `http://${SERVER_IP}:${SERVER_PORT}`;

    // Check if running from file system
    if (window.location.protocol === 'file:') {
        console.info('%c[RemoteConfig] Detected file:// protocol.', 'color: #00ff00; font-weight: bold;');
        console.info(`[RemoteConfig] Configuring remote API access to: ${API_BASE}`);

        // Store globally for other scripts that might check it
        window.API_BASE_URL = API_BASE;

        // Intercept Fetch API
        const originalFetch = window.fetch;
        window.fetch = function (url, options) {
            if (typeof url === 'string' && url.startsWith('/')) {
                const newUrl = API_BASE + url;
                // console.debug(`[RemoteConfig] Proxied fetch: ${url} -> ${newUrl}`);
                return originalFetch(newUrl, options);
            }
            return originalFetch(url, options);
        };

        // Intercept WebSocket if used (for banking realtime)
        const OriginalWebSocket = window.WebSocket;
        window.WebSocket = function (url, protocols) {
            if (typeof url === 'string' && url.startsWith('/')) {
                // Handle absolute path for websocket logic if needed, 
                // though usually WS urls are constructed differently.
                // This is a placeholder for safety.
            }
            // If code uses `ws://localhost:8000`, replace it
            if (url.includes('localhost:8000') || url.includes('127.0.0.1:8000')) {
                url = url.replace('localhost:8000', `${SERVER_IP}:${SERVER_PORT}`)
                    .replace('127.0.0.1:8000', `${SERVER_IP}:${SERVER_PORT}`);
                console.info(`[RemoteConfig] Proxied WebSocket: -> ${url}`);
            }
            return new OriginalWebSocket(url, protocols);
        };
        window.WebSocket.prototype = OriginalWebSocket.prototype;
        window.WebSocket.CONNECTING = OriginalWebSocket.CONNECTING;
        window.WebSocket.OPEN = OriginalWebSocket.OPEN;
        window.WebSocket.CLOSING = OriginalWebSocket.CLOSING;
        window.WebSocket.CLOSED = OriginalWebSocket.CLOSED;

    } else {
        // Normal server mode
        window.API_BASE_URL = '';
    }
})();
