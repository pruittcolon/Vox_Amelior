# Nemo Frontend: The Cognitive Dashboard

A high-performance, **Zero-Build** Single Page Application (SPA) designed for real-time visualization of cognitive AI processes.

> **Design Philosophy:** "Complexity in the backend, simplicity in the frontend."

---

## 🎨 Architectural Decisions

### 1. The "No-Build" Strategy
Unlike typical React/Vue apps requiring complex toolchains (Webpack, Babel, node_modules hell), this frontend uses **Native ES Modules (ESM)**.
*   **Benefit:** Zero compilation time. Instant hot-reload.
*   **Benefit:** Extremely lightweight (Browser caches raw files).
*   **Why:** Ensures the frontend is permanently viable without "dependency rot" 5 years from now.

### 2. Real-Time Event Stream
The dashboard connects to the API Gateway via **Server-Sent Events (SSE)** and **WebSockets**.
*   **Transcriptions:** Streamed token-by-token for a "typewriter" effect.
*   **System Health:** GPU VRAM usage, temperature, and lock status are pushed live to the UI.

### 3. Glassmorphism UI System
A custom CSS design system (`assets/css/design-tokens.css`) utilizing CSS Variables for theming.
*   **Performance:** Uses GPU-accelerated CSS properties (`backdrop-filter`, `transform`) for smooth 60fps animations.

---

## 🖥️ Dashboard Modules

| Module | File | Description |
| :--- | :--- | :--- |
| **Live Feed** | `index.html` | Real-time scrolling transcript with speaker identification avatars. |
| **Cognitive Chat** | `chat.html` | Interface for the Gemma LLM with RAG context injection controls. |
| **Memory Bank** | `memories.html` | CRUD interface for the Vector Database (FAISS). Allows manual editing of "memories". |
| **Analytics** | `analysis.html` | Plotly.js visualizations of System 2 (ML Service) outputs. |
| **Speaker ID** | `speakers.html` | Audio enrollment interface for Voice Biometrics. |

---

## 🔒 Security Integration

The frontend implements a strict **JWT-based Auth Flow**:
1.  **Login:** `login.html` exchanges credentials for an `HttpOnly` cookie.
2.  **Session:** No tokens are stored in `localStorage` (XSS protection).
3.  **CSRF:** All mutation requests require a synchronized CSRF token header.

---

## 🚀 Development

```bash
# No 'npm start' needed.
# The API Gateway serves these files directly.

# Edit a file -> Refresh Browser.
```

---

**Pruitt Colon**
*Full Stack Architect*