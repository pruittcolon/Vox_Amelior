# Nemo Frontend: The Cognitive Dashboard

A high-performance, **Zero-Build** Single Page Application (SPA) designed for real-time visualization of cognitive AI processes.

> **Design Philosophy:** "Complexity in the backend, simplicity in the frontend."

---

## Architectural Decisions

### 1. The "No-Build" Strategy
Unlike typical React/Vue apps requiring complex toolchains (Webpack, Babel, node_modules), this frontend uses **Native ES Modules (ESM)**.
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

### Core Pages
| Module | File | Description |
| :--- | :--- | :--- |
| **Dashboard** | `index.html` | Main landing page with system overview |
| **Live Feed** | `transcripts.html` | Real-time scrolling transcript with speaker identification |
| **Cognitive Chat** | `gemma.html` | Interface for the Gemma LLM with RAG context injection |
| **Memory Bank** | `memories.html` | CRUD interface for the Vector Database (FAISS) |
| **Emotions** | `emotions.html` | Speaker emotion tracking and visualization |
| **Nexus** | `nexus.html` | Unified visualization of all 22+ engine outputs |

### Analytics & ML
| Module | File | Description |
| :--- | :--- | :--- |
| **Databases** | `databases.html` | Database vectorization and analysis |
| **Analytics** | `analytics.html` | Business intelligence visualizations |
| **ML Dashboard** | `ml_dashboard.html` | ML engine monitoring and results |

### Enterprise Features
| Module | File | Description |
| :--- | :--- | :--- |
| **Banking** | `banking.html` | Fiserv DNA integration dashboard |
| **Salesforce** | `salesforce.html` | Salesforce CRM integration |
| **Financial** | `financial-dashboard.html` | Financial analytics and forecasting |
| **Automation** | `automation.html` | n8n workflow automation |

### Settings & Admin
| Module | File | Description |
| :--- | :--- | :--- |
| **Login** | `login.html` | JWT authentication flow |
| **Settings** | `settings.html` | User and system configuration |
| **Admin QA** | `admin_qa.html` | Quality assurance and testing |

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