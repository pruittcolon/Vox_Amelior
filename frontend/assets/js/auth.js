/**
 * Authentication and Session Management
 * Handles login state, role checks, and session validation
 * Supports both cookie-based and localStorage token-based auth (for iframe contexts like VS Code Simple Browser)
 */

const API_BASE = (() => {
  if (typeof window !== 'undefined') {
    const configured =
      window.API_BASE_URL ||
      window.API_GATEWAY_URL ||
      (window.NEMO_API_OPTIONS && window.NEMO_API_OPTIONS.baseURL);
    if (configured) {
      return configured.replace(/\/+$/, '');
    }
    const origin = window.location && window.location.origin;
    if (origin && origin !== 'null' && origin.startsWith('http')) {
      return origin.replace(/\/+$/, '');
    }
  }
  return 'http://localhost:8000';
})();

function buildApiUrl(path) {
  if (!path) return API_BASE;
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${normalized}`;
}

// Token storage key
const TOKEN_STORAGE_KEY = 'nemo_session_token';
const CSRF_STORAGE_KEY = 'nemo_csrf_token';

/**
 * Get stored auth token from localStorage
 */
function getStoredToken() {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch (e) {
    return null;
  }
}

/**
 * Store auth token in localStorage
 */
function setStoredToken(token) {
  try {
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  } catch (e) {
    console.warn('[AUTH] localStorage not available');
  }
}

/**
 * Get stored CSRF token from localStorage
 */
function getStoredCsrf() {
  try {
    return localStorage.getItem(CSRF_STORAGE_KEY);
  } catch (e) {
    return null;
  }
}

/**
 * Store CSRF token in localStorage
 */
function setStoredCsrf(token) {
  try {
    if (token) {
      localStorage.setItem(CSRF_STORAGE_KEY, token);
    } else {
      localStorage.removeItem(CSRF_STORAGE_KEY);
    }
  } catch (e) {
    console.warn('[AUTH] localStorage not available');
  }
}

/**
 * Build headers for authenticated requests
 * Uses Bearer token from localStorage if available
 */
function getAuthHeaders() {
  const headers = {};
  const token = getStoredToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const csrf = getStoredCsrf();
  if (csrf) {
    headers['X-CSRF-Token'] = csrf;
  }
  return headers;
}

/**
 * Make an authenticated fetch request
 * Automatically includes auth headers from localStorage
 */
async function authFetch(url, options = {}) {
  const authHeaders = getAuthHeaders();
  const mergedOptions = {
    ...options,
    credentials: 'include',
    headers: {
      ...authHeaders,
      ...(options.headers || {})
    }
  };
  return fetch(url, mergedOptions);
}

const Auth = {
  currentUser: null,
  
  /**
   * Check if user is authenticated
   * Call this on page load
   */
  async checkSession() {
    try {
      const response = await authFetch(buildApiUrl('/api/auth/check'), {
        method: 'GET'
      });
      
      const data = await response.json();
      
      if (data.valid && data.user) {
        // Normalize user object to always have username/user_id/role
        const raw = data.user;
        this.currentUser = {
          username: raw.username || raw.user_id || 'user',
          user_id: raw.user_id || raw.username || 'user',
          role: raw.role || 'viewer',
          speaker_id: raw.speaker_id || null
        };
        return true;
      } else {
        this.currentUser = null;
        // Clear stored token if session is invalid
        setStoredToken(null);
        setStoredCsrf(null);
        return false;
      }
    } catch (error) {
      console.error('Session check failed:', error);
      this.currentUser = null;
      return false;
    }
  },
  
  /**
   * Require authentication
   * Redirects to login if not authenticated
   */
  async requireAuth() {
    const isAuthenticated = await this.checkSession();
    
    if (!isAuthenticated) {
      // Redirect to login
      console.warn('[AUTH] Not authenticated, redirecting to login');
      window.location.href = 'login.html';
      return false;
    }
    
    return true;
  },
  
  /**
   * Check if user has required role
   * @param {string} requiredRole - 'admin', 'analyst', or 'viewer'
   */
  hasRole(requiredRole) {
    if (!this.currentUser) {
      return false;
    }
    
    const roleLevels = {
      'viewer': 1,
      'analyst': 2,
      'admin': 3
    };
    
    const userLevel = roleLevels[this.currentUser.role] || 0;
    const required = roleLevels[requiredRole] || 999;
    
    return userLevel >= required;
  },
  
  /**
   * Require specific role
   * Shows error and redirects if insufficient permissions
   */
  async requireRole(requiredRole) {
    await this.requireAuth();
    
    if (!this.hasRole(requiredRole)) {
      alert(`Access denied. This page requires ${requiredRole} role.`);
      window.location.href = '/index.html';
      return false;
    }
    
    return true;
  },
  
  /**
   * Logout user
   */
  async logout() {
    try {
      await authFetch(buildApiUrl('/api/auth/logout'), {
        method: 'POST'
      });
      
      this.currentUser = null;
      // Clear stored tokens
      setStoredToken(null);
      setStoredCsrf(null);
      window.location.href = 'login.html';
    } catch (error) {
      console.error('Logout failed:', error);
      // Clear tokens and redirect anyway
      setStoredToken(null);
      setStoredCsrf(null);
      window.location.href = 'login.html';
    }
  },
  
  /**
   * Get current user info
   */
  getCurrentUser() {
    return this.currentUser;
  },
  
  /**
   * Update UI based on user role
   * Hide/show elements based on data-require-role attribute
   * INCLUDES SPEAKER ISOLATION FOR NON-ADMIN USERS
   */
  updateUIForRole() {
    if (!this.currentUser) return;
    
    const roleLevels = {
      'viewer': 1,
      'analyst': 2,
      'admin': 3,
      'user': 1  // Default user role
    };
    
    const userLevel = roleLevels[this.currentUser.role] || 0;
    
    // Find all elements with role requirements
    document.querySelectorAll('[data-require-role]').forEach(element => {
      const requiredRole = element.getAttribute('data-require-role');
      const requiredLevel = roleLevels[requiredRole] || 999;
      
      if (userLevel >= requiredLevel) {
        element.style.display = '';
      } else {
        element.style.display = 'none';
      }
    });
    
    // SPEAKER ISOLATION: Hide speaker selectors for non-admin users
    if (this.currentUser.role !== 'admin') {
      document.querySelectorAll('.speaker-filter, .speaker-dropdown, [data-admin-only]').forEach(el => {
        el.style.display = 'none';
      });
      console.log('[AUTH] Non-admin user - speaker filters hidden');
    }
    
    // Update user display in header
    const currentUserEl = document.getElementById('current-user');
    if (currentUserEl) {
      currentUserEl.textContent = `${this.currentUser.username} (${this.currentUser.role})`;
    }
    
    // Show logout button
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
      logoutBtn.style.display = 'flex';
      logoutBtn.onclick = () => this.logout();
    }
    
    // Update user display (legacy support)
    const userDisplays = document.querySelectorAll('.user-display');
    userDisplays.forEach(display => {
      display.textContent = this.currentUser.username;
    });
    
    const roleDisplays = document.querySelectorAll('.role-display');
    roleDisplays.forEach(display => {
      display.textContent = this.currentUser.role;
      display.className = `role-display badge badge-${this.getRoleBadgeClass()}`;
    });
  },
  
  /**
   * Get badge class for role
   */
  getRoleBadgeClass() {
    switch (this.currentUser?.role) {
      case 'admin':
        return 'danger';
      case 'analyst':
        return 'primary';
      case 'viewer':
        return 'success';
      default:
        return 'secondary';
    }
  },
  
  /**
   * Initialize auth on page load
   * Call this in every protected page
   */
  async init(options = {}) {
    const { requireAuth = true, requireRole = null } = options;
    
    if (requireAuth) {
      const authenticated = await this.requireAuth();
      if (!authenticated) return false;
    }
    
    if (requireRole) {
      const authorized = await this.requireRole(requireRole);
      if (!authorized) return false;
    }
    
    // Update UI
    this.updateUIForRole();
    
    // Setup logout buttons
    document.querySelectorAll('[data-logout]').forEach(button => {
      button.addEventListener('click', () => this.logout());
    });
    
    return true;
  }
};

// Auto-check session on page load (but don't redirect unless explicitly called)
document.addEventListener('DOMContentLoaded', async () => {
  await Auth.checkSession();
});
