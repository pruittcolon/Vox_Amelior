#!/bin/bash
# ============================================================================
# Nemo Server HTTPS Startup Script
# ============================================================================
# Starts the Nemo Server with HTTPS using self-signed certificates.
# 
# Prerequisites:
#   - Run ./scripts/generate_ssl_certs.sh first to create certificates
#   - Docker and Docker Compose must be installed
#
# Usage: ./start-https.sh [--build] [--no-browser]
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

CERTS_DIR="$SCRIPT_DIR/certs"
HTTPS_PORT="${HTTPS_PORT:-8443}"

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC}  $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC}  $1"; }

# Check for SSL certificates
check_certs() {
    print_header "Checking SSL Certificates"
    
    if [ ! -f "$CERTS_DIR/cert.pem" ] || [ ! -f "$CERTS_DIR/key.pem" ]; then
        print_warning "SSL certificates not found!"
        print_info "Generating self-signed certificates..."
        
        if [ -f "$SCRIPT_DIR/scripts/generate_ssl_certs.sh" ]; then
            bash "$SCRIPT_DIR/scripts/generate_ssl_certs.sh"
        else
            print_error "Certificate generation script not found!"
            print_info "Create it at: scripts/generate_ssl_certs.sh"
            exit 1
        fi
    fi
    
    if [ -f "$CERTS_DIR/cert.pem" ] && [ -f "$CERTS_DIR/key.pem" ]; then
        print_success "SSL certificates found"
        # Show certificate info
        local expiry
        expiry=$(openssl x509 -enddate -noout -in "$CERTS_DIR/cert.pem" 2>/dev/null | cut -d= -f2)
        print_info "Certificate expires: $expiry"
    else
        print_error "Failed to generate/find SSL certificates"
        exit 1
    fi
}

# Set secure cookie environment variables
setup_https_env() {
    print_header "Configuring HTTPS Environment"
    
    # Create or update docker/.env with HTTPS settings
    local env_file="$SCRIPT_DIR/docker/.env"
    
    # Backup existing .env if it exists
    if [ -f "$env_file" ]; then
        cp "$env_file" "$env_file.backup"
    fi
    
    # Add/update HTTPS-specific settings
    # Use sed to update existing values or append new ones
    if grep -q "SESSION_COOKIE_SECURE" "$env_file" 2>/dev/null; then
        sed -i 's/SESSION_COOKIE_SECURE=.*/SESSION_COOKIE_SECURE=true/' "$env_file"
    else
        echo "SESSION_COOKIE_SECURE=true" >> "$env_file"
    fi
    
    if grep -q "SESSION_COOKIE_SAMESITE" "$env_file" 2>/dev/null; then
        sed -i 's/SESSION_COOKIE_SAMESITE=.*/SESSION_COOKIE_SAMESITE=none/' "$env_file"
    else
        echo "SESSION_COOKIE_SAMESITE=none" >> "$env_file"
    fi
    
    print_success "HTTPS cookie settings configured"
    print_info "  SESSION_COOKIE_SECURE=true"
    print_info "  SESSION_COOKIE_SAMESITE=none"
}

# Start with HTTPS proxy
start_https() {
    print_header "Starting HTTPS Server"
    
    print_info "Starting base services first..."
    
    # Start the regular services
    ./start.sh --no-browser "$@" &
    local start_pid=$!
    
    # Wait for base services to come up
    print_info "Waiting for base services..."
    sleep 30
    
    # Check if API Gateway is healthy
    for i in {1..30}; do
        if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
            print_success "API Gateway is healthy on HTTP"
            break
        fi
        sleep 2
    done
    
    # Start HTTPS reverse proxy using stunnel or a simple python script
    print_info "Starting HTTPS reverse proxy on port $HTTPS_PORT..."
    
    # Use Python to create a simple HTTPS proxy
    python3 - "$CERTS_DIR/cert.pem" "$CERTS_DIR/key.pem" "$HTTPS_PORT" << 'HTTPS_PROXY' &
import sys
import ssl
import http.server
import urllib.request
import socketserver

CERT_FILE = sys.argv[1]
KEY_FILE = sys.argv[2]
HTTPS_PORT = int(sys.argv[3])
BACKEND_URL = "http://localhost:8000"

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_proxy(self, method):
        # Build target URL
        target_url = f"{BACKEND_URL}{self.path}"
        
        # Get request body for POST/PUT
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        
        # Build headers (forward relevant headers)
        headers = {}
        for key in ['Content-Type', 'Accept', 'Cookie', 'X-CSRF-Token', 'ws_csrf']:
            if key in self.headers:
                headers[key] = self.headers[key]
        
        try:
            req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=120) as resp:
                # Send response status
                self.send_response(resp.status)
                
                # Forward response headers
                for key, value in resp.headers.items():
                    if key.lower() not in ('transfer-encoding', 'content-encoding'):
                        self.send_header(key, value)
                self.end_headers()
                
                # Forward response body
                self.wfile.write(resp.read())
                
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())
    
    def do_GET(self):
        self.do_proxy("GET")
    
    def do_POST(self):
        self.do_proxy("POST")
    
    def do_PUT(self):
        self.do_proxy("PUT")
    
    def do_DELETE(self):
        self.do_proxy("DELETE")
    
    def log_message(self, format, *args):
        print(f"[HTTPS] {args[0]}")

# Create SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(CERT_FILE, KEY_FILE)

# Start HTTPS server
with socketserver.TCPServer(("", HTTPS_PORT), ProxyHandler) as httpd:
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    print(f"🔒 HTTPS proxy running on https://localhost:{HTTPS_PORT}")
    httpd.serve_forever()
HTTPS_PROXY
    
    PROXY_PID=$!
    
    # Wait and verify
    sleep 3
    if kill -0 $PROXY_PID 2>/dev/null; then
        print_success "HTTPS proxy started successfully"
    else
        print_warning "HTTPS proxy may have failed to start"
    fi
    
    print_header "HTTPS Server Running"
    echo -e "${GREEN}🔒 HTTPS Server is running!${NC}"
    echo ""
    echo "  📡 HTTPS URL:  https://localhost:$HTTPS_PORT"
    echo "  📡 HTTP URL:   http://localhost:8000 (backend)"
    echo ""
    echo -e "${YELLOW}⚠️  Your browser will show a certificate warning.${NC}"
    echo "  Click 'Advanced' → 'Proceed to localhost (unsafe)' to continue."
    echo ""
    
    # Open browser
    if [[ "$*" != *"--no-browser"* ]]; then
        sleep 2
        if command -v xdg-open &> /dev/null; then
            xdg-open "https://localhost:$HTTPS_PORT/ui/login.html" >/dev/null 2>&1 || true
        elif command -v open &> /dev/null; then
            open "https://localhost:$HTTPS_PORT/ui/login.html" >/dev/null 2>&1 || true
        fi
    fi
    
    # Wait for background start.sh
    wait $start_pid
}

# Main
main() {
    clear
    echo -e "${BLUE}"
    cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         🔒  NEMO SERVER HTTPS STARTUP SCRIPT  🔒             ║
    ║                                                               ║
    ║            Secure HTTPS with Self-Signed Certs               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n"
    
    check_certs
    setup_https_env
    start_https "$@"
}

main "$@"
