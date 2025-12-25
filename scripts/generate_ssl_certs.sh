#!/bin/bash
# ============================================================================
# Generate Self-Signed SSL Certificates for HTTPS Development
# ============================================================================
# Usage: ./scripts/generate_ssl_certs.sh
#
# This script creates a self-signed certificate for localhost development.
# The certificate will be valid for localhost, 127.0.0.1, and any custom
# domains you add.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CERTS_DIR="$PROJECT_ROOT/certs"

# Certificate validity (days)
VALID_DAYS=365

# Certificate details
COUNTRY="US"
STATE="Colorado"
LOCALITY="Denver"
ORGANIZATION="NemoServer"
COMMON_NAME="localhost"

echo "🔐 Generating Self-Signed SSL Certificates for HTTPS Development"
echo "================================================================"

# Create certs directory if it doesn't exist
mkdir -p "$CERTS_DIR"

# Generate private key
echo "📝 Generating private key..."
openssl genrsa -out "$CERTS_DIR/key.pem" 2048

# Create certificate signing request config
cat > "$CERTS_DIR/cert.conf" << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
C = $COUNTRY
ST = $STATE
L = $LOCALITY
O = $ORGANIZATION
CN = $COMMON_NAME

[v3_req]
subjectAltName = @alt_names
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate self-signed certificate
echo "📜 Generating self-signed certificate..."
openssl req -new -x509 \
    -key "$CERTS_DIR/key.pem" \
    -out "$CERTS_DIR/cert.pem" \
    -days $VALID_DAYS \
    -config "$CERTS_DIR/cert.conf"

# Clean up config file
rm "$CERTS_DIR/cert.conf"

# Set proper permissions
chmod 600 "$CERTS_DIR/key.pem"
chmod 644 "$CERTS_DIR/cert.pem"

echo ""
echo "✅ SSL certificates generated successfully!"
echo ""
echo "📁 Certificate location: $CERTS_DIR"
echo "   - cert.pem (certificate)"
echo "   - key.pem (private key)"
echo ""
echo "📌 To use HTTPS, start the server with:"
echo "   ./start-https.sh"
echo ""
echo "⚠️  Note: Your browser will show a security warning for self-signed certs."
echo "   Click 'Advanced' → 'Proceed to localhost (unsafe)' to continue."
echo ""
