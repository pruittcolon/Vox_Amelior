#!/bin/bash
# Generate TLS certificates for Nemo Server
# 
# Best Practices Applied:
# - RSA 2048-bit key (NIST minimum)
# - SHA256 signing (modern standard)
# - SAN (Subject Alternative Name) for hostname flexibility
# - 90-day validity (forces rotation, matches Let's Encrypt)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SSL_DIR="$SCRIPT_DIR/../docker/ssl"
DOMAIN="${NEMO_DOMAIN:-localhost}"

echo "🔐 Generating TLS Certificates for Nemo Server"
echo "   Domain: $DOMAIN"
echo "   Output: $SSL_DIR"
echo ""

mkdir -p "$SSL_DIR"

# Generate private key (restricted permissions)
echo "📝 Generating RSA 2048-bit private key..."
openssl genrsa -out "$SSL_DIR/nemo.key" 2048 2>/dev/null
chmod 600 "$SSL_DIR/nemo.key"

# Generate certificate signing request with SAN
echo "📝 Creating certificate configuration..."
cat > "$SSL_DIR/nemo.cnf" << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_ext
req_extensions = v3_req

[dn]
C=US
ST=Arizona
L=Phoenix
O=Nemo Server
OU=Development
CN=$DOMAIN

[v3_req]
subjectAltName = @alt_names

[v3_ext]
subjectAltName = @alt_names
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = localhost
DNS.3 = api-gateway
DNS.4 = nginx
DNS.5 = *.nemo.local
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
EOF

# Generate self-signed certificate
echo "📝 Generating self-signed certificate (90 days)..."
openssl req -new -x509 -sha256 -days 90 \
    -key "$SSL_DIR/nemo.key" \
    -out "$SSL_DIR/nemo.crt" \
    -config "$SSL_DIR/nemo.cnf" \
    -extensions v3_ext 2>/dev/null

# Generate combined PEM for some services
cat "$SSL_DIR/nemo.crt" "$SSL_DIR/nemo.key" > "$SSL_DIR/nemo.pem"
chmod 600 "$SSL_DIR/nemo.pem"

# Verify certificate
echo ""
echo "✅ Certificates generated successfully!"
echo ""
echo "📄 Certificate Details:"
openssl x509 -in "$SSL_DIR/nemo.crt" -noout -subject -dates -issuer 2>/dev/null | sed 's/^/   /'
echo ""
echo "📁 Files created:"
echo "   Certificate: $SSL_DIR/nemo.crt"
echo "   Private Key: $SSL_DIR/nemo.key (mode 600)"
echo "   Combined:    $SSL_DIR/nemo.pem"
echo ""
echo "⚠️  For production, replace with CA-signed certificates (Let's Encrypt)!"
