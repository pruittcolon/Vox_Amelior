#!/bin/bash
#
# Istio Installation Script for Nemo Server
# Installs Istio service mesh with mTLS STRICT mode
#
# Usage: ./install_istio.sh [--dry-run]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ISTIO_VERSION="${ISTIO_VERSION:-1.20.0}"
DRY_RUN=false

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            print_warning "Dry run mode - no changes will be made"
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl first."
        exit 1
    fi
    print_success "kubectl installed: $(kubectl version --client -o json | jq -r '.clientVersion.gitVersion')"
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        print_info "Make sure you have a valid kubeconfig"
        exit 1
    fi
    print_success "Connected to cluster: $(kubectl cluster-info | head -1)"
    
    # Check if istioctl is installed
    if command -v istioctl &> /dev/null; then
        print_success "istioctl already installed: $(istioctl version --client 2>/dev/null || echo 'version check failed')"
    else
        print_warning "istioctl not found - will download"
    fi
}

# Download istioctl if needed
install_istioctl() {
    print_header "Installing istioctl"
    
    if command -v istioctl &> /dev/null; then
        print_info "istioctl already installed, skipping download"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would download Istio $ISTIO_VERSION"
        return
    fi
    
    print_info "Downloading Istio $ISTIO_VERSION..."
    
    # Download Istio
    curl -L https://istio.io/downloadIstio | ISTIO_VERSION=$ISTIO_VERSION sh -
    
    # Add to PATH for this session
    export PATH="$PWD/istio-$ISTIO_VERSION/bin:$PATH"
    
    # Suggest adding to PATH permanently
    print_success "istioctl downloaded"
    print_info "Add to PATH permanently:"
    print_info "  export PATH=\"$PWD/istio-$ISTIO_VERSION/bin:\$PATH\""
}

# Install Istio with custom configuration
install_istio() {
    print_header "Installing Istio Service Mesh"
    
    CONFIG_FILE="$PROJECT_ROOT/k8s/istio/istio-config.yaml"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Istio config not found: $CONFIG_FILE"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would install Istio with config: $CONFIG_FILE"
        print_info "[DRY RUN] Command: istioctl install -f $CONFIG_FILE -y"
        return
    fi
    
    # Pre-check
    print_info "Running pre-check..."
    istioctl x precheck
    
    # Install
    print_info "Installing Istio..."
    istioctl install -f "$CONFIG_FILE" -y
    
    # Verify installation
    print_info "Verifying installation..."
    istioctl verify-install
    
    print_success "Istio installed successfully"
}

# Apply peer authentication (mTLS)
configure_mtls() {
    print_header "Configuring mTLS STRICT Mode"
    
    PEER_AUTH_FILE="$PROJECT_ROOT/k8s/istio/peer-authentication.yaml"
    
    if [ ! -f "$PEER_AUTH_FILE" ]; then
        print_error "Peer authentication config not found: $PEER_AUTH_FILE"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would apply: $PEER_AUTH_FILE"
        return
    fi
    
    # Ensure nemo namespace exists with injection enabled
    kubectl get namespace nemo &> /dev/null || kubectl create namespace nemo
    kubectl label namespace nemo istio-injection=enabled --overwrite
    
    print_info "Applying peer authentication policies..."
    kubectl apply -f "$PEER_AUTH_FILE"
    
    print_success "mTLS STRICT mode configured"
}

# Apply authorization policies
configure_authorization() {
    print_header "Configuring Authorization Policies"
    
    POLICIES_DIR="$PROJECT_ROOT/k8s/istio/authorization-policies"
    
    if [ ! -d "$POLICIES_DIR" ]; then
        print_error "Authorization policies directory not found: $POLICIES_DIR"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would apply policies from: $POLICIES_DIR"
        for f in "$POLICIES_DIR"/*.yaml; do
            print_info "  - $(basename "$f")"
        done
        return
    fi
    
    print_info "Applying authorization policies..."
    kubectl apply -f "$POLICIES_DIR/"
    
    print_success "Authorization policies configured"
}

# Apply destination rules
configure_destination_rules() {
    print_header "Configuring Destination Rules"
    
    DEST_RULES_FILE="$PROJECT_ROOT/k8s/istio/destination-rules.yaml"
    
    if [ ! -f "$DEST_RULES_FILE" ]; then
        print_error "Destination rules not found: $DEST_RULES_FILE"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would apply: $DEST_RULES_FILE"
        return
    fi
    
    print_info "Applying destination rules..."
    kubectl apply -f "$DEST_RULES_FILE"
    
    print_success "Destination rules configured"
}

# Verify the installation
verify_installation() {
    print_header "Verifying Installation"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would run verification"
        return
    fi
    
    # Check Istio pods
    print_info "Checking Istio control plane..."
    kubectl get pods -n istio-system
    
    # Analyze configuration
    print_info "Analyzing Istio configuration..."
    istioctl analyze -n nemo || print_warning "Some issues found, review above"
    
    # Check peer authentication
    print_info "Checking peer authentication..."
    kubectl get peerauthentication -n nemo
    
    # Check authorization policies
    print_info "Checking authorization policies..."
    kubectl get authorizationpolicy -n nemo
    
    # Check destination rules
    print_info "Checking destination rules..."
    kubectl get destinationrule -n nemo
}

# Print next steps
print_next_steps() {
    print_header "Next Steps"
    
    echo "1. Restart your deployments to inject Istio sidecars:"
    echo "   kubectl rollout restart deployment -n nemo"
    echo ""
    echo "2. Verify sidecar injection:"
    echo "   kubectl get pods -n nemo -o jsonpath='{.items[*].spec.containers[*].name}'"
    echo ""
    echo "3. Test mTLS enforcement:"
    echo "   kubectl exec -n nemo deploy/api-gateway -c istio-proxy -- \\"
    echo "     curl -s http://gemma-service:8001/health"
    echo ""
    echo "4. View Kiali dashboard (if installed):"
    echo "   istioctl dashboard kiali"
    echo ""
    echo "5. Run security tests:"
    echo "   pytest tests/security/test_month1_mtls.py -v"
}

# Main
main() {
    print_header "Nemo Server - Istio Installation"
    echo "Version: $ISTIO_VERSION"
    echo "Config: $PROJECT_ROOT/k8s/istio/"
    echo ""
    
    check_prerequisites
    install_istioctl
    install_istio
    configure_mtls
    configure_authorization
    configure_destination_rules
    verify_installation
    print_next_steps
    
    print_header "Installation Complete"
    print_success "Istio service mesh installed with mTLS STRICT mode!"
}

main
