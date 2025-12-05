#!/usr/bin/env python3
"""
Nemo CLI - Main Entry Point

Unified command-line interface providing access to:
- API client (existing nemo_cli.py functionality)
- Service management (Docker, health checks)
- Testing framework (service-specific and full suite)
- Documentation validation

Architecture:
    nemo api <command>          # API interactions (transcribe, chat, etc.)
    nemo service <name> <cmd>   # Service management
    nemo test [service]         # Run tests
    nemo verify                 # Validate architecture docs
    nemo docs validate          # Alias for verify
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    """Main entry point for unified CLI"""
    parser = argparse.ArgumentParser(
        prog="nemo",
        description="Nemo Server Unified CLI",
        epilog="Run 'nemo <command> --help' for command-specific options"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ----------------------------------------------------------------
    # API command (existing nemo_cli.py functionality)
    # ----------------------------------------------------------------
    api_parser = subparsers.add_parser(
        "api",
        help="API interactions (transcribe, chat, search, etc.)"
    )
    api_parser.add_argument(
        "api_command",
        help="API command to run (use 'nemo api --help' for full list)"
    )
    api_parser.add_argument(
        "api_args",
        nargs=argparse.REMAINDER,
        help="Arguments for API command"
    )
    
    # ----------------------------------------------------------------
    # Service management
    # ----------------------------------------------------------------
    service_parser = subparsers.add_parser(
        "service",
        help="Manage Docker services"
    )
    service_parser.add_argument(
        "service_name",
        choices=[
            "all", "gateway", "gemma", "gpu-coordinator",
            "transcription", "rag", "emotion", "ml-service", "insights"
        ],
        help="Service to manage or 'all'"
    )
    service_parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "logs", "health", "test"],
        help="Action to perform"
    )
    service_parser.add_argument(
        "--follow", "-f",
        action="store_true",
        help="Follow logs (for 'logs' action)"
    )
    
    # ----------------------------------------------------------------
    # Testing
    # ----------------------------------------------------------------
    test_parser = subparsers.add_parser(
        "test",
        help="Run test suite"
    )
    test_parser.add_argument(
        "service",
        nargs="?",
        choices=[
            "all", "gateway", "gemma", "gpu-coordinator",
            "transcription", "rag", "emotion", "ml-service", "insights"
        ],
        default="all",
        help="Service to test (default: all)"
    )
    test_parser.add_argument(
        "--feature",
        help="Test specific feature"
    )
    test_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON for CI/CD"
    )
    test_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    # ----------------------------------------------------------------
    # Documentation validation
    # ----------------------------------------------------------------
    verify_parser = subparsers.add_parser(
        "verify",
        help="Validate architecture documentation"
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON"
    )
    verify_parser.add_argument(
        "--check",
        choices=["ports", "models", "schemas", "channels", "states"],
        help="Run specific validation check"
    )
    
    # Alias for verify
    docs_parser = subparsers.add_parser(
        "docs",
        help="Documentation commands"
    )
    docs_subparsers = docs_parser.add_subparsers(dest="docs_command")
    docs_validate = docs_subparsers.add_parser("validate", help="Validate ARCHITECTURE.md")
    docs_validate.add_argument("--json", action="store_true")
    
    # ----------------------------------------------------------------
    # Parse and route
    # ----------------------------------------------------------------
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate handler
    if args.command == "api":
        from nemo import api_client
        return api_client.handle_api_command(args.api_command, args.api_args)
    
    elif args.command == "service":
        from nemo import service_manager
        return service_manager.handle_service_command(
            args.service_name, args.action, follow=args.follow
        )
    
    elif args.command == "test":
        from nemo import test_runner
        return test_runner.handle_test_command(
            args.service, feature=args.feature, json_output=args.json, verbose=args.verbose
        )
    
    elif args.command == "verify":
        from nemo import validators
        return validators.handle_verify_command(json_output=args.json, check=args.check)
    
    elif args.command == "docs":
        if args.docs_command == "validate":
            from nemo import validators
            return validators.handle_verify_command(json_output=args.json)
        else:
            docs_parser.print_help()
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
