#!/usr/bin/env python3
"""
Generate API Documentation from OpenAPI Schema.

This script reads the OpenAPI schema from the running API Gateway
or from a static file and generates a human-readable Markdown
documentation file.

Usage:
    python scripts/generate_api_md.py
    python scripts/generate_api_md.py --from-file docs/api/openapi.json
    python scripts/generate_api_md.py --url http://localhost:8000/openapi.json

Output:
    docs/api.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


def fetch_openapi_from_url(url: str) -> Dict[str, Any]:
    """Fetch OpenAPI schema from a running service."""
    if not HTTPX_AVAILABLE:
        print("Error: httpx not installed. Run: pip install httpx")
        sys.exit(1)
    
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching OpenAPI schema: {e}")
        sys.exit(1)


def load_openapi_from_file(filepath: str) -> Dict[str, Any]:
    """Load OpenAPI schema from a file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading OpenAPI schema: {e}")
        sys.exit(1)


def get_method_badge(method: str) -> str:
    """Get a colored badge for HTTP method."""
    colors = {
        "get": "🟢",
        "post": "🟡",
        "put": "🟠",
        "patch": "🟣",
        "delete": "🔴"
    }
    return f"{colors.get(method.lower(), '⚪')} **{method.upper()}**"


def format_parameters(parameters: List[Dict]) -> str:
    """Format parameter list as markdown table."""
    if not parameters:
        return ""
    
    lines = [
        "| Name | Location | Type | Required | Description |",
        "|------|----------|------|----------|-------------|"
    ]
    
    for param in parameters:
        name = param.get("name", "")
        location = param.get("in", "")
        schema = param.get("schema", {})
        ptype = schema.get("type", "string")
        required = "✅" if param.get("required", False) else ""
        description = param.get("description", "").replace("\n", " ")[:50]
        
        lines.append(f"| `{name}` | {location} | {ptype} | {required} | {description} |")
    
    return "\n".join(lines)


def format_request_body(request_body: Optional[Dict]) -> str:
    """Format request body schema."""
    if not request_body:
        return ""
    
    content = request_body.get("content", {})
    json_content = content.get("application/json", {})
    schema = json_content.get("schema", {})
    
    if not schema:
        return ""
    
    lines = ["**Request Body:**"]
    
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        lines.append(f"- Schema: `{ref_name}`")
    elif schema.get("type") == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for prop_name, prop_schema in props.items():
            req = " *(required)*" if prop_name in required else ""
            ptype = prop_schema.get("type", "any")
            lines.append(f"- `{prop_name}`: {ptype}{req}")
    
    return "\n".join(lines)


def format_responses(responses: Dict) -> str:
    """Format response codes and descriptions."""
    if not responses:
        return ""
    
    lines = ["**Responses:**"]
    
    for code, response in responses.items():
        description = response.get("description", "")
        emoji = "✅" if code.startswith("2") else ("⚠️" if code.startswith("4") else "❌")
        lines.append(f"- `{code}` {emoji} {description}")
    
    return "\n".join(lines)


def generate_markdown(schema: Dict[str, Any]) -> str:
    """Generate markdown documentation from OpenAPI schema."""
    info = schema.get("info", {})
    paths = schema.get("paths", {})
    tags = schema.get("tags", [])
    
    # Build tag to description mapping
    tag_descriptions = {tag["name"]: tag.get("description", "") for tag in tags}
    
    # Group endpoints by tag
    endpoints_by_tag: Dict[str, List[tuple]] = {}
    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ["get", "post", "put", "patch", "delete"]:
                endpoint_tags = details.get("tags", ["Untagged"])
                for tag in endpoint_tags:
                    if tag not in endpoints_by_tag:
                        endpoints_by_tag[tag] = []
                    endpoints_by_tag[tag].append((path, method, details))
    
    # Generate document
    lines = [
        f"# {info.get('title', 'API Documentation')}",
        "",
        f"> Version: {info.get('version', '1.0.0')}",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        info.get("description", ""),
        "",
        "---",
        "",
        "## Table of Contents",
        ""
    ]
    
    # TOC
    for tag in sorted(endpoints_by_tag.keys()):
        anchor = tag.lower().replace(" ", "-")
        lines.append(f"- [{tag}](#{anchor})")
    
    lines.extend(["", "---", ""])
    
    # Common headers section
    lines.extend([
        "## Authentication",
        "",
        "All protected endpoints require one of:",
        "- **Bearer Token**: `Authorization: Bearer <jwt_token>`",
        "- **Session Cookie**: `ws_session` cookie from login",
        "",
        "---",
        "",
        "## Common Headers",
        "",
        "| Header | Description |",
        "|--------|-------------|",
        "| `X-Request-ID` | Unique request identifier for tracing |",
        "| `X-CSRF-Token` | CSRF token for mutating requests |",
        "| `Accept` | `application/json` |",
        "",
        "---",
        "",
        "## Error Responses",
        "",
        "All errors follow RFC 7807 format:",
        "```json",
        "{",
        '  "error_code": "ERROR_CODE",',
        '  "message": "Human readable message",',
        '  "details": {},',
        '  "request_id": "req_abc123"',
        "}",
        "```",
        "",
        "---",
        ""
    ])
    
    # Endpoints by tag
    for tag in sorted(endpoints_by_tag.keys()):
        tag_desc = tag_descriptions.get(tag, "")
        lines.append(f"## {tag}")
        if tag_desc:
            lines.append(f"*{tag_desc}*")
        lines.append("")
        
        for path, method, details in endpoints_by_tag[tag]:
            summary = details.get("summary", path)
            description = details.get("description", "")
            
            lines.append(f"### {get_method_badge(method)} `{path}`")
            lines.append("")
            lines.append(f"**{summary}**")
            if description:
                lines.append("")
                lines.append(description)
            lines.append("")
            
            # Parameters
            params = details.get("parameters", [])
            if params:
                lines.append(format_parameters(params))
                lines.append("")
            
            # Request body
            req_body = details.get("requestBody")
            if req_body:
                lines.append(format_request_body(req_body))
                lines.append("")
            
            # Responses
            responses = details.get("responses", {})
            if responses:
                lines.append(format_responses(responses))
                lines.append("")
            
            lines.append("---")
            lines.append("")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate API documentation")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/openapi.json",
        help="URL to fetch OpenAPI schema from"
    )
    parser.add_argument(
        "--from-file",
        help="Load OpenAPI schema from file instead of URL"
    )
    parser.add_argument(
        "--output",
        default="docs/api.md",
        help="Output file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout instead of writing file"
    )
    
    args = parser.parse_args()
    
    # Load schema
    if args.from_file:
        print(f"Loading schema from: {args.from_file}")
        schema = load_openapi_from_file(args.from_file)
    else:
        print(f"Fetching schema from: {args.url}")
        schema = fetch_openapi_from_url(args.url)
    
    # Generate markdown
    markdown = generate_markdown(schema)
    
    # Output
    if args.dry_run:
        print(markdown)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
        print(f"✅ Generated: {args.output}")
        print(f"   Endpoints: {len(schema.get('paths', {}))}")


if __name__ == "__main__":
    main()
