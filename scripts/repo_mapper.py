
import os
import ast
import re
from pathlib import Path

# Configuration
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "REPO_MAP.md"

try:
    from repo_context import SECTION_CONTEXT, AGENTIC_GUIDE
except ImportError:
    SECTION_CONTEXT = {}
    AGENTIC_GUIDE = ""

IGNORE_DIRS = {
    ".git", ".github", "__pycache__", "node_modules", "build", "dist", 
    ".idea", ".vscode", "venv", ".venv", ".gemini", "archive", "tests", "ml-service-old"
}
IGNORE_FILES = {
    "package-lock.json", ".DS_Store", "yarn.lock"
}
EXTENSIONS = {".py", ".dart", ".js", ".ts", ".tsx", ".jsx"}

def get_file_info(file_path):
    """Extracts information from a single file."""
    ext = file_path.suffix
    content = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return {"error": str(e)}

    info = {
        "path": str(file_path.relative_to(REPO_ROOT)),
        "classes": [],
        "functions": [],
        "imports": [],
        "routes": [], # New: Store API routes
        "models": [], # New: Store Pydantic models
        "description": ""
    }


    if ext == ".py":
        _parse_python(content, info)
    elif ext == ".dart":
        _parse_dart(content, info)
    elif ext in {".js", ".ts", ".tsx", ".jsx"}:
        _parse_js(content, info)
    
    return info

def _parse_python(content, info):
    try:
        tree = ast.parse(content)
        info["description"] = ast.get_docstring(tree) or ""
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Detect Pydantic Models (heuristic)
                is_model = any(
                    (isinstance(base, ast.Name) and base.id in ('BaseModel', 'Schema')) 
                    for base in node.bases
                )
                if is_model:
                    info["models"].append(node.name)
                else:
                    info["classes"].append(node.name)

            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"): # Skip private functions for brevity
                    info["functions"].append(node.name)
                
                # Detect FastAPI Routes
                if node.decorator_list:
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Call):
                            if (isinstance(dec.func, ast.Attribute) and 
                                dec.func.attr in ('get', 'post', 'put', 'delete', 'patch')):
                                # Try to extract path
                                if dec.args and isinstance(dec.args[0], ast.Constant):
                                    method = dec.func.attr.upper()
                                    path = dec.args[0].value
                                    info["routes"].append(f"{method} {path}")

            elif isinstance(node, ast.Import):
                for name in node.names:
                    info["imports"].append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    info["imports"].append(f"{module}.{name.name}")
    except SyntaxError:
        info["error"] = "Syntax Error parsing Python file"

def _parse_dart(content, info):
    # Regex for Dart classes
    class_pattern = re.compile(r'class\s+(\w+)')
    info["classes"].extend(class_pattern.findall(content))

    # Imports
    import_pattern = re.compile(r"import\s+['\"]([^'\"]+)['\"]")
    info["imports"].extend(import_pattern.findall(content))

def _parse_js(content, info):
    # Classes
    class_pattern = re.compile(r'class\s+(\w+)')
    info["classes"].extend(class_pattern.findall(content))
    
    # Functions
    func_pattern = re.compile(r'function\s+(\w+)')
    info["functions"].extend(func_pattern.findall(content))
    
    const_pattern = re.compile(r'const\s+(\w+)\s*=\s*(?:async\s*)?(?:function|\(.*\)\s*=>)')
    info["functions"].extend(const_pattern.findall(content))

    # Imports
    import_pattern = re.compile(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]")
    info["imports"].extend(import_pattern.findall(content))

def generate_markdown(file_infos):
    lines = []
    
    # Inject Agentic Guide First
    if AGENTIC_GUIDE:
        lines.append(AGENTIC_GUIDE)
        lines.append("\n---\n")

    lines.append("# Repository Map\n")
    lines.append("This file is auto-generated using `scripts/repo_mapper.py`.\n")
    
    # Organize by top-level directory
    structure = {}
    for info in file_infos:
        parts = Path(info["path"]).parts
        if len(parts) > 1:
            top_level = parts[0]
            if top_level not in structure:
                structure[top_level] = []
            structure[top_level].append(info)
        else:
            if "root" not in structure:
                structure["root"] = []
            structure["root"].append(info)

    for section in sorted(structure.keys()):
        section_title = section.replace('-', ' ').title()
        if section == "root":
            lines.append(f"\n## Root Files\n")
        else:
            lines.append(f"\n## {section_title}\n")
        
        # Inject Context if available
        if section in SECTION_CONTEXT:
             lines.append(f"\n{SECTION_CONTEXT[section]}\n")

        last_sub_context = None
        for info in sorted(structure[section], key=lambda x: x["path"]):
            # Check for sub-context (e.g. services-api-gateway)
            parts = Path(info["path"]).parts
            if len(parts) >= 2:
                sub_context_key = f"{parts[0]}-{parts[1]}"
                if sub_context_key != last_sub_context and sub_context_key in SECTION_CONTEXT:
                    lines.append(f"\n{SECTION_CONTEXT[sub_context_key]}\n")
                    last_sub_context = sub_context_key

            lines.append(f"\n### [`{info['path']}`]({info['path']})")
            if info["description"]:
                lines.append(f"\n> {info['description'].strip().splitlines()[0]}") # First line of docstring
            
            # Routes (Priority for Agents)
            if info["routes"]:
                 lines.append(f"- **API Routes**: `{'`, `'.join(info['routes'])}`")

            # Data Models
            if info["models"]:
                lines.append(f"- **Data Models**: `{', '.join(info['models'])}`")

            if info["classes"]:
                lines.append(f"- **Classes**: `{', '.join(info['classes'][:5])}`" + (", ..." if len(info["classes"])>5 else ""))
            
            if info["functions"]:
                # Limit to 5 important ones to avoid clutter
                valid_funcs = [f for f in info["functions"] if len(f) > 2] # Filter tiny noise
                if valid_funcs:
                    lines.append(f"- **Functions**: `{', '.join(valid_funcs[:5])}`" + (", ..." if len(valid_funcs)>5 else ""))
            
            if info["imports"]:
                # Filter internal vs external imports could be cool, but just listing first few is good for now
                relevant_imports = [i for i in info["imports"] if not i.startswith("typing") and not i.startswith("os") and not i.startswith("sys")]
                if relevant_imports:
                    lines.append(f"- **Dependencies**: `{', '.join(relevant_imports[:5])}`" + (", ..." if len(relevant_imports)>5 else ""))


    return "\n".join(lines)

def main():
    print(f"Scanning repository: {REPO_ROOT}")
    file_infos = []
    
    for root, dirs, files in os.walk(REPO_ROOT):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES:
                continue
            
            file_path = Path(root) / file
            if file_path.suffix in EXTENSIONS:
                print(f"Processing: {file_path.relative_to(REPO_ROOT)}")
                info = get_file_info(file_path)
                file_infos.append(info)
    
    markdown = generate_markdown(file_infos)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"Repository map generated at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
