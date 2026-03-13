#!/bin/bash

print_boundary() {
   echo -e "\n...........$1............"
   echo -e "File: $2\n"
   cat "$2"
   echo -e "\n...........$1 END............\n"
}

find . -type f \( -name "*.py" -o -name "*.tsx" -o -name "*.ts" -o -name "*.jsx" -o -name "*.css" -o -name "*.html" -o -name "*.sh" -o -name "*.txt" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" -o -name "*.json" -o -name "*.sql" -o -name "Dockerfile" -o -name "requirements.txt" -o -name ".env" -o -name ".gitignore" \) -not -path "*/node_modules/*" -not -path "*/__pycache__/*" -not -path "*/dist/*" -not -path "*/build/*" -not -path "*/.git/*" -not -path "*/venv/*" -not -path "*/.venv/*" | grep -v -E '(lock|Lock|\.pyc$)' | while read -r file; do
    filename=$(basename "$file")
    print_boundary "$filename" "$file"
done