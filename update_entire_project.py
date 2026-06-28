import os
import re

with open("entire_project.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    
    # Check if this is a header for a file
    if line.startswith("## ") and i + 2 < len(lines) and lines[i+2].startswith("`") and not lines[i+2].startswith("```"):
        filename = line[3:].strip()
        filepath = filename.replace("\\", os.path.sep).replace("/", os.path.sep)
        
        # Check if the file exists
        if os.path.exists(filepath) and os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                actual_content = f.read()
            
            # Add the empty line and the extension line
            new_lines.append(lines[i+1])
            new_lines.append(lines[i+2])
            
            # Add the actual content
            if actual_content and not actual_content.endswith("\n"):
                actual_content += "\n"
            new_lines.append(actual_content)
            
            # Skip lines in the original file until the closing backtick
            i += 3
            while i < len(lines):
                if lines[i].strip() == "`":
                    new_lines.append(lines[i])
                    break
                i += 1
        else:
            print("File not found:", filepath)
            
    i += 1

with open("entire_project.md", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("entire_project.md successfully updated with the latest file contents.")
