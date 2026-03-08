with open('c:/Users/HP/source/repos/final/online1.h', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
inserted = False
found_namespace = False

# First, remove the `};` at the very end of the file that was incorrectly added
if lines[-1].strip() == '};':
    lines.pop()

for i, line in enumerate(lines):
    if "namespace project_opencv_ajsum" in line and not inserted:
        # We found the start of the UI class
        new_lines.append("}; // END CAMERA INSTANCE\n\n")
        new_lines.append(line)
        inserted = True
        found_namespace = True
    else:
        new_lines.append(line)

with open('c:/Users/HP/source/repos/final/online1.h', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

if found_namespace:
    print("SUCCESS: Inserted closing brace before namespace.")
else:
    print("FAILED: Did not find namespace project_opencv_ajsum")
