with open('app/pages/1_Data_Explorer.py', 'r') as file:
    content = file.readlines()

# Display lines with issue for reference
for i, line in enumerate(content[445:456]):
    print(f"Line {i+446}: '{line.rstrip()}' (length: {len(line) - len(line.lstrip())})")

# Fix the indentation for line 449 (fig.update_layout)
fixed_content = content.copy()
for i in range(448, 455):  # Fix indentation from line 449 to 454
    if fixed_content[i].strip().startswith('fig'):
        # Ensure proper indentation (12 spaces for fig.update_layout and its params)
        stripped = fixed_content[i].lstrip()
        fixed_content[i] = ' ' * 12 + stripped

# Write back the fixed content
with open('app/pages/1_Data_Explorer.py', 'w') as file:
    file.writelines(fixed_content)

print("\nFile has been fixed. Indentation issue corrected.") 