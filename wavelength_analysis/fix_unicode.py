"""Fix Unicode characters in analyzer.py"""

with open('core/analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Count occurrences
checkmarks = content.count('✓')
warnings = content.count('⚠')

print(f"Found {checkmarks} checkmark characters")
print(f"Found {warnings} warning symbols")

# Replace them
content = content.replace('✓ ', '')
content = content.replace('✓', '')
content = content.replace('⚠ ', 'WARNING: ')
content = content.replace('⚠', 'WARNING:')

# Write back
with open('core/analyzer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all Unicode characters")