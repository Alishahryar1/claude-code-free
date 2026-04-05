import os
import sys

# Fix escaped quotes in Python files
fixed_count = 0

for root, dirs, files in os.walk('providers'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file has escaped quotes
                if '\\\"' in content:
                    # Replace escaped quotes with proper quotes
                    fixed_content = content.replace('\\\"\\\"\\\"', '"""')
                    
                    if fixed_content != content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        
                        print(f'Fixed: {filepath}')
                        fixed_count += 1
            except Exception as e:
                print(f'Error processing {filepath}: {e}', file=sys.stderr)

print(f'\nTotal files fixed: {fixed_count}')
