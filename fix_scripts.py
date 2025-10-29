# fix_scripts.py
"""Fix Unicode issues in scripts"""
import os

def fix_file(filepath):
    """Remove emojis from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace emojis with ASCII
    replacements = {
        '🎯': '[TARGET]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '🚀': '[START]',
        '🎤': '[MIC]',
        '📦': '[PACKAGE]',
        '🧠': '[MODEL]',
        '⚠️': '[WARNING]',
        '💾': '[SAVE]',
        '📥': '[DOWNLOAD]',
        '📊': '[DATA]',
        '🏋️': '[TRAIN]',
        '⚡': '[SPIKE]',
        '🔧': '[BUILD]',
        '🎉': '[DONE]',
        '📋': '[INFO]',
        '📚': '[DOCS]'
    }
    
    for emoji, text in replacements.items():
        content = content.replace(emoji, text)
    
    # Save with proper encoding
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

# Fix the problematic files
files_to_fix = [
    'train_simple.py',
    'export_onnx_simple.py',
    'inference_demo.py',
    'week3_workflow.py'
]

for file in files_to_fix:
    if os.path.exists(file):
        fix_file(file)

print("Files fixed!")