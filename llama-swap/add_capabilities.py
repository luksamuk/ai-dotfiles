#!/usr/bin/env python3
"""
Add 'capabilities' section to llama-swap model YAML files based on metadata.features.
Keeps metadata.features as-is for reference.
Uses text-level insertion to preserve comments and formatting.
"""
import re
import sys
import os

def features_to_capabilities(features: dict, context_length) -> list[str]:
    """Convert metadata.features to the new capabilities YAML lines."""
    lines = []
    
    completion = features.get('completion', False)
    vision = features.get('vision', False)
    audio = features.get('audio', False)
    embedding = features.get('embedding', False)
    tools = features.get('tools', False)
    
    # Determine input modalities
    in_modes = []
    out_modes = []
    
    # Embedding-only models
    if embedding and not completion:
        in_modes = ['text']
        out_modes = []
    # TTS models: text in, audio out
    elif audio and not completion:
        in_modes = ['text']
        out_modes = ['audio']
    # ASR models: audio+text in, text out
    elif audio and completion and not vision:
        in_modes = ['text', 'audio']
        out_modes = ['text']
    # Vision models (some also support audio)
    elif vision:
        in_modes = ['text', 'image']
        if audio:
            in_modes.append('audio')
        out_modes = ['text']
    # Standard chat/completion models
    elif completion:
        in_modes = ['text']
        out_modes = ['text']
    else:
        in_modes = ['text']
        out_modes = ['text']
    
    # Build YAML lines
    lines.append("  capabilities:")
    lines.append(f"    in: {in_modes}")
    lines.append(f"    out: {out_modes}")
    lines.append(f"    tools: {str(tools).lower()}")
    lines.append(f"    reranker: false")
    lines.append(f"    context: {int(context_length) if context_length else 0}")
    
    return lines


def parse_features_from_text(content: str) -> tuple[dict, int]:
    """Parse metadata.features and metadata.context_length from YAML text."""
    features = {}
    context_length = 0
    
    # Find features block
    in_features = False
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('features:'):
            in_features = True
            # Handle inline: features: {}
            if '{}' in stripped:
                in_features = False
            continue
        
        if in_features:
            # Check if we've left the features block (next key at same or lower indent)
            if stripped and not stripped.startswith('#') and ':' in stripped and not stripped.startswith('-'):
                # Could be a feature like "completion: true"
                if line.startswith('    '):  # Inside features
                    key_val = stripped.split(':')
                    if len(key_val) == 2:
                        key = key_val[0].strip()
                        val = key_val[1].strip().rstrip(',')
                        features[key] = val.lower() == 'true'
                else:
                    in_features = False
            elif stripped.startswith('}') or stripped == '':
                if stripped == '}':
                    in_features = False
    
    # Find context_length
    m = re.search(r'context_length:\s*(\d+)', content)
    if m:
        context_length = int(m.group(1))
    
    return features, context_length


def add_capabilities_to_file(filepath: str, dry_run: bool = False) -> str:
    """Add capabilities to a model YAML file using text insertion."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if capabilities already exists
    if re.search(r'^\s{2}capabilities:', content, re.MULTILINE):
        return f"SKIP {os.path.basename(filepath)}: capabilities already exists"
    
    features, context_length = parse_features_from_text(content)
    if not features:
        return f"SKIP {os.path.basename(filepath)}: no features found"
    
    cap_lines = features_to_capabilities(features, context_length)
    cap_text = '\n'.join(cap_lines)
    
    # Insert capabilities before the metadata block
    # Find "  metadata:" line and insert before it
    metadata_match = re.search(r'^(\s{2})metadata:', content, re.MULTILINE)
    if not metadata_match:
        return f"SKIP {os.path.basename(filepath)}: no metadata block found"
    
    insert_pos = metadata_match.start()
    new_content = content[:insert_pos] + cap_text + '\n' + content[insert_pos:]
    
    if not dry_run:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return f"OK {os.path.basename(filepath)}: in={[x for x in cap_lines[1] if x.isalpha()]} → capabilities added"
    else:
        return f"DRY {os.path.basename(filepath)}: would add capabilities"


def main():
    dry_run = '--dry-run' in sys.argv
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    files_processed = 0
    files_skipped = 0
    
    for subdir in ['', '_disabled', '_removed']:
        target_dir = os.path.join(models_dir, subdir) if subdir else models_dir
        if not os.path.isdir(target_dir):
            continue
        
        label = subdir.replace('_', '') if subdir else 'active'
        print(f"\n=== {label.upper()} MODELS ===")
        
        for f in sorted(os.listdir(target_dir)):
            if not f.endswith('.yaml'):
                continue
            filepath = os.path.join(target_dir, f)
            result = add_capabilities_to_file(filepath, dry_run=dry_run)
            if result.startswith('OK') or result.startswith('DRY'):
                files_processed += 1
            else:
                files_skipped += 1
            print(result)
    
    action = "Would add" if dry_run else "Added"
    print(f"\n{action} capabilities to {files_processed} files, skipped {files_skipped}")


if __name__ == '__main__':
    main()