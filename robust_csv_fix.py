#!/usr/bin/env python3
import re

def robust_csv_fix(input_file, output_file):
    """More robust CSV fixing that handles truncated lines and quote issues"""
    
    print(f"Robustly fixing CSV: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines to process: {len(lines)}")
    
    # Get expected column count from header
    header = lines[0].strip()
    expected_cols = header.count('","') + 1
    print(f"Expected columns: {expected_cols}")
    
    fixed_lines = []
    removed_count = 0
    fixed_count = 0
    
    # Add header
    fixed_lines.append(header)
    
    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check if line is properly formed
        # A valid CSV line should start and end with quotes for this format
        if not (line.startswith('"') and line.endswith('"')):
            print(f"Line {i}: Improperly formed - skipping")
            removed_count += 1
            continue
        
        # Count quotes - should be even for properly escaped CSV
        quote_count = line.count('"')
        if quote_count % 2 != 0:
            print(f"Line {i}: Odd quote count ({quote_count}) - attempting fix")
            # Try to fix by adding missing quote at the end
            if not line.endswith('"'):
                line += '"'
                fixed_count += 1
            else:
                print(f"Line {i}: Cannot fix odd quotes - skipping")
                removed_count += 1
                continue
        
        # Count columns by counting field separators
        # For quoted CSV, count ","" patterns (end quote + comma + start quote)
        col_count = line.count('","') + 1
        
        if col_count != expected_cols:
            print(f"Line {i}: Expected {expected_cols} columns, got {col_count}")
            
            # If too few columns, line is likely truncated - skip it
            if col_count < expected_cols:
                print(f"Line {i}: Truncated line - skipping")
                removed_count += 1
                continue
            
            # If too many columns, try to merge extras (shouldn't happen with quoted CSV)
            if col_count > expected_cols:
                print(f"Line {i}: Too many columns - skipping (complex case)")
                removed_count += 1
                continue
        
        # Additional validation: check for incomplete quoted fields
        # Look for patterns that suggest truncation
        suspicious_patterns = [
            r',"[^"]*$',  # Ends with comma-quote-text but no closing quote
            r'","[^"]*$',  # Ends with quote-comma-quote-text but no closing quote
        ]
        
        is_suspicious = False
        for pattern in suspicious_patterns:
            if re.search(pattern, line):
                print(f"Line {i}: Suspicious pattern found - skipping")
                removed_count += 1
                is_suspicious = True
                break
        
        if is_suspicious:
            continue
        
        # Line seems valid, add it
        fixed_lines.append(line)
        
        if i % 10000 == 0:
            print(f"Processed {i} lines...")
    
    # Write the fixed file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    print(f"Processing complete!")
    print(f"Original lines: {len(lines)}")
    print(f"Output lines: {len(fixed_lines)}")
    print(f"Fixed lines: {fixed_count}")
    print(f"Removed lines: {removed_count}")

if __name__ == "__main__":
    robust_csv_fix('data/data_clean_bigquery.csv', 'data/data_clean_bigquery_robust.csv') 