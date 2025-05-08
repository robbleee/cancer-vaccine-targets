import tokenize
import io

def check_indentation(filename):
    with open(filename, 'rb') as f:
        tokens = list(tokenize.tokenize(f.readline))
    
    prev_line = 0
    indent_levels = {}
    
    for token in tokens:
        if token.type == tokenize.INDENT:
            line = token.start[0]
            if line not in indent_levels:
                indent_levels[line] = len(token.string)
                
        elif token.type == tokenize.DEDENT:
            line = token.start[0]
            if line in indent_levels:
                del indent_levels[line]
        
        if token.start[0] != prev_line:
            prev_line = token.start[0]
            if token.type == tokenize.INDENT or token.type == tokenize.DEDENT:
                print(f"Line {prev_line}: {'INDENT' if token.type == tokenize.INDENT else 'DEDENT'}")
            
            # Check specifically for line numbers around 449
            if 446 <= prev_line <= 453:
                print(f"Line {prev_line}: token type={token.type}, token string='{token.string}'")
    
    return indent_levels

# Check the file
filename = "app/pages/1_Data_Explorer.py"
try:
    indentation = check_indentation(filename)
    print(f"Found indentation levels: {indentation}")
except Exception as e:
    print(f"Error: {str(e)}") 