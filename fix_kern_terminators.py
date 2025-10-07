"""
Fix kern files by adding missing *- terminators to spines.

In kern format, each spine (column/voice) must be terminated with *-
This script validates and fixes kern strings to ensure proper termination.
"""

import re

def count_spines(kern_line: str) -> int:
    """Count the number of spines (tab-separated columns) in a line."""
    return len(kern_line.split('\t'))

def fix_kern_terminators(kern_content: str) -> tuple[str, bool]:
    """
    Fix kern content by ensuring all spines are properly terminated with *-.

    Args:
        kern_content: Raw kern format string

    Returns:
        Tuple of (fixed_kern_content, was_modified)
    """
    lines = kern_content.strip().split('\n')

    if not lines:
        return kern_content, False

    # Find the number of spines from the first data line
    num_spines = None
    for line in lines:
        if line.strip() and not line.startswith('!!!'):
            num_spines = count_spines(line)
            break

    if num_spines is None:
        return kern_content, False

    # Check if the last line is a terminator line
    last_line = lines[-1].strip()

    # Expected terminator line: one *- per spine
    expected_terminator = '\t'.join(['*-'] * num_spines)

    was_modified = False

    # Check various termination scenarios
    if last_line == expected_terminator:
        # Already properly terminated
        return kern_content, False

    elif '*-' in last_line:
        # Has some terminators but might be incomplete
        terminators = last_line.split('\t')
        if len(terminators) < num_spines:
            # Pad with missing terminators
            while len(terminators) < num_spines:
                terminators.append('*-')
            lines[-1] = '\t'.join(terminators)
            was_modified = True
        elif len(terminators) > num_spines:
            # Too many terminators, truncate
            lines[-1] = '\t'.join(terminators[:num_spines])
            was_modified = True
        else:
            # Right number but some might not be *-
            fixed_terminators = ['*-' if t != '*-' else t for t in terminators]
            if fixed_terminators != terminators:
                lines[-1] = '\t'.join(fixed_terminators)
                was_modified = True
    else:
        # No terminator line at all, add it
        lines.append(expected_terminator)
        was_modified = True

    return '\n'.join(lines), was_modified


def validate_kern(kern_content: str) -> tuple[bool, str]:
    """
    Validate kern content for proper termination.

    Returns:
        Tuple of (is_valid, error_message)
    """
    lines = kern_content.strip().split('\n')

    if not lines:
        return False, "Empty kern content"

    # Find number of spines
    num_spines = None
    for line in lines:
        if line.strip() and not line.startswith('!!!'):
            num_spines = count_spines(line)
            break

    if num_spines is None:
        return False, "No data lines found"

    # Check last line
    last_line = lines[-1].strip()
    terminators = last_line.split('\t')

    if len(terminators) != num_spines:
        return False, f"Expected {num_spines} terminators, found {len(terminators)}"

    if not all(t == '*-' for t in terminators):
        return False, f"Not all terminators are '*-': {terminators}"

    return True, "Valid"


if __name__ == "__main__":
    # Test examples
    test_kern_incomplete = """**kern\t**kern
*clefG2\t*clefF4
4c\t4C
4d\t4D
*-"""

    test_kern_missing = """**kern\t**kern
*clefG2\t*clefF4
4c\t4C
4d\t4D"""

    print("Test 1: Incomplete terminators")
    fixed, modified = fix_kern_terminators(test_kern_incomplete)
    print(f"Modified: {modified}")
    print(f"Valid: {validate_kern(fixed)[0]}")
    print(f"Result:\n{fixed}\n")

    print("Test 2: Missing terminators")
    fixed, modified = fix_kern_terminators(test_kern_missing)
    print(f"Modified: {modified}")
    print(f"Valid: {validate_kern(fixed)[0]}")
    print(f"Result:\n{fixed}\n")
