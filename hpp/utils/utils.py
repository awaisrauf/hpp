"""Parsing helpers for the agent loop: code blocks, FINAL answers, REPL output formatting."""

import re
from typing import List, Dict, Optional, Tuple, Any


def find_code_blocks(text: str) -> List[str]:
    """Return the contents of all ```repl``` code blocks in text (empty list if none)."""
    pattern = r'```repl\s*\n(.*?)\n```'
    results = []
    
    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)
    
    return results

def find_final_answer(text: str) -> Optional[Tuple[str, str]]:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return (type, content).
    Returns None if neither pattern is found.
    """
    # Check for FINAL_VAR pattern first — allow anywhere on a line,
    # optional surrounding markdown bold/backticks (e.g. **FINAL_VAR(x)**)
    final_var_pattern = r'[*`]*FINAL_VAR\((.*?)\)[*`]*'
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ('FINAL_VAR', match.group(1).strip())

    # Check for FINAL pattern — same relaxed matching
    final_pattern = r'[*`]*FINAL\((.*?)\)[*`]*'
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ('FINAL', match.group(1).strip())
    
    return None


def add_execution_result_to_messages(messages: List[Dict[str, str]], 
                                   code: str, 
                                   result: str,
                                   max_character_length: int = 100000,
                                   ) -> List[Dict[str, str]]:
    """
    Add code execution result to the conversation messages.
    
    Args:
        messages: Current conversation messages
        code: The code that was executed
        result: Result from code execution
        max_character_length: Maximum character length of the result

    Returns:
        Updated messages list
    """
    # Truncate result if it exceeds 100k characters
    if len(result) > max_character_length:
        result = result[:max_character_length] + "..."
    
    # Add the code execution result
    execution_message = {
        "role": "user",
        "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}"
    }
    messages.append(execution_message)
    return messages

def format_execution_result(
    stdout: str,
    stderr: str,
    locals_dict: Dict[str, Any],
    truncate_length: int = 100
) -> str:
    """
    Format the execution result as a string for display.
    
    Args:
        stdout: Standard output from execution
        stderr: Standard error from execution
        locals_dict: Local variables after execution
        truncate_length: Maximum length of the string to display per var
    """
    result_parts = []
    
    if stdout:
        result_parts.append(f"\n{stdout}")
    
    if stderr:
        result_parts.append(f"\n{stderr}")
    
    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in locals_dict.items():
        if not key.startswith('_') and not key in ['__builtins__', '__name__', '__doc__']:
            try:
                # Only show simple types or short representations
                if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                    if isinstance(value, str) and len(value) > truncate_length:
                        important_vars[key] = f"'{value[:truncate_length]}...'"
                    else:
                        important_vars[key] = repr(value)
            except:
                important_vars[key] = f"<{type(value).__name__}>"
    
    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")
    
    return "\n\n".join(result_parts) if result_parts else "No output"


def execute_code(repl_env, code: str) -> Tuple[str, Any]:
    """Execute code in the REPL environment. Returns (formatted result, REPLResult or None on failure)."""
    try:
        result = repl_env.code_execution(code)
        formatted_result = format_execution_result(
            result.stdout, result.stderr, result.locals
        )
        return formatted_result, result
        
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        return error_msg, None

def check_for_final_answer(response: str, repl_env) -> Optional[str]:
    """Check if response contains a final answer."""
    result = find_final_answer(response)
    if result is None:
        return None

    answer_type, content = result

    if answer_type == 'FINAL':
        return content.strip('"').strip("'")
    elif answer_type == 'FINAL_VAR':
        # Get the variable directly from the REPL environment
        try:
            # Strip spaces, quotes, and newlines from variable name
            variable_name = content.strip().strip('"').strip("'").strip('\n').strip('\r')

            # Check if variable exists in the REPL environment's locals
            if variable_name in repl_env.locals:
                variable_value = repl_env.locals[variable_name]
                return str(variable_value)
            else:
                print(f"FINAL_VAR: Variable '{variable_name}' not found in REPL environment")
                return None
        except Exception as e:
            print(f"FINAL_VAR: Error retrieving variable '{variable_name}': {str(e)}")
            return None

    return None

def convert_context_for_repl(context):
    """Split the context into (json-serialisable data, plain string); one of them is None."""
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None
    
    return context_data, context_str
