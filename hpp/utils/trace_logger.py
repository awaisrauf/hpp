"""
Trace logger: records every LLM turn, REPL execution and VLM call of a run to
<output_dir>/<file_name>_trace.json, and prints a colored summary to the terminal.
"""

import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import threading
import sys


# ANSI Color codes for terminal
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class TraceLogger:
    """Logger that captures complete execution trace with both file output and enhanced terminal display."""
    
    def __init__(self, output_dir: str, file_name: str, enabled: bool = True, terminal_output: bool = True):
        self.enabled = enabled
        self.terminal_output = terminal_output
        
        if not enabled:
            return
            
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_name = file_name
        self.json_file = self.output_dir / f"{file_name}_trace.json"
        
        # Initialize trace structure
        self.trace = {
            "metadata": {
                "file_name": file_name,
                "start_time": None,
                "end_time": None,
                "total_time": None,
                "status": "in_progress",
                "main_model": None,
                "sub_model": None,
                "video_path": None,
                "query": None,
                "system_prompt": None,
                "max_iterations": None,
                "total_iterations": 0,
                "final_answer_iteration": None,
                "question_metadata": None,
            },
            "iterations": [],
            "final_answer": None,
            "error": None
        }
        
        # Counter for sub-LLM calls and REPL executions across all iterations
        self.total_sub_llm_calls = 0
        self.total_repl_executions = 0
        self._lock = threading.Lock()
        
        # Tracking variables for current iteration/execution (used by sub-LLM)
        self._current_iteration = None
        self._current_execution = None
    
    def _print(self, text: str, color: str = None, bold: bool = False, dim: bool = False):
        """Print colored text to terminal."""
        if not self.terminal_output:
            return
            
        prefix = ""
        if bold:
            prefix += Colors.BOLD
        if dim:
            prefix += Colors.DIM
        if color:
            prefix += getattr(Colors, color.upper(), "")
        
        suffix = Colors.RESET if prefix else ""
        print(f"{prefix}{text}{suffix}")
        sys.stdout.flush()
    
    def _separator(self, char: str = "=", length: int = 100, color: str = "CYAN"):
        """Print a separator line."""
        self._print(char * length, color=color)
    
    def start_session(self, metadata: Dict[str, Any]):
        """Initialize session with metadata and print header."""
        if not self.enabled:
            return
            
        with self._lock:
            self.trace["metadata"].update({
                "start_time": datetime.now().isoformat(),
                "main_model": metadata.get("main_model"),
                "sub_model": metadata.get("sub_model"),
                "video_path": metadata.get("video_path"),
                "query": metadata.get("query"),
                "system_prompt": metadata.get("system_prompt"),
                "max_iterations": metadata.get("max_iterations"),
                "ground_truth": metadata.get("ground_truth"),
                "ground_truth_other": metadata.get("ground_truth_other"),
                "question_metadata": metadata.get("question_metadata"),
            })
            self._save()
        
        if self.terminal_output:
            self._print("\n")
            self._separator("=", color="BRIGHT_CYAN")
            self._print(f"{'🚀 NEW HPP SESSION':^100}", color="BRIGHT_CYAN", bold=True)
            self._separator("=", color="BRIGHT_CYAN")
            self._print(f"Time:        {self.trace['metadata']['start_time']}", color="CYAN")
            self._print(f"Main Model:  {metadata.get('main_model')}", color="CYAN")
            self._print(f"Sub Model:   {metadata.get('sub_model')}", color="CYAN")
            if metadata.get('video_path'):
                video_name = Path(metadata['video_path']).name
                self._print(f"Video:       {video_name}", color="CYAN")
            self._separator("-", color="BRIGHT_BLACK")
            self._print(f"QUERY: {metadata.get('query')}", color="YELLOW", bold=True)
            self._separator("=", color="BRIGHT_CYAN")
            self._print("")
    
    def start_iteration(self, iteration_number: int) -> Dict:
        """Start a new iteration."""
        if not self.enabled:
            return {}

        with self._lock:
            iteration = {
                "iteration": iteration_number,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration": None,
                "input_messages": None,  # Full prompt sent to model
                "input_messages_formatted": None,  # Human-readable version
                "model_response": {
                    "raw_text": None,
                    "has_code": False,
                    "code_blocks": []
                },
                "repl_executions": [],
                "final_answer_found": False
            }
            self.trace["iterations"].append(iteration)
            self.trace["metadata"]["total_iterations"] = iteration_number

        if self.terminal_output:
            self._print("")
            self._separator("─", 100, color="BRIGHT_BLUE")
            self._print(f"📍 ITERATION {iteration_number} / {self.trace['metadata']['max_iterations']}",
                       color="BRIGHT_BLUE", bold=True)
            self._separator("─", 100, color="BRIGHT_BLUE")

        return iteration

    def log_input_messages(self, iteration_number: int, messages: List[Dict[str, Any]]):
        """Log the full input messages sent to the model for this iteration."""
        if not self.enabled:
            return

        with self._lock:
            iteration = self._get_iteration(iteration_number)

            # Store raw messages
            iteration["input_messages"] = messages

            # Create human-readable formatted version
            formatted_parts = []
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")

                # Truncate very long content for readability
                if len(content) > 2000:
                    display_content = content[:2000] + f"\n\n... [truncated, {len(content)} chars total]"
                else:
                    display_content = content

                formatted_parts.append(f"═══ MESSAGE {i+1} ({role}) ═══\n{display_content}")

            iteration["input_messages_formatted"] = "\n\n".join(formatted_parts)
            iteration["input_message_count"] = len(messages)
            self._save()

        if self.terminal_output:
            self._print(f"📨 Input: {len(messages)} messages to model", color="CYAN", dim=True)

    def log_model_response(self, iteration_number: int, response: str, has_code: bool,
                           code_blocks: List[str] = None, raw_with_reasoning: str = None):
        """Log the main model's response."""
        if not self.enabled:
            return

        with self._lock:
            iteration = self._get_iteration(iteration_number)
            iteration["model_response"] = {
                "raw_text": response,
                "raw_with_reasoning": raw_with_reasoning,
                "has_code": has_code,
                "code_blocks": code_blocks or []
            }
            self._save()
        
        if self.terminal_output:
            self._print("")
            self._print("🤖 MODEL RESPONSE:", color="MAGENTA", bold=True)
            self._print("┌" + "─"*98 + "┐", color="BRIGHT_BLACK")
            
            # Print response line by line (truncate for terminal)
            lines = response.split('\n')
            display_lines = lines[:40]  # Show first 40 lines
            
            for line in display_lines:
                # Truncate long lines
                if len(line) > 98:
                    line = line[:95] + "..."
                self._print(f"│ {line:<97}│", color="WHITE")
            
            if len(lines) > 40:
                self._print(f"│ {f'... ({len(lines) - 40} more lines)':^97}│", color="YELLOW")
            
            self._print("└" + "─"*98 + "┘", color="BRIGHT_BLACK")
            
            if has_code:
                self._print(f"📝 Contains {len(code_blocks or [])} code block(s)", color="YELLOW")
    
    def start_repl_execution(self, iteration_number: int, code: str) -> Dict:
        """Start logging a REPL execution."""
        if not self.enabled:
            return {}
            
        with self._lock:
            self.total_repl_executions += 1
            execution_number = self.total_repl_executions
            
            iteration = self._get_iteration(iteration_number)
            
            execution = {
                "execution_number": execution_number,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration": None,
                "code": code,
                "stdout": None,
                "stderr": None,
                "sub_llm_calls": []
            }
            
            iteration["repl_executions"].append(execution)
        
        if self.terminal_output:
            self._print("")
            self._print(f"💻 REPL EXECUTION #{execution_number}", color="BLUE", bold=True)
            self._print("Code:", color="CYAN")
            self._print("┌─────", color="BRIGHT_BLACK")
            
            code_lines = code.split('\n')
            for i, line in enumerate(code_lines[:15], 1):  # Show first 15 lines
                self._print(f"{i:3d} │ {line}", color="CYAN", dim=True)
            
            if len(code_lines) > 15:
                self._print(f"... │ ({len(code_lines) - 15} more lines)", color="YELLOW", dim=True)
            
            self._print("└─────", color="BRIGHT_BLACK")
        
        return execution
    
    def log_sub_llm_call(self, iteration_number: int, execution_number: int, 
                        prompt: str, image_paths: List[str], response: str):
        """Log a sub-LLM call within a REPL execution."""
        if not self.enabled:
            return
            
        with self._lock:
            self.total_sub_llm_calls += 1
            call_number = self.total_sub_llm_calls
            
            iteration = self._get_iteration(iteration_number)
            execution = self._get_repl_execution(iteration, execution_number)
            
            img_list = image_paths if isinstance(image_paths, list) else [image_paths] if image_paths else []
            
            sub_call = {
                "call_number": call_number,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "image_paths": img_list,
                "num_images": len(img_list),
                "response": response
            }
            
            execution["sub_llm_calls"].append(sub_call)
            self._save()
        
        if self.terminal_output:
            self._print("")
            self._print(f"  🔍 SUB-LLM CALL #{call_number}", color="BRIGHT_YELLOW", bold=True)
            
            # Show prompt (truncated)
            prompt_display = prompt[:150] + "..." if len(prompt) > 150 else prompt
            self._print(f"  Prompt: {prompt_display}", color="YELLOW")
            
            # Show images if present
            if img_list:
                self._print(f"  Images: {len(img_list)} image(s)", color="YELLOW")
                for i, img in enumerate(img_list[:3], 1):
                    self._print(f"    {i}. {Path(img).name}", color="YELLOW", dim=True)
                if len(img_list) > 3:
                    self._print(f"    ... and {len(img_list) - 3} more", color="YELLOW", dim=True)
            
            # Show response (truncated)
            response_lines = response.split('\n')
            self._print("  Response:", color="GREEN")
            for line in response_lines[:8]:
                display_line = line[:90] + "..." if len(line) > 90 else line
                self._print(f"    {display_line}", color="GREEN")
            if len(response_lines) > 8:
                self._print(f"    ... ({len(response_lines) - 8} more lines)", color="YELLOW", dim=True)
    
    def end_repl_execution(self, iteration_number: int, execution_number: int, 
                          stdout: str, stderr: str, execution_time: float):
        """End logging a REPL execution."""
        if not self.enabled:
            return
            
        with self._lock:
            iteration = self._get_iteration(iteration_number)
            execution = self._get_repl_execution(iteration, execution_number)
            
            execution["end_time"] = datetime.now().isoformat()
            execution["duration"] = execution_time
            execution["stdout"] = stdout
            execution["stderr"] = stderr
            
            self._save()
        
        if self.terminal_output:
            # Show output
            if stderr:
                lines = stderr.split('\n')
                self._print("Error:", color="RED", bold=True)
                for line in lines[:10]:
                    self._print(f"  {line}", color="RED")
                if len(lines) > 10:
                    self._print(f"  ... ({len(lines) - 10} more lines)", color="YELLOW")
            elif stdout:
                lines = stdout.split('\n')
                self._print("Output:", color="GREEN")
                for line in lines[:12]:
                    self._print(f"  {line}", color="GREEN")
                if len(lines) > 12:
                    self._print(f"  ... ({len(lines) - 12} more lines)", color="YELLOW")
            
            self._print(f"⏱️  Execution time: {execution_time:.3f}s", color="CYAN", dim=True)
    
    def end_iteration(self, iteration_number: int, final_answer_found: bool = False):
        """End an iteration."""
        if not self.enabled:
            return
            
        with self._lock:
            iteration = self._get_iteration(iteration_number)
            iteration["end_time"] = datetime.now().isoformat()
            
            # Calculate duration
            start = datetime.fromisoformat(iteration["start_time"])
            end = datetime.fromisoformat(iteration["end_time"])
            iteration["duration"] = (end - start).total_seconds()
            
            iteration["final_answer_found"] = final_answer_found
            
            if final_answer_found:
                self.trace["metadata"]["final_answer_iteration"] = iteration_number
            
            self._save()
        
        if self.terminal_output and final_answer_found:
            self._print(f"\n✅ Final answer found in iteration {iteration_number}!", color="BRIGHT_GREEN", bold=True)
    
    def end_session(self, final_answer: str = None, error: str = None, status: str = "completed", usage: dict = None):
        """End the session."""
        if not self.enabled:
            return

        with self._lock:
            self.trace["metadata"]["end_time"] = datetime.now().isoformat()
            self.trace["metadata"]["status"] = status
            self.trace["final_answer"] = final_answer
            self.trace["error"] = error
            if usage is not None:
                self.trace["usage"] = usage
            
            # Calculate total duration
            if self.trace["metadata"]["start_time"]:
                start = datetime.fromisoformat(self.trace["metadata"]["start_time"])
                end = datetime.fromisoformat(self.trace["metadata"]["end_time"])
                self.trace["metadata"]["total_time"] = (end - start).total_seconds()
            
            self.trace["metadata"]["total_sub_llm_calls"] = self.total_sub_llm_calls
            self.trace["metadata"]["total_repl_executions"] = self.total_repl_executions
            
            self._save()
        
        if self.terminal_output:
            self._print("")
            self._separator("=", color="BRIGHT_GREEN")
            self._print(f"{'✨ FINAL ANSWER':^100}", color="BRIGHT_GREEN", bold=True)
            self._separator("=", color="BRIGHT_GREEN")
            
            if final_answer:
                self._print(f"\n{final_answer}\n", color="GREEN", bold=True)
            elif error:
                self._print(f"\nError: {error}\n", color="RED")
            
            self._separator("=", color="BRIGHT_GREEN")
            self._print(f"Total Time:         {self.trace['metadata']['total_time'] or 0:.2f}s", color="CYAN")
            self._print(f"Total Iterations:   {self.trace['metadata']['total_iterations']}", color="CYAN")
            self._print(f"REPL Executions:    {self.total_repl_executions}", color="CYAN")
            self._print(f"Sub-LLM Calls:      {self.total_sub_llm_calls}", color="CYAN")
            self._print(f"Status:             {status}", color="CYAN")
            self._separator("=", color="BRIGHT_GREEN")
            self._print(f"\n💾 Trace saved to: {self.json_file}", color="GREEN")
            self._print("")
    
    def log_repl_error(self, iteration: int, execution_number: int, code: str, stderr: str):
        """Append a REPL error to repl_errors.log for easy monitoring."""
        if not self.enabled or not stderr:
            return
        error_file = self.output_dir / "repl_errors.log"
        with self._lock:
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().isoformat()}] {self.file_name} | iter {iteration} exec {execution_number}\n")
                f.write("  Code:\n")
                for line in code.strip().split('\n'):
                    f.write(f"    {line}\n")
                f.write("  Error:\n")
                for line in stderr.strip().split('\n'):
                    f.write(f"    {line}\n")
                f.write(f"{'-'*60}\n")

    def _get_iteration(self, iteration_number: int) -> Dict:
        """Get iteration by number."""
        for iteration in self.trace["iterations"]:
            if iteration["iteration"] == iteration_number:
                return iteration
        raise ValueError(f"Iteration {iteration_number} not found")
    
    def _get_repl_execution(self, iteration: Dict, execution_number: int) -> Dict:
        """Get REPL execution by number."""
        for execution in iteration["repl_executions"]:
            if execution["execution_number"] == execution_number:
                return execution
        raise ValueError(f"REPL execution {execution_number} not found")
    
    def _save(self):
        """Save trace to JSON file."""
        if not self.enabled:
            return

        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)
