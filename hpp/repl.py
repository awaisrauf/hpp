"""
The probing environment: a persistent Python REPL pre-loaded with video probing
functions and a VLM. The coding LLM writes ```repl``` blocks that run here.

Functions exposed to the LLM (documented in hpp/prompt.py):
  structure   get_video_info, get_shots, get_scenes, get_clusters, get_shot_detail
  frames      get_frame_at, get_frames, sample, get_overview, zoom, sample_shot, get_keyframes
  clips       get_clip
  search      search_in_shots, search_in_scenes
  subtitles   subtitles (variable), search_subtitles, find_times, get_subtitles, summarize_subtitles
  perception  vlm_query
  answer      FINAL / FINAL_VAR
"""

import io
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from typing import Optional


class VLM:
    """Small vision-language model used for localized perception (e.g. Qwen3-VL-2B)."""

    def __init__(self, model: str, trace_logger=None, vlm_client=None, usage_tracker=None):
        self.model = model
        self.trace_logger = trace_logger
        self.usage_tracker = usage_tracker
        if vlm_client is not None:
            self.client = vlm_client
        else:
            from hpp.utils.vlm import HFVLMClient
            self.client = HFVLMClient(model=model)

    def completion(self, prompt, images_paths=None, video_path=None, timeout=None) -> str:
        try:
            response = self.client.completion(
                messages=prompt,
                images_paths=images_paths,
                video_path=video_path,
                timeout=timeout,
            )
            if self.usage_tracker:
                self.usage_tracker.log_sub_llm(
                    getattr(self.client, "last_input_tokens", 0),
                    getattr(self.client, "last_output_tokens", 0),
                )
            if self.trace_logger:
                image_list = []
                if images_paths:
                    image_list = images_paths if isinstance(images_paths, list) else [images_paths]
                self.trace_logger.log_sub_llm_call(
                    iteration_number=self.trace_logger._current_iteration,
                    execution_number=self.trace_logger._current_execution,
                    prompt=str(prompt),
                    image_paths=image_list,
                    response=response,
                )
            return response
        except Exception as e:
            return f"Error making LLM query: {str(e)}"


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float = None


# Built-ins available inside the REPL. eval/exec/compile/input/globals/locals are removed to keep the
# LLM on the provided functions. This is NOT a security sandbox: the code runs with your permissions.
_SAFE_BUILTINS = {
    'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
    'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
    'type': type, 'isinstance': isinstance, 'enumerate': enumerate,
    'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
    'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
    'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct,
    'repr': repr, 'ascii': ascii, 'format': format,
    '__import__': __import__, 'open': open,
    'any': any, 'all': all, 'hasattr': hasattr, 'getattr': getattr,
    'setattr': setattr, 'delattr': delattr, 'dir': dir, 'vars': vars,
    'range': range, 'reversed': reversed, 'slice': slice, 'iter': iter, 'next': next,
    'pow': pow, 'divmod': divmod, 'complex': complex, 'bytes': bytes,
    'bytearray': bytearray, 'memoryview': memoryview, 'hash': hash, 'id': id,
    'callable': callable, 'issubclass': issubclass, 'super': super,
    'property': property, 'staticmethod': staticmethod, 'classmethod': classmethod,
    'object': object,
    'BaseException': BaseException, 'Exception': Exception,
    'ArithmeticError': ArithmeticError, 'LookupError': LookupError,
    'EnvironmentError': EnvironmentError, 'AssertionError': AssertionError,
    'NotImplementedError': NotImplementedError, 'UnicodeError': UnicodeError,
    'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
    'IndexError': IndexError, 'AttributeError': AttributeError,
    'FileNotFoundError': FileNotFoundError, 'OSError': OSError, 'IOError': IOError,
    'RuntimeError': RuntimeError, 'NameError': NameError, 'ImportError': ImportError,
    'StopIteration': StopIteration, 'GeneratorExit': GeneratorExit,
    'SystemExit': SystemExit, 'KeyboardInterrupt': KeyboardInterrupt,
    'Warning': Warning, 'UserWarning': UserWarning, 'DeprecationWarning': DeprecationWarning,
    'PendingDeprecationWarning': PendingDeprecationWarning, 'SyntaxWarning': SyntaxWarning,
    'RuntimeWarning': RuntimeWarning, 'FutureWarning': FutureWarning,
    'ImportWarning': ImportWarning, 'UnicodeWarning': UnicodeWarning,
    'BytesWarning': BytesWarning, 'ResourceWarning': ResourceWarning,
    'input': None, 'eval': None, 'exec': None, 'compile': None,
    'globals': None, 'locals': None,
}


class REPLEnv:
    def __init__(
        self,
        recursive_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        context_json: Optional[dict | list] = None,
        context_str: Optional[str] = None,
        video_path: Optional[str] = None,
        frame_metadata: Optional[dict] = None,
        subtitle_path: Optional[str] = None,
        setup_code: str = None,
        trace_logger=None,
        vlm_client=None,
        usage_tracker=None,
        gpu_lock=None,
        search_threshold: Optional[float] = None,
    ):
        self.usage_tracker = usage_tracker
        self._gpu_lock = gpu_lock              # optional threading.Lock around GPU work
        self.search_threshold = search_threshold  # min score to keep search results (None = keep all)
        self.temp_dir = tempfile.mkdtemp(prefix="repl_env_")
        self.video_path = video_path
        self.frame_metadata = frame_metadata or {}
        self.vlm = VLM(model=recursive_model, trace_logger=trace_logger,
                       vlm_client=vlm_client, usage_tracker=usage_tracker)

        self.globals = {'__builtins__': dict(_SAFE_BUILTINS)}
        self.locals = {}
        self._exec_namespace = None  # active namespace during exec (used by FINAL_VAR)

        self.load_context(context_json, context_str)
        self.load_video_context()
        self._bind_functions(subtitle_path)

        if setup_code:
            self.code_execution(setup_code)

    # ------------------------------------------------------------------
    # Functions visible to the LLM
    # ------------------------------------------------------------------
    def _bind_functions(self, subtitle_path):
        from hpp.utils.hierarchy import (
            get_shots as _get_shots, get_scenes as _get_scenes,
            get_clusters as _get_clusters, get_clip as _get_clip,
            get_video_metadata as _get_video_metadata,
        )
        from hpp.utils.sampling import (
            get_keyframes as _get_keyframes, get_frame_at as _get_frame_at,
            sample_uniform_frames as _sample_uniform_frames, zoom as _zoom,
            get_video_overview as _get_video_overview,
        )
        from hpp.utils.subtitles import (
            load_subtitles as _load_subtitles, get_subtitles as _get_subtitles,
            search_subtitles as _search_subtitles, find_times as _find_times,
            summarize_subtitles as _summarize_subtitles,
        )

        vp, od, g = self.video_path, self.temp_dir, self.globals

        def require_video():
            if vp is None:
                raise ValueError("No video loaded")

        def on_gpu(fn, *args, **kwargs):
            if self._gpu_lock:
                with self._gpu_lock:
                    return fn(*args, **kwargs)
            return fn(*args, **kwargs)

        # ── perception ───────────────────────────────────────────────
        def vlm_query(prompt: str, image_paths=None, video_path=None) -> str:
            """Query the VLM with image path(s) or a video clip path."""
            if image_paths is None:
                paths = None
            elif isinstance(image_paths, str):
                paths = [image_paths]
            else:
                paths = list(image_paths)
            return self.vlm.completion(prompt, images_paths=paths, video_path=video_path, timeout=240)

        # ── hierarchy ────────────────────────────────────────────────
        def get_shots():
            require_video()
            return _get_shots(vp, output_dir=od)

        def get_scenes():
            require_video()
            return on_gpu(_get_scenes, vp, output_dir=od)

        def get_clusters():
            require_video()
            return on_gpu(_get_clusters, vp, output_dir=od)

        def get_video_info():
            require_video()
            return _get_video_metadata(vp)

        def get_shot_detail(shot_id):
            require_video()
            for shot in _get_shots(vp, output_dir=od):
                if shot.get('shot_id') == shot_id:
                    frames = _sample_uniform_frames(vp, start=shot['start'], end=shot['end'], n=8, output_dir=od)
                    return {**shot, 'keyframes': frames}
            raise ValueError(f"Shot {shot_id} not found")

        # ── frame / clip sampling ────────────────────────────────────
        def get_frame_at(timestamp):
            require_video()
            return _get_frame_at(vp, timestamp, output_dir=od)

        def get_frames(start, end, n=10):
            require_video()
            return _sample_uniform_frames(vp, start=start, end=end, n=n, output_dir=od)

        def get_overview(n=10):
            require_video()
            return _get_video_overview(vp, n=n, output_dir=od)

        def zoom(target, context=0, n=5):
            require_video()
            return _zoom(vp, target, context=context, n=n, output_dir=od)

        def sample_shot(shot_id, n=8):
            require_video()
            for shot in _get_shots(vp, output_dir=od):
                if shot.get('shot_id') == shot_id:
                    return _sample_uniform_frames(vp, start=shot['start'], end=shot['end'], n=n, output_dir=od)
            raise ValueError(f"Shot {shot_id} not found")

        def sample(level="auto", ids=None, n=None):
            """Hierarchy-aware keyframe sampling over shots / scenes / clusters."""
            require_video()
            if level == "shots":
                items, id_key = _get_shots(vp, output_dir=od), 'shot_id'
            elif level == "scenes":
                items, id_key = _get_scenes(vp, output_dir=od), 'scene_id'
            elif level == "clusters":
                items, id_key = _get_clusters(vp, output_dir=od), 'cluster_id'
            else:  # auto: scenes for long videos, shots otherwise
                shots = _get_shots(vp, output_dir=od)
                if len(shots) > 15:
                    items, id_key = _get_scenes(vp, output_dir=od), 'scene_id'
                else:
                    items, id_key = shots, 'shot_id'
            if ids is not None:
                items = [it for it in items if it.get(id_key) in ids]
            if n is not None and len(items) > n:
                step = len(items) / n
                items = [items[int(i * step)] for i in range(n)]
            return _get_keyframes(items)

        def get_clip(start, end):
            require_video()
            return _get_clip(vp, start, end, output_dir=od)

        # ── semantic search ──────────────────────────────────────────
        # get_shots() first so the hierarchy's keyframes land in this REPL's temp dir
        def search_in_shots(query, n=5):
            require_video()
            if self.usage_tracker:
                self.usage_tracker.log_search()
            from hpp.utils.search import search_in_shots as _search_in_shots
            _get_shots(vp, output_dir=od)
            return on_gpu(_search_in_shots, vp, query, n=n, threshold=self.search_threshold)

        def search_in_scenes(query, n=5):
            require_video()
            if self.usage_tracker:
                self.usage_tracker.log_search()
            from hpp.utils.search import search_in_scenes as _search_in_scenes
            _get_shots(vp, output_dir=od)
            return on_gpu(_search_in_scenes, vp, query, n=n, threshold=self.search_threshold)

        # ── subtitles (pre-loaded as `subtitles`) ────────────────────
        subs = _load_subtitles(subtitle_path) if subtitle_path else None

        def search_subtitles(subs_, pattern):
            return [] if subs_ is None else _search_subtitles(subs_, pattern)

        def find_times(subs_, pattern):
            return [] if subs_ is None else _find_times(subs_, pattern)

        def get_subtitles(subs_, start=None, end=None):
            return [] if subs_ is None else _get_subtitles(subs_, start=start, end=end)

        def summarize_subtitles(subs_or_maxlines=None, max_lines=50):
            # accepts summarize_subtitles(subtitles) or summarize_subtitles(max_lines=N)
            if isinstance(subs_or_maxlines, list):
                s = subs_or_maxlines
            elif isinstance(subs_or_maxlines, int):
                s, max_lines = subs, subs_or_maxlines
            else:
                s = subs
            if s is None:
                return "No subtitles available."
            return _summarize_subtitles(s, max_lines=max_lines)

        # ── final answer ─────────────────────────────────────────────
        def final_var(variable_name: str) -> str:
            """Return a REPL variable (or a bare option letter) as the final answer."""
            name = variable_name.strip().strip('"').strip("'").strip('\n').strip('\r')
            try:
                if self._exec_namespace and name in self._exec_namespace:
                    return str(self._exec_namespace[name])
                if name in self.locals:
                    return str(self.locals[name])
                if len(name) == 1 and name in "ABCDE":
                    return name
                return f"Error: Variable '{name}' not found in REPL environment"
            except Exception as e:
                return f"Error retrieving variable '{name}': {str(e)}"

        g.update({
            'vlm_query': vlm_query,
            'get_shots': get_shots, 'get_scenes': get_scenes, 'get_clusters': get_clusters,
            'get_video_info': get_video_info, 'get_video_metadata': get_video_info,
            'get_shot_detail': get_shot_detail,
            'get_frame_at': get_frame_at, 'get_frames': get_frames, 'get_overview': get_overview,
            'zoom': zoom, 'sample_shot': sample_shot, 'sample': sample, 'get_keyframes': _get_keyframes,
            'get_clip': get_clip,
            'search_in_shots': search_in_shots, 'search_in_scenes': search_in_scenes,
            'subtitles': subs if subs else None,
            'search_subtitles': search_subtitles, 'find_times': find_times,
            'get_subtitles': get_subtitles, 'summarize_subtitles': summarize_subtitles,
            'FINAL_VAR': final_var, 'FINAL': final_var,
        })

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def load_context(self, context_json: Optional[dict | list] = None, context_str: Optional[str] = None):
        if context_json is not None:
            path = os.path.join(self.temp_dir, "context.json")
            with open(path, "w") as f:
                json.dump(context_json, f, indent=2)
            self.code_execution(f"import json\nwith open(r'{path}', 'r') as f:\n    context = json.load(f)\n")
        if context_str is not None:
            path = os.path.join(self.temp_dir, "context.txt")
            with open(path, "w") as f:
                f.write(context_str)
            self.code_execution(f"import os\nwith open(r'{path}', 'r') as f:\n    context = f.read()\n")

    def load_video_context(self):
        if self.video_path:
            self.code_execution(f"video_path = r'{self.video_path}'")
        if self.frame_metadata:
            self.code_execution(f"video_metadata = {self.frame_metadata}")

    def __del__(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def code_execution(self, code) -> REPLResult:
        """Run a code block notebook-style; the value of a trailing expression is printed."""
        start_time = time.time()
        stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        old_cwd = os.getcwd()
        sys.stdout, sys.stderr = stdout_buffer, stderr_buffer
        try:
            os.chdir(self.temp_dir)
        except OSError:
            pass
        # Warnings would reach the LLM through stderr; the paper runs suppressed them.
        warn_ctx = warnings.catch_warnings()
        warn_ctx.__enter__()
        warnings.simplefilter("ignore")
        try:
            lines = code.split('\n')
            import_lines = [l for l in lines if l.startswith(('import ', 'from '))]
            other_lines = [l for l in lines if not l.startswith(('import ', 'from '))]

            # imports go into globals so later blocks can use them
            if import_lines:
                exec('\n'.join(import_lines), self.globals, self.globals)

            if other_lines:
                other_code = '\n'.join(other_lines)
                ns = {**self.globals, **self.locals}
                self._exec_namespace = ns
                non_comment = [l for l in other_lines if l and not l.startswith('#')]

                if non_comment:
                    last_line = non_comment[-1]
                    is_expression = (
                        not last_line.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ',
                                                  'try:', 'with ', 'return ', 'yield ', 'break', 'continue', 'pass'))
                        and '=' not in last_line.split('#')[0]
                        and not last_line.endswith(':')
                        and not last_line.startswith('print(')
                    )
                    if is_expression:
                        try:
                            if len(non_comment) > 1:
                                idx = other_lines.index(last_line)
                                if idx > 0:
                                    exec('\n'.join(other_lines[:idx]), ns, ns)
                            result = eval(last_line, ns, ns)
                            if result is not None:
                                print(repr(result))
                        except:  # noqa: E722  (fall back to running the whole block)
                            exec(other_code, ns, ns)
                    else:
                        exec(other_code, ns, ns)
                else:
                    exec(other_code, ns, ns)

                for key, value in ns.items():
                    if key not in self.globals:
                        self.locals[key] = value
                self._exec_namespace = None

            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
        except Exception as e:
            self._exec_namespace = None
            stderr_content = stderr_buffer.getvalue() + str(e)
            stdout_content = stdout_buffer.getvalue()
        finally:
            warn_ctx.__exit__(None, None, None)
            sys.stdout, sys.stderr = old_stdout, old_stderr
            try:
                os.chdir(old_cwd)
            except OSError:
                pass

        self.locals['_stdout'] = stdout_content
        self.locals['_stderr'] = stderr_content
        return REPLResult(stdout_content, stderr_content, self.locals.copy(), time.time() - start_time)
