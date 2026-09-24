"""Qwen2.5-1.5B-Instruct-Q4_K_M on llama.cpp -- streaming token generation.

Runs llama.cpp's official prebuilt `llama-server` binary as a local subprocess
and streams over HTTP/SSE on 127.0.0.1. No compiler and no Python-version-specific
wheel, so it works on Python 3.13 where `llama-cpp-python` has no prebuilt wheel
and its source build fails. Still fully offline -- the socket never leaves
loopback.

Context policy: rendered history is capped and the oldest turns are dropped,
because prompt length maps almost linearly onto time-to-first-token on CPU --
the latency the caller actually feels. `warmup()` deliberately ingests the real
system prompt so that prefix is cached before the call starts rather than being
paid on the caller's first question.

Generation is cancellable mid-stream: the orchestrator sets a stop_event on
barge-in and we close the HTTP response, which makes llama-server abort the slot
instead of burning cores on audio nobody will hear.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .prompts import TURN_REMINDER, build_system_prompt


@dataclass
class Turn:
    role: str          # "user" | "assistant"
    content: str
    tokens: int = 0


@dataclass
class GenerationStats:
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0
    interrupted: bool = False
    prompt_tokens: int = 0

    @property
    def tokens_per_second(self) -> float:
        gen_ms = max(1e-6, self.total_ms - self.ttft_ms)
        return (self.tokens - 1) / (gen_ms / 1000.0) if self.tokens > 1 else 0.0


# ==========================================================================
#  Shared history / context management
# ==========================================================================

class BaseLLM:
    def __init__(self, cfg, system_prompt: str | None = None) -> None:
        self.cfg = cfg
        # None = "not specified, build the default persona". "" = "explicitly
        # no system prompt, stay a plain generic model" (see build_llm's
        # system_prompt: none/generic/off option). `x or default` collapses
        # both cases to the same thing because "" is falsy, which is why this
        # is an explicit is-None check instead.
        if system_prompt is None:
            self.system_prompt = build_system_prompt(compact=True)
        else:
            self.system_prompt = system_prompt
        # Restated on every turn -- see prompts.TURN_REMINDER for why. This is
        # part of the Riya persona machinery (the "1-3 sentence" rule etc.),
        # so a generic model (empty system_prompt) gets none of it either.
        self.turn_reminder = TURN_REMINDER if self.system_prompt else ""
        self.history: list[Turn] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------- tokenizing
    def _count(self, text: str) -> int:
        """Backends override with a real tokenizer; this is the fallback."""
        return max(1, len(text) // 4)

    # ---------------------------------------------------------------- history
    def add_user(self, text: str) -> None:
        self.history.append(Turn("user", text, self._count(text)))

    def add_assistant(self, text: str) -> None:
        if text.strip():
            self.history.append(Turn("assistant", text, self._count(text)))

    def reset(self) -> None:
        self.history.clear()

    def drop_last_assistant(self) -> None:
        """Undo the most recent assistant turn (used when a reply is rejected)."""
        if self.history and self.history[-1].role == "assistant":
            self.history.pop()

    def drop_last_user(self, expected: str | None = None) -> bool:
        """Undo the most recent user turn, if it is still the last thing said.

        Used when a turn is abandoned before the caller heard anything: the text
        is going to be re-sent as part of the next turn, and leaving it here
        would put two user messages back to back.
        """
        if not self.history or self.history[-1].role != "user":
            return False
        if expected is not None and expected.strip() not in self.history[-1].content:
            return False
        self.history.pop()
        return True

    def recent_assistant_lines(self, limit: int = 6) -> list[str]:
        lines = [t.content for t in self.history if t.role == "assistant"]
        return lines[-limit:]

    def _trimmed_history(self) -> list[Turn]:
        """Drop oldest turns until under budget, keeping min_kept_turns."""
        budget = int(self.cfg["history_token_budget"])
        min_kept = int(self.cfg["min_kept_turns"])

        def _tok(t) -> int:
            # Token counts come back 0 when the backend does not report them,
            # which silently disabled trimming entirely: `total` stayed 0, the
            # loop never ran, and prompt length grew 892 -> 1091 over ten
            # turns of a live call, dragging time-to-first-token up with it.
            # Fall back to a character estimate so the budget always has
            # something to trim against.
            return max(int(t.tokens or 0), len(t.content) // 4)

        kept = list(self.history)
        total = sum(_tok(t) for t in kept)
        while total > budget and len(kept) > min_kept:
            dropped = kept.pop(0)
            total -= _tok(dropped)
            # Never leave a dangling assistant turn as the first message.
            if kept and kept[0].role == "assistant":
                total -= _tok(kept[0])
                kept.pop(0)
        return kept

    def build_messages(self, extra_note: str | None = None) -> list[dict]:
        """System + trimmed history, with any per-turn note pinned to the END.

        The note used to be concatenated onto the system prompt. That is the
        worst possible place for it: llama-server caches the KV prefix, and
        changing byte one of the system message invalidates the entire ~430-token
        persona, forcing a full re-ingest. The turns that most needed to be fast
        -- the ones right after a barge-in, which is when the note fires -- were
        the ones paying for a cold prefix. Appending to the final user message
        keeps the whole prefix warm and the model still reads it last, which is
        where a small model weights instructions most heavily anyway.
        """
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})

        # The standing reminder is attached to EVERY user turn, not just the last.
        #
        # Attaching it only to the last one looks cheaper but silently destroys
        # the prompt cache. Turn N sends [system][u1+R][a1]; turn N+1 sends
        # [system][u1][a1][a2][u3+R] -- u1 no longer carries R, so the longest
        # common prefix ends at u1 and llama-server re-ingests the entire
        # conversation. Measured effect: time-to-first-token went flat at
        # ~1940 ms regardless of how little had changed, because every turn was
        # a full cold ingest. Applying it uniformly keeps the prefix stable and
        # still puts the rules next to the generation point.
        #
        # It is never written into self.history, so the transcript stays clean.
        for t in self._trimmed_history():
            if t.role == "user" and self.turn_reminder:
                msgs.append({"role": "user", "content": f"{t.content}\n{self.turn_reminder}"})
            else:
                msgs.append({"role": t.role, "content": t.content})

        # A one-off note (interruption, correction) applies only to this turn and
        # sits at the very end, after the reminder.
        if extra_note:
            if msgs and msgs[-1]["role"] == "user":
                msgs[-1] = {
                    "role": "user",
                    "content": f"{msgs[-1]['content']}\n{extra_note}",
                }
            else:
                msgs.append({"role": "user", "content": extra_note})
        return msgs

    @property
    def context_tokens(self) -> int:
        return self._system_tokens + sum(t.tokens for t in self._trimmed_history())

    # ------------------------------------------------------------- interface
    def stream(self, stop_event=None, extra_system=None, stats=None) -> Iterator[str]:
        raise NotImplementedError

    def warmup(self) -> float:
        raise NotImplementedError

    def abort(self) -> None:
        """Tear down any in-flight request. Overridden by backends."""

    def close(self) -> None:
        pass


# ==========================================================================
#  Backend A: llama-server subprocess over loopback HTTP  (default)
# ==========================================================================

class LlamaServerLLM(BaseLLM):
    def __init__(self, model_path: str | Path, cfg, system_prompt: str | None = None) -> None:
        super().__init__(cfg, system_prompt)
        system_prompt = self.system_prompt

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {self.model_path}. Run: python scripts/download_models.py"
            )

        server_cfg = cfg.get("server", {}) or {}
        self.host = server_cfg.get("host", "127.0.0.1")
        self.port = int(server_cfg.get("port", 8080))
        self.base_url = f"http://{self.host}:{self.port}"
        self.startup_timeout = float(server_cfg.get("startup_timeout_s", 120))
        self.n_threads = int(cfg["n_threads"])
        self.n_threads_batch = int(cfg.get("n_threads_batch") or self.n_threads)
        self._process: subprocess.Popen | None = None
        self._owns_process = False
        self._spawn_args: list[str] = []
        self._log_handle = None
        self.log_path: Path | None = None

        binary = server_cfg.get("binary")
        self.binary_path = Path(binary) if binary else None

        if self._port_open():
            # Someone already has a server up (manually started, or a previous
            # run). Reuse it rather than fighting over the port.
            self._owns_process = False
        elif server_cfg.get("auto_start", True):
            self._spawn(server_cfg)
        else:
            raise RuntimeError(
                f"No llama-server reachable at {self.base_url} and auto_start is off.\n"
                f"Start one with:\n"
                f'  "{self.binary_path}" -m "{self.model_path}" '
                f"-c {cfg['n_ctx']} -t {self.n_threads} --host {self.host} --port {self.port}"
            )

        self._wait_until_ready()
        # Safe to use the real tokenizer here: nothing is generating yet.
        self._system_tokens = self._count_exact(system_prompt)
        # Handle to the in-flight SSE response, so a barge-in can tear it down
        # from another thread instead of waiting for a blocking read to return.
        self._active_response = None
        self._response_lock = threading.Lock()

    # ----------------------------------------------------------- process mgmt
    def _port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            return s.connect_ex((self.host, self.port)) == 0

    def _spawn(self, server_cfg: dict) -> None:
        if self.binary_path is None or not self.binary_path.exists():
            raise FileNotFoundError(
                f"llama-server binary not found at {self.binary_path}.\n"
                "Run: python scripts/download_models.py --only llamacpp"
            )

        # Core args only -- anything version-dependent lives in extra_args so a
        # flag that a given llama.cpp build does not recognise is one config
        # line to delete rather than a code change.
        args = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-c", str(int(self.cfg["n_ctx"])),
            "-b", str(int(self.cfg["n_batch"])),
            "-t", str(self.n_threads),          # --threads (generation)
            # Prompt ingest runs before any audio exists, so it never competes
            # with TTS and can use the whole machine even when generation
            # cannot. Tying both to the same number left cores idle during the
            # most latency-sensitive part of the turn.
            "-tb", str(self.n_threads_batch),   # --threads-batch (prompt ingest)
            "-ngl", "0",                        # CPU only
            "--host", self.host,
            "--port", str(self.port),
        ]
        args += [str(a) for a in (server_cfg.get("extra_args") or [])]
        self._spawn_args = args

        creationflags = 0
        if os.name == "nt":
            # Keep the server's console window from popping up over the caller.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        # Log to a file rather than DEVNULL so a failed start is diagnosable.
        self.log_path = self.model_path.parent / "llama-server.log"
        self._log_handle = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._log_handle.write("> " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n\n")
        self._log_handle.flush()

        self._process = subprocess.Popen(
            args,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        self._owns_process = True

    def _wait_until_ready(self) -> None:
        deadline = time.perf_counter() + self.startup_timeout
        last_error: Exception | None = None
        while time.perf_counter() < deadline:
            if self._process is not None and self._process.poll() is not None:
                tail = ""
                if self.log_path and self.log_path.exists():
                    try:
                        if self._log_handle:
                            self._log_handle.flush()
                        tail = "\n".join(
                            self.log_path.read_text(encoding="utf-8", errors="replace")
                            .splitlines()[-15:]
                        )
                    except Exception:
                        pass
                cmd = " ".join(f'"{a}"' if " " in a else a for a in self._spawn_args)
                raise RuntimeError(
                    f"llama-server exited immediately (code {self._process.returncode}).\n"
                    f"Command was:\n  {cmd}\n"
                    f"Log tail ({self.log_path}):\n{tail}\n"
                    "If a flag was rejected, remove it from llm.server.extra_args in config.yaml."
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(0.4)

        self.close()
        raise TimeoutError(
            f"llama-server did not become ready within {self.startup_timeout:.0f}s "
            f"({last_error})"
        )

    def close(self) -> None:
        if self._process is not None and self._owns_process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # ----------------------------------------------------------------- tokens
    def _count(self, text: str) -> int:
        """Estimate locally. Do NOT call /tokenize here.

        This runs from add_user() and add_assistant(), i.e. squarely on the
        critical path between the caller finishing a sentence and the first token
        coming back. llama-server answers /tokenize on the same worker that runs
        generation, so a round trip issued while a previous turn is still
        streaming does not take 5 ms -- it queues behind the whole generation.
        The count is only used to decide when to drop old turns from history, and
        4 chars/token is easily accurate enough for that.
        """
        return max(1, len(text) // 4)

    def _count_exact(self, text: str) -> int:
        """Real tokenizer. Only safe to call when no generation is in flight."""
        try:
            body = json.dumps({"content": text}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/tokenize", data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return len(json.load(resp).get("tokens", []))
        except Exception:
            return max(1, len(text) // 4)

    # ------------------------------------------------------------- generation
    def stream(
        self,
        stop_event: threading.Event | None = None,
        extra_system: str | None = None,
        stats: GenerationStats | None = None,
    ) -> Iterator[str]:
        """Yield response text deltas from the server's SSE stream."""
        stats = stats if stats is not None else GenerationStats()
        stats.prompt_tokens = self.context_tokens

        payload = {
            "messages": self.build_messages(extra_system),
            "temperature": float(self.cfg["temperature"]),
            "top_p": float(self.cfg["top_p"]),
            "max_tokens": int(self.cfg["max_tokens"]),
            "repeat_penalty": float(self.cfg["repeat_penalty"]),
            # Default repeat_last_n is 64 tokens -- shorter than one of Riya's
            # own replies, so it does nothing to stop her reissuing the same
            # question three turns running. Widening it to cover the recent
            # history is what makes the penalty actually bite across turns.
            "repeat_last_n": int(self.cfg.get("repeat_last_n", 512)),
            "presence_penalty": float(self.cfg.get("presence_penalty", 0.6)),
            "frequency_penalty": float(self.cfg.get("frequency_penalty", 0.4)),
            "stop": list(self.cfg["stop"]),
            "stream": True,
            "cache_prompt": True,   # reuse the KV prefix across turns -> lower TTFT
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )

        started = time.perf_counter()
        first_token_at: float | None = None
        count = 0

        # Tear down any previous turn's request BEFORE queuing for the lock.
        #
        # This was the single worst latency bug in the pipeline. The old code
        # held self._lock for the whole SSE loop and only checked stop_event
        # between received lines. On barge-in the orchestrator cancels the
        # asyncio task, but the LLM worker is a plain daemon thread sitting in a
        # blocking socket read -- it does not notice, and llama-server does not
        # emit anything to unblock it while it is still ingesting the prompt. So
        # the dead turn kept the lock, the new turn blocked on it, and
        # time-to-first-token came back as 10518 ms. Aborting the connection is
        # what actually frees both the lock and llama-server's slot.
        self.abort()
        if not self._lock.acquire(timeout=5.0):
            raise RuntimeError("previous generation would not release within 5 s")

        try:
            try:
                # Socket-level timeout is a backstop only; abort() is the fast path.
                resp = urllib.request.urlopen(req, timeout=30)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"llama-server request failed: {exc}") from exc

            with self._response_lock:
                self._active_response = resp

            # Watchdog: stop_event is set from the event loop, but this thread is
            # parked in a blocking read. Closing the socket from a third thread
            # is the only thing that reliably interrupts it, and it makes
            # llama-server drop the slot rather than finish generating audio
            # nobody will hear.
            watchdog_done = threading.Event()

            def _watch() -> None:
                while not watchdog_done.wait(0.02):
                    if stop_event is not None and stop_event.is_set():
                        try:
                            resp.close()
                        except Exception:
                            pass
                        return

            watcher = None
            if stop_event is not None:
                watcher = threading.Thread(target=_watch, name="llm-abort", daemon=True)
                watcher.start()

            try:
                for raw in resp:
                    if stop_event is not None and stop_event.is_set():
                        stats.interrupted = True
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    piece = (choices[0].get("delta") or {}).get("content")
                    if not piece:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        stats.ttft_ms = (first_token_at - started) * 1000.0
                    count += 1
                    # Update as we go, not after the loop. The consumer may stop
                    # iterating early (the sentence ceiling does exactly that),
                    # which closes this generator at the yield below -- the code
                    # after the loop never runs, and the turn was being reported
                    # as "0 gen tok" despite having spoken two chunks.
                    stats.tokens = count
                    stats.total_ms = (time.perf_counter() - started) * 1000.0
                    yield piece
            except (OSError, ValueError, http.client.HTTPException):
                # Expected when the watchdog closed the socket underneath us.
                stats.interrupted = True
            finally:
                watchdog_done.set()
                if watcher is not None:
                    watcher.join(timeout=0.2)
                try:
                    resp.close()
                except Exception:
                    pass
                with self._response_lock:
                    if self._active_response is resp:
                        self._active_response = None
        finally:
            self._lock.release()

        stats.tokens = count
        stats.total_ms = (time.perf_counter() - started) * 1000.0

    def abort(self) -> None:
        """Kill any in-flight generation. Safe from any thread, returns fast."""
        with self._response_lock:
            resp, self._active_response = self._active_response, None
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def warmup(self) -> float:
        """Prime llama-server's prefix cache with the REAL system prompt.

        This is the single biggest latency win on a slow CPU. The system prompt
        is ~800 tokens; ingesting it costs seconds on a small core count, and
        without this it would be paid on the caller's first question. Sending it
        here (with cache_prompt) means turn one only has to ingest the user's
        words, not the whole persona.
        """
        started = time.perf_counter()
        try:
            payload = {
                "messages": self.build_messages() + [{"role": "user", "content": "Hello."}],
                "max_tokens": 1,
                "temperature": 0.0,
                "cache_prompt": True,
            }
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp.read()
        except Exception:
            pass
        return (time.perf_counter() - started) * 1000.0


# ==========================================================================
#  Backend B: Ollama (already-running `ollama serve`, e.g. hr-bot:latest)
# ==========================================================================

class OllamaLLM(BaseLLM):
    """Talks to an already-running Ollama instance over loopback HTTP.

    Unlike LlamaServerLLM this backend never spawns or owns a process --
    Ollama is expected to already be serving (background service on Windows,
    or `ollama serve`), with the target model already pulled
    (`ollama list` should show it). All the barge-in / abort / watchdog
    machinery below is identical to the llama-server backend, because
    Ollama's native streaming response is also just a blocking line-by-line
    HTTP read that closing the socket from another thread reliably
    interrupts -- the same trick that fixed the 10s-TTFT bug in
    LlamaServerLLM.stream() applies here unchanged.
    """

    def __init__(self, model_name: str, cfg, system_prompt: str | None = None) -> None:
        super().__init__(cfg, system_prompt)
        system_prompt = self.system_prompt

        ollama_cfg = cfg.get("ollama", {}) or {}
        self.model_name = str(ollama_cfg.get("model", model_name))
        self.host = ollama_cfg.get("host", "127.0.0.1")
        self.port = int(ollama_cfg.get("port", 11434))
        self.base_url = f"http://{self.host}:{self.port}"
        self.keep_alive = ollama_cfg.get("keep_alive", "30m")
        # Optional thread pin, same spirit as plan_threads() in config.py --
        # leave unset to let Ollama use its own default.
        self.num_thread = ollama_cfg.get("num_thread")
        # Was set in config.yaml but never read here -- same bug shape as the
        # PiperTTS num_threads issue: a config key that looks like it does
        # something but doesn't. Added so a smaller context window actually
        # reduces Ollama's KV cache VRAM footprint instead of being a no-op.
        self.num_ctx = ollama_cfg.get("num_ctx")
        # Hybrid-reasoning models (MiniCPM5, Qwen3, DeepSeek-R1, ...) emit a
        # <think>...</think> block before the real answer unless told not to.
        # Ollama exposes this as a TOP-LEVEL "think" field on the request, not
        # something inside "options". Default False: this pipeline's
        # max_tokens budget (tens of tokens, not hundreds) cannot absorb a
        # reasoning trace, and a voice call cannot wait through one either --
        # a hung "Thinking..." block is multiple seconds of dead air on a
        # live phone call. Only flip this on if you also raise max_tokens
        # substantially and accept much higher time-to-first-token.
        self.think = bool(ollama_cfg.get("think", False))

        if not self._reachable():
            raise RuntimeError(
                f"No Ollama server reachable at {self.base_url}.\n"
                "Start it (it usually runs as a background service on "
                "Windows; otherwise run `ollama serve`), and make sure the "
                f"model is pulled: ollama list | findstr {self.model_name}"
            )

        # No /tokenize endpoint on Ollama -- same char/4 estimate BaseLLM
        # already uses as its fallback.
        self._system_tokens = self._count(system_prompt)
        self._active_response = None
        self._response_lock = threading.Lock()

        # Load the model into memory now, not on the caller's first question.
        self._preload()

    # ----------------------------------------------------------- connectivity
    def _reachable(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _preload(self) -> None:
        """Ask Ollama to load the model with our keep_alive before any real call."""
        try:
            payload = {
                "model": self.model_name,
                "keep_alive": self.keep_alive,
                "think": self.think,
            }
            # Match whatever context size the real stream() calls will use.
            # Ollama sizes the KV cache (and therefore how much VRAM the
            # loaded model needs) from num_ctx at LOAD time. Without this,
            # preload() loads the model at the default context, then the
            # first real turn's stream() call requests a different num_ctx
            # and forces a second, hidden reload right when it matters most
            # -- a plausible contributor to the turn-1 latency spikes seen
            # in testing (multiple seconds even after warmup reported done).
            if self.num_ctx:
                payload["options"] = {"num_ctx": int(self.num_ctx)}
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp.read()
        except Exception:
            pass  # best-effort; a real problem surfaces on the first stream() call

    def close(self) -> None:
        # We don't own the Ollama process -- never terminate it. It idles out
        # per keep_alive on its own.
        pass

    # ----------------------------------------------------------------- tokens
    def _count_exact(self, text: str) -> int:
        # No /tokenize endpoint to call exactly, unlike llama-server. This
        # only affects the initial system-prompt count used for the history
        # budget, not generation itself, so the estimate is fine here.
        return self._count(text)

    # ------------------------------------------------------------- generation
    def stream(
        self,
        stop_event: threading.Event | None = None,
        extra_system: str | None = None,
        stats: GenerationStats | None = None,
    ) -> Iterator[str]:
        """Yield response text deltas from Ollama's native streaming /api/chat.

        Ollama streams newline-delimited JSON objects, one per line -- no
        "data:" prefix, and end-of-stream is a final object with
        "done": true rather than a literal "[DONE]" line. Everything else
        (watchdog thread, abort-by-closing-the-socket, stats bookkeeping) is
        copied straight from LlamaServerLLM.stream() because the concurrency
        story is identical.
        """
        stats = stats if stats is not None else GenerationStats()
        stats.prompt_tokens = self.context_tokens

        options = {
            "temperature": float(self.cfg["temperature"]),
            "top_p": float(self.cfg["top_p"]),
            "num_predict": int(self.cfg["max_tokens"]),
            "repeat_penalty": float(self.cfg["repeat_penalty"]),
            "repeat_last_n": int(self.cfg.get("repeat_last_n", 512)),
            "presence_penalty": float(self.cfg.get("presence_penalty", 0.6)),
            "frequency_penalty": float(self.cfg.get("frequency_penalty", 0.4)),
            "stop": list(self.cfg["stop"]),
        }
        if self.num_thread:
            options["num_thread"] = int(self.num_thread)
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)

        payload = {
            "model": self.model_name,
            "messages": self.build_messages(extra_system),
            "stream": True,
            "keep_alive": self.keep_alive,
            "think": self.think,
            "options": options,
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        first_token_at: float | None = None
        count = 0

        # Tear down any previous turn's request BEFORE queuing for the lock --
        # same reasoning as LlamaServerLLM.stream(): a dead turn holding the
        # lock is what blows up barge-in latency.
        self.abort()
        if not self._lock.acquire(timeout=5.0):
            raise RuntimeError("previous generation would not release within 5 s")

        try:
            try:
                resp = urllib.request.urlopen(req, timeout=30)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Ollama request failed: {exc}") from exc

            with self._response_lock:
                self._active_response = resp

            watchdog_done = threading.Event()

            def _watch() -> None:
                while not watchdog_done.wait(0.02):
                    if stop_event is not None and stop_event.is_set():
                        try:
                            resp.close()
                        except Exception:
                            pass
                        return

            watcher = None
            if stop_event is not None:
                watcher = threading.Thread(target=_watch, name="llm-abort", daemon=True)
                watcher.start()

            try:
                for raw in resp:
                    if stop_event is not None and stop_event.is_set():
                        stats.interrupted = True
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        break
                    piece = (chunk.get("message") or {}).get("content")
                    if not piece:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        stats.ttft_ms = (first_token_at - started) * 1000.0
                    count += 1
                    stats.tokens = count
                    stats.total_ms = (time.perf_counter() - started) * 1000.0
                    yield piece
            except (OSError, ValueError, http.client.HTTPException):
                stats.interrupted = True
            finally:
                watchdog_done.set()
                if watcher is not None:
                    watcher.join(timeout=0.2)
                try:
                    resp.close()
                except Exception:
                    pass
                with self._response_lock:
                    if self._active_response is resp:
                        self._active_response = None
        finally:
            self._lock.release()

        stats.tokens = count
        stats.total_ms = (time.perf_counter() - started) * 1000.0

    def abort(self) -> None:
        with self._response_lock:
            resp, self._active_response = self._active_response, None
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def warmup(self) -> float:
        """Throwaway 1-token generation to force the model into RAM.

        Ollama does not expose explicit prompt-cache control the way
        llama-server's `cache_prompt` flag does -- it manages its own KV
        cache per loaded model and reuses matching prefixes automatically.
        So this is best-effort, not a guaranteed cold-start eliminator the
        way LlamaServerLLM.warmup() is.

        Uses stream: true, matching real stream() calls exactly. This used
        to send stream: false -- fixing the num_ctx mismatch (see git log)
        got warmup() itself down from 5000-7700ms to ~400ms, but turn 1 of
        every real call still spiked several seconds on its own right after
        a fast warmup. Streaming and non-streaming are different code paths
        server-side; a non-streaming ping never exercises whatever
        first-time setup the streaming path pays for once per process. This
        makes warmup pay that cost here instead of on the caller's first
        real turn.
        """
        started = time.perf_counter()
        try:
            options = {"num_predict": 1, "temperature": 0.0}
            if self.num_ctx:
                options["num_ctx"] = int(self.num_ctx)
            if self.num_thread:
                options["num_thread"] = int(self.num_thread)
            payload = {
                "model": self.model_name,
                "messages": self.build_messages() + [{"role": "user", "content": "Hello."}],
                "stream": True,
                "keep_alive": self.keep_alive,
                "think": self.think,
                "options": options,
            }
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=600)
            try:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        break
            finally:
                resp.close()
        except Exception:
            pass
        return (time.perf_counter() - started) * 1000.0


# ==========================================================================
#  Factory
# ==========================================================================

def build_llm(
    cfg,
    model_file: str | Path,
    system_prompt: str | None = None,
    call: dict | None = None,
) -> BaseLLM:
    """Construct the LLM backend described by cfg.

    `call` is the requisition brief (role, client, location, ...) that the
    outbound persona is built around; see prompts.DEFAULT_CALL.
    """
    if system_prompt is None:
        style = str(cfg.get("system_prompt", "compact")).lower()
        if style in ("none", "generic", "off", "no", "false"):
            # No persona, no call framing -- just the model, unprompted.
            system_prompt = ""
        else:
            # "full" adds ~350 tokens of objection-handling depth. "compact" keeps
            # the persona, the agenda, the one-question rule and the boundaries, and
            # is what warmup() pins in the prefix cache.
            system_prompt = build_system_prompt(call, compact=(style != "full"))

    backend = str(cfg.get("backend", "server")).lower()
    if backend in ("server", "llama-server", "http"):
        return LlamaServerLLM(model_file, cfg, system_prompt)
    if backend == "ollama":
        return OllamaLLM(model_file, cfg, system_prompt)
    raise ValueError(
        f"Unknown llm.backend {backend!r}. Supported: 'server' (llama-server "
        "subprocess) or 'ollama' (an already-running ollama serve instance)."
    )
