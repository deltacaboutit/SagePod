"""
SagePod - turn investment research notes (Markdown) into a podcast-style
investment committee briefing, using only free, locally-run models.

  * Script writing:  an open-weights LLM run with Hugging Face `transformers`
                     (default: Qwen2.5-7B-Instruct, Apache-2.0 licence).
  * Voices:          Kokoro-82M text-to-speech (Apache-2.0 licence).
  * Web interface:   Streamlit, which works well in a phone browser.

Nothing is sent to a paid API. The models are downloaded once from Hugging
Face and then run on the SageMaker instance's GPU.

Run it with:   ./run.sh            (see README.md for phone access)
"""

from __future__ import annotations

import datetime as dt
import gc
import hmac
import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import streamlit as st

# --------------------------------------------------------------------------
# Settings (can be overridden with environment variables)
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("SAGEPOD_JOBS_DIR", APP_DIR / "podcasts"))
LLM_MODEL = os.environ.get("SAGEPOD_LLM", "Qwen/Qwen2.5-7B-Instruct")
APP_PASSWORD = os.environ.get("SAGEPOD_PASSWORD", "")

SAMPLE_RATE = 24_000          # Kokoro outputs 24 kHz audio
WORDS_PER_MINUTE = 140        # measured Kokoro pace at speed 1.0, including pauses
MAX_NOTE_CHARS = 60_000       # ~15k tokens; keeps prompts inside the context window

# The two people in the conversation. Labels are what the LLM writes in the
# script; the names are spoken in the dialogue.
SPEAKERS = {
    "ANALYST": {"name": "Sarah", "role": "the lead analyst presenting the idea"},
    "CHAIR": {"name": "David", "role": "the investment committee chair"},
}

# Kokoro voice choices (American "a*" and British "b*" voices).
VOICES = {
    "Heart (US, female)": "af_heart",
    "Bella (US, female)": "af_bella",
    "Nicole (US, female)": "af_nicole",
    "Emma (UK, female)": "bf_emma",
    "Michael (US, male)": "am_michael",
    "Fenrir (US, male)": "am_fenrir",
    "Puck (US, male)": "am_puck",
    "George (UK, male)": "bm_george",
    "Fable (UK, male)": "bm_fable",
}

# The podcast is written section by section. A small local model writes much
# better in short focused pieces than in one 3,000-word go. The weights say
# what share of the total running time each section gets.
SEGMENTS = [
    ("Opening and the ask", 0.08,
     "Open the meeting. The chair welcomes everyone and introduces the idea. "
     "The analyst states the company, the recommendation (for example buy, "
     "sell, or hold), the proposed position or price target if given, and a "
     "one-minute summary of why. Do not go deep yet."),
    ("Business overview", 0.13,
     "Explain what the company does, how it makes money, its market position, "
     "competitors, and anything about management or industry structure in "
     "the notes. The chair asks clarifying questions."),
    ("Investment thesis", 0.20,
     "Walk through each pillar of the investment thesis one by one, with the "
     "supporting evidence from the notes. The chair pushes on why the market "
     "has this wrong and what the analyst sees that others do not."),
    ("Financials and valuation", 0.17,
     "Cover the key financial figures, trends, estimates, and the valuation "
     "approach, multiples, and price target from the notes. Explain what the "
     "numbers mean in plain words. Round numbers sensibly when speaking."),
    ("Risks and mitigants", 0.15,
     "Go through the main risks to the thesis. For each one, the chair "
     "challenges and the analyst explains how likely it is, how bad it would "
     "be, and any mitigants. Include bear-case views if the notes have them."),
    ("Catalysts and scenarios", 0.10,
     "Cover upcoming catalysts and timing, and the bull, base, and bear "
     "scenarios with their implied outcomes if the notes give them."),
    ("Committee Q&A", 0.10,
     "The chair asks the toughest remaining questions an investment committee "
     "would ask, such as position sizing, what would make the analyst change "
     "their mind, and how this fits the portfolio. The analyst answers candidly."),
    ("Wrap-up and decision", 0.07,
     "The chair summarises the discussion. The analyst restates the "
     "recommendation and the key things to monitor. Close the meeting "
     "naturally, with a short sign-off."),
]


# --------------------------------------------------------------------------
# Job bookkeeping: each podcast is a folder with a status.json file, so the
# list of podcasts survives page reloads and phone screen locks.
# --------------------------------------------------------------------------
def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def read_status(job_id: str) -> dict:
    try:
        return json.loads((_job_dir(job_id) / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_status(job_id: str, **updates) -> None:
    status = read_status(job_id)
    status.update(updates)
    status["updated"] = time.time()
    path = _job_dir(job_id) / "status.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(path)  # atomic, so the UI never reads half a file


def list_jobs() -> list[dict]:
    if not JOBS_DIR.exists():
        return []
    jobs = []
    for d in JOBS_DIR.iterdir():
        if d.is_dir() and (d / "status.json").exists():
            s = read_status(d.name)
            if s:
                jobs.append(s)
    return sorted(jobs, key=lambda s: s.get("created", 0), reverse=True)


# --------------------------------------------------------------------------
# Model loading (each model is loaded once, then kept in GPU memory)
# --------------------------------------------------------------------------
class Models:
    def __init__(self):
        self._llm = None
        self._tts = None
        self._lock = threading.Lock()

    def llm(self):
        with self._lock:
            if self._llm is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(LLM_MODEL)
                model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                )
                model.eval()
                self._llm = (tok, model)
            return self._llm

    def tts(self):
        with self._lock:
            if self._tts is None:
                from kokoro import KPipeline

                device = "cuda" if _cuda() else "cpu"
                self._tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
            return self._tts


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def chat(models: Models, system: str, user: str, max_new_tokens: int) -> str:
    """Send one chat turn to the local LLM and return its reply."""
    import torch

    tok, model = models.llm()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # Some newer models "think out loud" first; drop that part.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# --------------------------------------------------------------------------
# Step 1: write the script
# --------------------------------------------------------------------------
def build_system_prompt() -> str:
    a, c = SPEAKERS["ANALYST"], SPEAKERS["CHAIR"]
    return f"""You write scripts for an audio briefing styled as an investment committee meeting.
There are exactly two speakers:
- ANALYST: {a['name']}, {a['role']}. Confident, clear, data-driven, honest about uncertainty.
- CHAIR: {c['name']}, {c['role']}. Sharp, sceptical, asks probing questions, keeps the meeting moving.

Rules:
1. Write ONLY dialogue lines. Every line starts with "ANALYST:" or "CHAIR:" followed by what they say.
2. No stage directions, sound effects, headings, bullet points, tables, emojis, or markdown.
3. This will be read aloud by a text-to-speech voice, so write for the ear: natural spoken sentences,
   contractions, short paragraphs. Say numbers the way a person would say them out loud, for example
   "about four point two billion dollars", "twelve and a half percent", "fiscal twenty twenty-five",
   "eighteen times earnings". Spell out abbreviations the first time, e.g. "EBITDA, that's earnings
   before interest, taxes, depreciation and amortisation".
4. Use ONLY facts, figures and opinions found in the research notes. Never invent numbers, names or
   events. If the notes do not cover something, the speakers can briefly say so and move on.
5. Make it a real conversation: the chair interrupts with questions, the analyst answers and expands.
   Vary turn length; most turns are two to five sentences, some longer explanations are fine."""


def build_segment_prompt(notes: str, title: str, brief: str, words: int,
                         index: int, total: int, previous_tail: str) -> str:
    position = (
        "This is the FIRST section; start the meeting from the beginning."
        if index == 0
        else "This section continues directly from the previous one. Do not greet anyone or restart the meeting."
    )
    if index == total - 1:
        position += " This is the LAST section; end the meeting naturally."
    else:
        position += " Do not wrap up or say goodbye; the meeting continues after this section."

    prev = (
        f"\nHere are the last lines spoken so far, for continuity (do not repeat them):\n{previous_tail}\n"
        if previous_tail
        else ""
    )
    return f"""RESEARCH NOTES (the only source of facts):
<<<
{notes}
>>>

You are writing section {index + 1} of {total} of the meeting: "{title}".
What to cover in this section: {brief}
{position}
{prev}
Length: about {words} words of dialogue. Do not cover topics that belong to other sections in depth.

Write the dialogue for this section now."""


# Matches "ANALYST: ...", "**CHAIR:** ...", "Sarah (Analyst): ..." and similar.
_LABELS = {k: k for k in SPEAKERS} | {v["name"].upper(): k for k, v in SPEAKERS.items()}
LINE_RE = re.compile(rf"^\W*({'|'.join(_LABELS)})\W*(?:\([^)]*\))?\W*:\s*(.+)$", re.I)


def parse_script(text: str) -> list[tuple[str, str]]:
    """Turn raw LLM output into a list of (speaker, line) pairs."""
    turns: list[tuple[str, str]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = LINE_RE.match(raw)
        if m:
            turns.append((_LABELS[m.group(1).upper()], m.group(2).strip()))
        elif turns and not raw.startswith(("#", "---", "|")):
            # A wrapped continuation of the previous speaker's line.
            sp, line = turns[-1]
            turns[-1] = (sp, f"{line} {raw}")
    cleaned = []
    for sp, line in turns:
        line = re.sub(r"\[[^\]]*\]|\*\([^)]*\)\*", "", line)   # [laughs], *(pauses)*
        line = re.sub(r"[*_`#]+", "", line).strip()             # stray markdown
        line = re.sub(r"\s+([.,!?;:])", r"\1", line)
        if line:
            cleaned.append((sp, line))
    return cleaned


def count_words(turns: list[tuple[str, str]]) -> int:
    return sum(len(line.split()) for _, line in turns)


def write_script(models: Models, notes: str, minutes: float, progress) -> list[tuple[str, str]]:
    target_words = int(minutes * WORDS_PER_MINUTE)
    system = build_system_prompt()
    script: list[tuple[str, str]] = []

    for i, (title, weight, brief) in enumerate(SEGMENTS):
        words = max(150, int(target_words * weight))
        progress(f"Writing section {i + 1}/{len(SEGMENTS)}: {title}", 0.05 + 0.55 * i / len(SEGMENTS))
        tail = "\n".join(f"{s}: {l}" for s, l in script[-4:])
        prompt = build_segment_prompt(notes, title, brief, words, i, len(SEGMENTS), tail)
        # ~1.5 tokens per word, plus headroom.
        section = parse_script(chat(models, system, prompt, max_new_tokens=int(words * 2) + 300))

        # Small models often write too little. Ask once for more if needed.
        if count_words(section) < 0.6 * words and section:
            more_prompt = (
                prompt
                + "\n\nYou already wrote this much of the section:\n"
                + "\n".join(f"{s}: {l}" for s, l in section)
                + f"\n\nContinue the same section from where it stopped, adding about "
                f"{words - count_words(section)} more words. Do not repeat earlier lines."
            )
            section += parse_script(chat(models, system, more_prompt, max_new_tokens=int(words * 2) + 300))

        if not section:
            raise RuntimeError(f"The language model returned no usable dialogue for section '{title}'.")
        script.extend(section)
    return script


# --------------------------------------------------------------------------
# Step 2: make the text easier to pronounce
# --------------------------------------------------------------------------
_SCALE = {"k": "thousand", "m": "million", "mm": "million", "b": "billion", "bn": "billion", "t": "trillion"}
_ABBREV = {
    r"\bYoY\b": "year over year", r"\bQoQ\b": "quarter over quarter",
    r"(\d)\s?bps?\b": r"\1 basis points", r"\bvs\.?\s": "versus ",
    r"\bP/E\b": "P E", r"\bEV/EBITDA\b": "E V to EBITDA", r"\bEV\b": "E V",
    r"\bFCF\b": "free cash flow", r"\bCAGR\b": "compound annual growth rate",
    r"\bTAM\b": "total addressable market", r"\bM&A\b": "M and A", r"\bR&D\b": "R and D",
    r"\bS&P\b": "S and P", r"\be\.g\.": "for example", r"\bi\.e\.": "that is",
    r"\betc\.": "and so on", r"\bmgmt\b": "management", r"\bEPS\b": "E P S",
}


def speakable(text: str) -> str:
    """Rewrite symbols and finance shorthand that TTS voices tend to mangle."""
    t = text.replace("&", " and ").replace("~", "about ").replace("—", ", ").replace("–", " to ")
    # $4.2B -> 4.2 billion dollars ; $350M -> 350 million dollars
    t = re.sub(r"\$\s?(\d[\d,]*\.?\d*)\s?(bn|mm|[kmbt])\b",
               lambda m: f"{m.group(1)} {_SCALE[m.group(2).lower()]} dollars", t, flags=re.I)
    t = re.sub(r"\$\s?(\d[\d,]*\.?\d*)", r"\1 dollars", t)
    t = re.sub(r"(\d[\d,]*\.?\d*)\s?(bn|mm)\b", lambda m: f"{m.group(1)} {_SCALE[m.group(2).lower()]}", t, flags=re.I)
    t = re.sub(r"(\d)\s?%", r"\1 percent", t)
    t = re.sub(r"(\d+(?:\.\d+)?)x\b", r"\1 times", t)                    # 18x -> 18 times
    t = re.sub(r"\bFY\s?'?(\d{2})\b", r"fiscal 20\1", t)                  # FY25 -> fiscal 2025
    t = re.sub(r"\b([1-4])Q\s?'?(\d{2})\b", r"Q\1 20\2", t)               # 3Q24 -> Q3 2024
    t = re.sub(r"\bQ([1-4])\b", r"Q \1", t)
    for pat, rep in _ABBREV.items():
        t = re.sub(pat, rep, t)
    return re.sub(r"\s{2,}", " ", t).strip()


# --------------------------------------------------------------------------
# Step 3: turn the script into audio
# --------------------------------------------------------------------------
def synthesize(models: Models, script, voices: dict, speed: float, progress) -> np.ndarray:
    pipe = models.tts()
    pause = np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)
    pieces = [np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)]
    for i, (speaker, line) in enumerate(script):
        if i % 5 == 0:
            progress(f"Recording voices: line {i + 1} of {len(script)}", 0.62 + 0.35 * i / len(script))
        for result in pipe(speakable(line), voice=voices[speaker], speed=speed):
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            pieces.append(np.asarray(audio, dtype=np.float32))
        pieces.append(pause)
    pieces.append(np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32))
    audio = np.concatenate(pieces)
    peak = float(np.max(np.abs(audio))) or 1.0
    return audio * (0.95 / peak)


def save_mp3(audio: np.ndarray, path: Path, bitrate: int = 96) -> None:
    import lameenc

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(2)
    path.write_bytes(enc.encode(pcm.tobytes()) + enc.flush())


def script_to_text(script) -> str:
    names = {k: v["name"] for k, v in SPEAKERS.items()}
    return "\n\n".join(f"{names[s]} ({s.title()}): {l}" for s, l in script)


def text_to_script(text: str) -> list[tuple[str, str]]:
    """Read back a script the user edited in the app."""
    return parse_script(text)


# --------------------------------------------------------------------------
# Background worker: one podcast at a time on the GPU, so you can close the
# browser tab or lock your phone while it runs.
# --------------------------------------------------------------------------
class Worker:
    def __init__(self):
        self.models = Models()
        self.queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, job_id: str) -> None:
        self.queue.put(job_id)

    def _loop(self):
        while True:
            job_id = self.queue.get()
            try:
                self._run(job_id)
            except Exception as e:  # report any failure in the UI
                write_status(job_id, state="failed", message=f"{type(e).__name__}: {e}",
                             detail=traceback.format_exc())
            finally:
                gc.collect()
                if _cuda():
                    import torch

                    torch.cuda.empty_cache()

    def _run(self, job_id: str):
        d = _job_dir(job_id)
        s = read_status(job_id)
        voices = {"ANALYST": s["analyst_voice"], "CHAIR": s["chair_voice"]}

        def progress(msg, frac):
            write_status(job_id, state="running", message=msg, progress=round(frac, 3))

        if s.get("mode") == "revoice":
            script = text_to_script((d / "script.txt").read_text())
        else:
            progress("Loading the language model (first run downloads ~15 GB)", 0.02)
            notes = (d / "notes.md").read_text()
            script = write_script(self.models, notes, s["minutes"], progress)
            (d / "script.txt").write_text(script_to_text(script))

        progress("Loading the voice model", 0.61)
        audio = synthesize(self.models, script, voices, s["speed"], progress)
        progress("Saving the audio file", 0.98)
        save_mp3(audio, d / "podcast.mp3")
        write_status(job_id, state="done", progress=1.0,
                     message="Ready", minutes_actual=round(len(audio) / SAMPLE_RATE / 60, 1),
                     words=count_words(script))


@st.cache_resource
def get_worker() -> Worker:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    # Jobs that were mid-way when the app last stopped cannot resume.
    for s in list_jobs():
        if s.get("state") in ("queued", "running"):
            write_status(s["id"], state="failed", message="Interrupted because the app restarted. Try again.")
    return Worker()


def new_job(title: str, notes: str | None, script_text: str | None, settings: dict) -> str:
    job_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = _job_dir(job_id)
    d.mkdir(parents=True)
    if notes is not None:
        (d / "notes.md").write_text(notes)
    if script_text is not None:
        (d / "script.txt").write_text(script_text)
    write_status(job_id, id=job_id, title=title, created=time.time(), state="queued",
                 progress=0.0, message="Waiting in line", **settings)
    get_worker().submit(job_id)
    return job_id


# --------------------------------------------------------------------------
# Web interface
# --------------------------------------------------------------------------
def check_password() -> bool:
    if not APP_PASSWORD or st.session_state.get("authed"):
        return True
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw, APP_PASSWORD):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password.")
    return False


def settings_form() -> dict:
    with st.expander("Settings", expanded=False):
        minutes = st.slider("Target length (minutes)", 10, 25, 17)
        names = list(VOICES)
        a = st.selectbox(f"{SPEAKERS['ANALYST']['name']} (analyst) voice", names, index=0)
        c = st.selectbox(f"{SPEAKERS['CHAIR']['name']} (committee chair) voice", names,
                         index=names.index("Michael (US, male)"))
        speed = st.slider("Speaking speed", 0.8, 1.3, 1.0, 0.05)
    return {"minutes": minutes, "analyst_voice": VOICES[a], "chair_voice": VOICES[c], "speed": speed}


def render_job(s: dict) -> None:
    d = _job_dir(s["id"])
    when = dt.datetime.fromtimestamp(s.get("created", 0)).strftime("%b %d, %H:%M")
    state = s.get("state")
    icon = {"done": "✅", "failed": "❌", "running": "⏳", "queued": "🕒"}.get(state, "•")
    with st.container(border=True):
        st.markdown(f"**{icon} {s.get('title', s['id'])}**  \n{when}")
        if state in ("queued", "running"):
            st.progress(min(1.0, float(s.get("progress", 0))), text=s.get("message", ""))
        elif state == "failed":
            st.error(s.get("message", "Something went wrong."))
            if s.get("detail"):
                with st.expander("Technical details"):
                    st.code(s["detail"])
        elif state == "done":
            mp3 = d / "podcast.mp3"
            st.caption(f"{s.get('minutes_actual', '?')} minutes · {s.get('words', '?')} words")
            if mp3.exists():
                st.audio(str(mp3), format="audio/mpeg")
                st.download_button("Download MP3", mp3.read_bytes(), file_name=f"{_slug(s['title'])}.mp3",
                                   mime="audio/mpeg", key=f"dl-{s['id']}", use_container_width=True)

        script_file = d / "script.txt"
        if state in ("done", "failed") and script_file.exists():
            with st.expander("Script: read, edit and re-record"):
                edited = st.text_area("Script", script_file.read_text(), height=300,
                                      key=f"ed-{s['id']}", label_visibility="collapsed")
                col1, col2 = st.columns(2)
                col1.download_button("Download script", edited, file_name=f"{_slug(s['title'])}.txt",
                                     key=f"dls-{s['id']}", use_container_width=True)
                if col2.button("Re-record audio", key=f"rv-{s['id']}", use_container_width=True):
                    if not text_to_script(edited):
                        st.error("No speaker lines found. Keep the 'Name (Role):' labels at the start of lines.")
                    else:
                        keys = ("minutes", "analyst_voice", "chair_voice", "speed")
                        new_job(f"{s['title']} (re-recorded)", None, edited,
                                {**{k: s[k] for k in keys}, "mode": "revoice"})
                        st.rerun()
        if state in ("done", "failed"):
            if st.button("Delete", key=f"del-{s['id']}"):
                shutil.rmtree(d, ignore_errors=True)
                st.rerun()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:60] or "podcast"


@st.fragment(run_every=5)
def active_jobs() -> None:
    """Jobs still in progress. Refreshes itself every few seconds."""
    jobs = [s for s in list_jobs() if s.get("state") in ("queued", "running")]
    ids = {s["id"] for s in jobs}
    finished = st.session_state.get("active_ids", set()) - ids
    st.session_state.active_ids = ids
    if finished:
        st.rerun()  # redraw the whole page so the finished podcast shows its player
    for s in jobs:
        render_job(s)


def finished_jobs() -> None:
    """Finished jobs are drawn outside the auto-refresh, so playback is never cut off."""
    jobs = [s for s in list_jobs() if s.get("state") not in ("queued", "running")]
    for s in jobs:
        render_job(s)


def main() -> None:
    st.set_page_config(page_title="SagePod", page_icon="🎙️", layout="centered")
    st.title("🎙️ SagePod")
    st.caption("Research notes → investment committee podcast")
    if not check_password():
        return
    get_worker()

    # No file-type filter here: some phone browsers grey out .md files when one is set.
    upload = st.file_uploader("Upload research notes (Markdown file)")
    with st.expander("...or paste the notes instead"):
        pasted = st.text_area("Notes", height=200, label_visibility="collapsed")
    settings = settings_form()

    notes, default_title = None, ""
    if upload is not None:
        notes = upload.getvalue().decode("utf-8", errors="replace")
        default_title = Path(upload.name).stem.replace("_", " ").replace("-", " ")
    elif pasted.strip():
        notes, default_title = pasted, "Pasted notes"

    if notes is not None:
        if len(notes) > MAX_NOTE_CHARS:
            st.warning(f"These notes are long, so only the first {MAX_NOTE_CHARS:,} characters will be used.")
            notes = notes[:MAX_NOTE_CHARS]
        title = st.text_input("Podcast title", default_title)
        if st.button("Make podcast", type="primary", use_container_width=True):
            new_job(title or default_title, notes, None, {**settings, "mode": "full"})
            st.success("Started. This usually takes 5 to 15 minutes. You can lock your phone and check back.")

    st.subheader("Your podcasts")
    active_jobs()
    finished_jobs()
    if not list_jobs():
        st.caption("No podcasts yet.")


if __name__ == "__main__":
    main()
