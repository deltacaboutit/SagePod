# SagePod

SagePod turns a Markdown file of investment research notes into a 15–20 minute
audio briefing. The briefing sounds like an analyst presenting the idea to an
investment committee, with the committee chair asking questions. You use it
from your phone's web browser, and you get an MP3 file to download.

Everything runs on your own SageMaker machine:

| Job | Free model used | Licence |
|---|---|---|
| Writing the script | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Apache 2.0 |
| Speaking the script | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache 2.0 |
| Web page | [Streamlit](https://streamlit.io) | Apache 2.0 |

There are no API keys and no per-use fees. The only cost is the SageMaker
machine itself, which AWS bills by the hour while it is running.

## How it works

1. You upload a `.md` file (or paste the text) on the web page.
2. The language model (an AI that writes text) writes the conversation in eight
   sections: the opening and the recommendation, the business, the thesis,
   financials and valuation, risks, catalysts and scenarios, committee
   questions, and the wrap-up. Writing in sections helps a smaller model
   produce a long script that stays on topic. It is told to use only facts
   from your notes.
3. The text-to-speech model (an AI that reads text aloud) records each line
   with two different voices.
4. The audio is joined together and saved as an MP3. You can play it on the
   page or download it.

The job runs in the background. You can lock your phone or close the tab and
come back later. You can also open the script, fix anything you don't like,
and press **Re-record audio** to make a new MP3 without rewriting it.

## 1. Pick a SageMaker machine

You need a machine with a GPU (a graphics chip that runs AI models quickly).
The model needs about 16 GB of GPU memory.

- **Recommended:** `ml.g5.xlarge` or `ml.g6.xlarge` (24 GB GPU each).
- Use an image that already includes PyTorch, such as the default
  *SageMaker Distribution* image in SageMaker Studio JupyterLab.
- Give the machine at least **50 GB of storage**, because the models take about 16 GB.

A **SageMaker Studio Code Editor space** (the VS Code version), a **Studio
JupyterLab space**, or a **SageMaker Notebook Instance** all work. In Code
Editor, open a terminal with **Terminal → New Terminal** to run the commands
below. When you create the Code Editor space, pick a GPU instance type such
as `ml.g5.xlarge`. If you don't, it runs on a small machine without a GPU.

## 2. Set it up (once)

Open a terminal (in Code Editor: **Terminal → New Terminal**) and run:

```bash
git clone <this repository's URL> SagePod
cd SagePod
./setup.sh
```

This installs the Python packages, downloads the models (about 15 GB, which
takes a few minutes), and downloads `cloudflared`, a small free tool that
creates a web address you can open from your phone.

## 3. Start the app and open it on your phone

### Option A: a private link (easiest)

```bash
SAGEPOD_PASSWORD='choose-a-good-password' ./run.sh
```

After a few seconds the terminal prints a line like:

```
 Open this on your phone:  https://some-random-words.trycloudflare.com
```

Open that link on your phone and enter your password. The link is reachable
from the internet, which is why the password is required. The link changes
each time you restart the app.

### Option B: through your AWS login

This keeps the app behind your AWS sign-in, but you must sign in to the AWS
console on your phone first.

- **Studio Code Editor (VS Code):** run `./run.sh codeeditor`. Take the
  address of your Code Editor tab, which looks like
  `https://abc123.studio.us-east-1.sagemaker.aws/codeeditor/default/?folder=...`,
  and replace everything after `.aws` with `/codeeditor/default/proxy/8501/`.
  If that page doesn't load, use Option A instead.
- **Studio JupyterLab:** run `./run.sh studio`. Take the address of your
  JupyterLab page, which looks like
  `https://abc123.studio.us-east-1.sagemaker.aws/jupyterlab/default/lab`, and
  replace the `/lab` ending with `/proxy/absolute/8501/`.
- **Notebook Instance:** run `./run.sh notebook` and open
  `https://<notebook-name>.notebook.<region>.sagemaker.aws/proxy/absolute/8501/`.

The app keeps running while the terminal stays open in Code Editor or
JupyterLab. You can
close your browser and the app keeps going.

## Settings

On the web page, under **Settings**:

- **Target length:** 10–25 minutes (default 17). The language model controls
  the exact length, so expect results within a few minutes of the target.
- **Voices:** choose from nine voices for the analyst and the chair.
- **Speaking speed:** 0.8 (slower) to 1.3 (faster).

Advanced settings use environment variables (values you put before the command):

| Variable | What it does | Default |
|---|---|---|
| `SAGEPOD_PASSWORD` | Password for the web page | none |
| `SAGEPOD_LLM` | Which Hugging Face language model to use | `Qwen/Qwen2.5-7B-Instruct` |
| `SAGEPOD_JOBS_DIR` | Where podcasts are saved | `./podcasts` |

For example, a machine with a bigger GPU (such as `ml.g5.12xlarge`) can run a
stronger writer:
`SAGEPOD_LLM='Qwen/Qwen2.5-32B-Instruct' SAGEPOD_PASSWORD='...' ./run.sh`.

## How long it takes

On an `ml.g5.xlarge`, a 17-minute podcast usually takes about 5–15 minutes
to make. The very first run is slower if the models still need to download.

## Tips

- **Stop the machine when you're done.** SageMaker bills by the hour while
  it's running. Your podcasts and the downloaded models stay saved.
- Notes with clear headings (recommendation, thesis, valuation, risks,
  catalysts) give the best results, because the script follows that order.
- Check important numbers. The model is told to use only your notes, but AI
  models can still make mistakes. The script is saved next to each podcast so
  you can check and edit it.
- `examples/sample_note.md` is a made-up note you can use to try the app.

## Files

| File | Purpose |
|---|---|
| `app.py` | The app: web page, script writing, voice recording |
| `setup.sh` | One-time install and model download |
| `run.sh` | Starts the app |
| `requirements.txt` | Python packages |
| `podcasts/` | Your finished podcasts (one folder each, with notes, script and MP3) |
