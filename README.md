# AstaScope

AstaScope is a small command-line chat agent built with [LangChain](https://python.langchain.com/) and Claude (`claude-opus-5-5`). It is kept to one file, `agent.py`, but uses many of LangChain's core features.

## Requirements

- Python 3.10 or newer (tested with 3.13)
- An Anthropic API key from the [Claude Console](https://console.anthropic.com/)

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Add your API key

Open `.env` and replace the placeholder with your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

If `.env` doesn't exist (for example, in a fresh clone), create it from the template first: `cp .env.example .env`.

`agent.py` loads `.env` automatically. The file is protected in two ways:

- **Git:** `.gitignore` lists `.env`, so it is never committed or pushed. Only `.env.example`, which holds a placeholder, is committed.
- **Claude Code:** `.claude/settings.json` denies Claude permission to read or edit `.env`, or print it with common shell commands.

### Add your documents

Put the documents you want AstaScope to search in the `docs` folder. It reads `.md`, `.txt` and `.rst` files, including files in subfolders, and ignores other file types. The files are loaded when AstaScope starts, so restart it after adding or changing files.

Each file is split into chunks of about 1,500 characters at paragraph breaks. When AstaScope searches, it gets back the 4 best-matching chunks, each labeled with the file it came from. AstaScope is told to name that file in its answer, so you can check where the information came from.

### Add papers (PDFs)

Put PDF files in the `papers` folder, start AstaScope and type `/convert`. It uses [marker-pdf](https://github.com/datalab-to/marker) to turn each PDF into Markdown, with the figures saved as image files, under `docs/papers/<file name>/`. The converted text is added to AstaScope's search right away, without a restart.

`/convert` keeps an index in `docs/papers/index.json` with one entry per converted paper: the PDF file name, the paper title (taken from the first heading in the converted text), the PDF's size in bytes, the date it was converted, the path to the Markdown file and the number of images. On later runs it skips any PDF that the index already lists with the same file size, so only new or changed PDFs are converted. To convert a paper again, delete its entry from the index.

The first `/convert` downloads marker's models (a few GB) and runs slowly. Conversion runs on your machine, so the PDFs are not sent to Claude.

### Combine the papers' references

After converting, type `/references` to build one list of every reference cited by the converted papers, with duplicates removed. It is saved to `docs/papers/references.md`, sorted by first author, and each entry lists the papers that cite it and its reference number in each. The file is added to AstaScope's search, so you can ask questions such as "which papers cite Swinburn?".

- References are found under a `References` (or `Bibliography`) heading, or after a bold **References** in a paragraph, as in Lancet papers. They are split by following the numbering 1, 2, 3, …, so papers must use numbered references; author–year styles are not supported.
- Two references count as the same if their titles match after ignoring case, punctuation and spacing. So differences in page ranges, journal abbreviations or author lists don't stop a match, but a different spelling in the title does.
- When the same reference appears in several papers, the longest version is kept (usually the one with the full author list).
- An **Authors** section at the top of the file gives the number of unique authors and lists the 20 who appear on the most references. Each person counts once per reference. Authors are read from the start of each reference in the "Surname Initials" style (for example, `Flegal KM`). Organisation authors, such as WHO, aren't counted, and neither are authors hidden by "et al". "Wu Y-K" and "Wu YK" count as one person. Names with the same surname where one set of initials is shorter, such as "Puhl R" and "Puhl RM", are counted separately but listed as possibly the same person.
- `/references` reads all converted papers each time and replaces the file. It doesn't send anything to Claude.

### Summarize the papers

Type `/summarize` to print a one-line summary of each converted paper. Unlike the other commands, this one sends each paper's text to Claude (up to the first 60,000 characters, without the reference list), so you're asked to confirm each request as usual. Summaries are saved in `docs/papers/index.json`, so later runs print them without asking Claude again. Reconverting a paper clears its summary.

## Run

```bash
python agent.py
```

At startup AstaScope reports what it loaded from `docs`, for example `Loaded 12 documents (48 chunks) from docs/.` If it says 0, check that your files are in `docs` and end in `.md`, `.txt` or `.rst`.

Enter your name, then chat. Type `/exit` (or press `Ctrl-C`) to quit.

While it runs, AstaScope prints a line for each model call and each tool call. Every reply lists the tools it used.

### Confirming before each request to Claude

Before anything is sent to Claude, AstaScope shows how many messages it will send and a preview of the last one, then asks:

```
Send to Claude? [y]es / [n]o / [a]lways:
```

- **y** sends this request.
- **n** cancels the rest of this answer. Your question stays in the conversation's memory.
- **a** sends this request and stops asking for the rest of the session.

One question can take several requests (one per tool step), so you may be asked more than once per question. The check is a LangChain callback on the model, so it covers every request, including the `explain_simply` chain and conversation summaries.

### Commands

Commands run locally and are never sent to Claude, except `/summarize`.

| Command | What it does |
|---|---|
| `/help` | List the commands |
| `/notes` | List the notes in the knowledge base |
| `/docs` | List the files loaded from `docs` and how many chunks each has |
| `/convert` | Convert new PDFs in `papers` to Markdown with images, and update the index (see [Add papers](#add-papers-pdfs)) |
| `/references` | Combine the reference lists of all converted papers, without duplicates (see [Combine the papers' references](#combine-the-papers-references)) |
| `/summarize` | Print a one-line summary of each converted paper, asking Claude for new ones (see [Summarize the papers](#summarize-the-papers)) |
| `/history` | Show the messages in this conversation |
| `/new` | Start a fresh conversation |
| `/exit`, `/quit` | Quit |

### Things to try

| Prompt | What it shows |
|---|---|
| `what's 17% of 2340?` | The calculator tool |
| `what time is it?` | The current-time tool |
| `what do the notes say about middleware?` | Searching documents with a retriever |
| A question about one of your files in `docs` | Searching your own documents; the answer names the file it used |
| `explain recursion simply` | A tool that runs its own prompt chain |
| `save a note that I like Python` | Asking for your approval (`y/N`) before a tool runs |
| `what did I just ask you?` | Memory of earlier turns |

## LangChain features used

| Feature | Where in `agent.py` |
|---|---|
| Chat model | `ChatAnthropic` with effort set to `medium` |
| Callbacks | `ConfirmBeforeSend` asks before each request to Claude (`on_chat_model_start`) |
| Documents and retriever | `load_docs` splits the files in `docs` into `Document` chunks; a custom `BM25Retriever` (a `BaseRetriever` subclass) searches them along with the notes (no embeddings needed) |
| Chains (LCEL) | `prompt \| model \| StrOutputParser()`, used inside the `explain_simply` tool |
| Tools | Five `@tool` functions |
| Runtime context | The `Context` dataclass; tools read it through `ToolRuntime` |
| Middleware | `@dynamic_prompt`, `@before_model`, `HumanInTheLoopMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware` |
| Structured output | Each reply is returned as an `Answer` object (`response_format=Answer`) |
| Memory | The `InMemorySaver` checkpointer, keyed by `thread_id` |
| Streaming and pause/resume | `agent.stream(..., stream_mode="updates")`, then `Command(resume=...)` to continue after approval |

## Notes

- Memory and saved notes are kept in memory only, so they are lost when the program exits. AstaScope never changes your files in `docs`; `/convert` only writes inside `docs/papers`.
- Search matches keywords, not meaning, so a question only finds passages that use some of the same words.
- The model has Anthropic's server-side fallback turned on (`fallbacks: "default"`): if a safety check declines a request, the API retries it on another model. To turn it off, remove the `betas` and `model_kwargs` lines from the `ChatAnthropic(...)` call.
