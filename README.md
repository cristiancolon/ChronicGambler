## BluffBot: Iterative Poker Bots for Bounty Hold'em

This repo captures an end-to-end journey building increasingly capable heads-up poker bots for a Bounty Hold'em variant, culminating in a neural-network-driven agent. The codebase includes multiple bot lines (weeks/branches), an engine to pit them against each other, and lightweight training utilities.

If you just want to run matches, see Run a match. If you want the full story of how each bot evolved, start with the Narrative of iterations.

### What is Bounty Hold'em?
It's Texas Hold'em heads-up with an extra bounty rule: each player is secretly assigned a rank (2-9,T,J,Q,K,A). If you win a pot and your bounty rank appears in your hole or community cards, you win 50% more plus a small constant. This affects optimal strategy, risk/reward, and bluff/call thresholds.

## Narrative of iterations

The project evolved through distinct bot generations. The goal of this section is storytelling: how each bot became more sophisticated than the previous one. Only the final bot is documented in the traditional, operational sense later.

- w1_bot (baseline)
  - First functional agent built on the provided `skeleton` with very simple heuristics and a tiny amount of Monte Carlo hand strength estimation.
  - Purpose: get the full loop working (engine <-> bot), legal actions, minimal preflop filtering, bounty awareness basics.

- w2 line (richer heuristics and profiling)
  - w2_bot: Adds opponent profiling signals (basic Aggression Factor), pot-odds/EV computations, simple dynamic bet sizing, and cleaner pre/post-flop separation.
  - w2_bot_cristi: Improves premium hand detection, bounty-hit checks, and Monte Carlo evaluation; adds safer raise validation and slightly more nuanced postflop lines.
  - w2_bot_fabian: Refines preflop overfold reduction, clearer 3-bet/4-bet paths, semi-bluffs, and opponent AF-based adaptation.
  - w2_jputa: A very exploratory branch with extensive bluff frameworks, multi-threshold policies, dynamic counters, and many safety rails; useful for discovering failure modes and edge conditions.

- el_robot track (side branch focused on practical exploitability)
  - el_robot → el_robot_2 → el_robot_3: Tight range generation, bounty-aware EV shaping, adaptive “checkfold” thresholds when far ahead, and empirical bet-size models. These bots aggressively tuned practical edges vs heuristics-heavy opponents.

- w3_bot_cristi (principled postflop with simulation and texture)
  - Adds Monte Carlo hand-strength estimation per street, crude board texture classification (wet/dry), dynamic bet sizing by street, selective semi-bluffs, and opponent-type categorization (TAG/LAG/Overly Aggressive) that gates lines and thresholds.
  - Much cleaner preflop tiering and postflop continuation logic. This was a strong heuristic milestone.

- w4_bot_cristi (most advanced – learned policy guidance)
  - Introduces a learned policy/value head via a Deep CFR-inspired model (`neural_net.py`) that consumes card embeddings and a history of bet sizes. The bot (`player.py`) feeds in masked action spaces and samples among: Fold, Check, Call, Raise 1/2 pot, Raise 3/2 pot.
  - This is the most sophisticated agent in the repo: it marries the full engine integration with a NN policy, using the same skeleton protocol. It expects a PyTorch checkpoint.

## Repo layout

- engine.py: Local engine to run two bots head-to-head using socket protocol and `commands.json` in each bot directory.
- config.py: Match configuration (players, names, timeouts, stack sizes, bounty constants, etc.).
- w1_bot, w2_bot, w2_bot_cristi, w2_bot_fabian, w2_jputa, w3_bot_cristi, w4_bot_cristi: Bot directories. Each contains a `skeleton/` with protocol code and a `player.py` implementing the bot.
- el_robot, el_robot_2, el_robot_3: Alternate bot line focused on practical exploitability.
- player_chatbot: An interactive “human-in-the-loop” client; can be driven by GPT or manual input.
- training, training_cristi: CFR/Deep CFR experiments, local engines/roundstate mocks, and models. Useful for reproducing/finetuning learned policies.

## Run a match

Prereqs:
- Python 3.10+
- pip install: `eval7`, `torch` (CPU is OK), and for some variants `numpy`/`pandas`.

Quickstart:
1) Install dependencies
   - Minimal to run w4 (CPU):
     - `pip install eval7 torch`
   - For earlier bots and training scripts you may also need:
     - `pip install numpy pandas`

2) Choose players in `config.py`
   - Example (default as of now):
     - `PLAYER_1_PATH = "./w1_bot"`
     - `PLAYER_2_PATH = "./w4_bot_cristi"`
   - Swap to compare different bots (e.g., `./w3_bot_cristi` vs `./w4_bot_cristi`).

3) Run the engine
   - `python3 engine.py`
   - Logs are written to `gamelog.txt` and per-bot logs `A.txt`, `B.txt`.

Notes:
- The engine launches each bot using the bot’s `commands.json`. If you add a new bot, copy an existing `commands.json` and adjust it.
- `player_chatbot` is interactive; set `PLAYER_2_PATH = "./player_chatbot"` to play via terminal (optionally with GPT-4 if you provide an API key in that module).

## The final bot: w4_bot_cristi

This is the strongest and most “productionized” bot here. It uses a PyTorch model to select among five masked actions and reads the same game protocol as earlier bots.

- Files
  - `w4_bot_cristi/player.py`: Inference-time bot logic that wires the model to the engine’s skeleton, builds action masks, and samples an action.
  - `w4_bot_cristi/neural_net.py`: `DeepCFRModel` with card and bet-history embeddings and an action head.
  - `w4_bot_cristi/skeleton/*`: Unmodified protocol helpers (actions/states/runner/bot baseclass).

- Action space
  - Fold, Check, Call, Raise 1/2 pot, Raise 3/2 pot. A legality mask from the current `RoundState` filters invalid choices before sampling.

- Inputs and features
  - Cards: hole, flop, turn, river groups embedded and fused.
  - Bets: recent bet sizes (log-transformed and flagged for occurrence) embedded via an MLP trunk.
  - Combined representation is normalized and fed to a linear head for per-action regrets/scores.

- Checkpoint
  - `player.py` loads `models/player_1_model3.pth` relative to the bot’s working directory. Ensure the file exists at `w4_bot_cristi/models/player_1_model3.pth` or update the path in `w4_bot_cristi/player.py` accordingly.
  - If you trained a new model under `training_cristi/models/`, copy or symlink it into `w4_bot_cristi/models/`.

- Running w4 vs any bot
  - Set `PLAYER_2_PATH = "./w4_bot_cristi"` in `config.py` (and choose any opponent for `PLAYER_1_PATH`).
  - `python3 engine.py`

- Known limitations
  - Discrete raise sizes (1/2 pot and 3/2 pot) are a simplification. Extending to a larger discrete set is straightforward once the checkpoint is retrained.
  - Checkpoint path must be valid at runtime.

## Training and experimentation

- `training/` and `training_cristi/` include:
  - local_engine.py, local_roundstate.py: small harnesses to iterate faster without the full socket engine.
  - cfr.py, cfr_test.py, neural_net.py: Deep CFR-ish training loops and model definitions.
  - models/: where checkpoints are typically saved (under `training_cristi`).

To iterate on w4:
- Train in `training_cristi/` (or adapt `training/`), export a `.pth` file, and point `w4_bot_cristi/player.py` to it.

## Building your own bot

1) Copy any bot directory (e.g., `w3_bot_cristi/`) to `my_bot/`.
2) Keep the `skeleton/` directory for protocol compliance.
3) Implement your policy in `player.py`.
4) Ensure `commands.json` exists and points to `python3 player.py`.
5) Set `PLAYER_1_PATH`/`PLAYER_2_PATH` to `./my_bot` and run `engine.py`.

## Dependencies (common)

- Core: `eval7`
- ML: `torch`
- Utilities (some bots): `numpy`, `pandas`

Install whatever your chosen bots import. The engine itself only depends on `eval7`.

## FAQ

- I get a file-not-found error for the model when running w4.
  - Create `w4_bot_cristi/models/` and place `player_1_model3.pth` inside, or edit the path in `w4_bot_cristi/player.py` to where your checkpoint actually lives.

- How do I play manually?
  - Set `PLAYER_2_PATH = "./player_chatbot"` in `config.py`, then `python3 engine.py`. It will prompt you for moves in the terminal.

- Where does the bounty mechanic come from?
  - See `engine.py` and `config.py` for `BOUNTY_RATIO`, `BOUNTY_CONSTANT`, and how bounty hits are detected/applied.

---

Enjoy exploring the bots—from early heuristics to a learned policy—and feel free to iterate on the final w4 agent with your own checkpoints and action sets.