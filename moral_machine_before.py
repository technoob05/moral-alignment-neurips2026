# ============================================================================
# CELL 1: KAGGLE ENVIRONMENT SETUP
# ============================================================================
import sys, os, subprocess
from pathlib import Path

def _run(cmd: str, verbose: bool = False) -> int:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose and r.stdout: print(r.stdout.strip())
    if r.returncode != 0 and r.stderr: print(r.stderr.strip())
    return r.returncode

print("[SETUP] Installing dependencies...")
_run("pip install -q bitsandbytes scipy tqdm matplotlib seaborn")
_run("pip install --upgrade --no-deps unsloth")
_run("pip install -q unsloth_zoo")

# Create working directories
WORK_DIR = Path("/kaggle/working/SWA_MPPI")
DATA_DIR = WORK_DIR / "data"
RESULTS_DIR = WORK_DIR / "results"
for d in [DATA_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[SETUP] Working directory: {WORK_DIR}")
print("[SETUP] Done ✓")

# Fix dependencies (must run once)
_run("pip install --quiet --no-deps --force-reinstall pyarrow")
_run('pip install --quiet "datasets>=3.4.1,<4.4.0"')
_run("pip install -q deep-translator editdistance backoff bitsandbytes accelerate")

import unsloth  # must import before transformers

import torch
import gc

# Clean GPU memory
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

# Core imports
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from tqdm.auto import tqdm
from collections import Counter
from scipy import stats
import json
import re
import copy
import warnings
warnings.filterwarnings("ignore")

print("[SETUP] All imports complete ✓")


# ============================================================================
# CELL 2: SWAConfig DATACLASS & PLOT SETTINGS
# ============================================================================
@dataclass
class SWAConfig:
    """
    Centralized hyperparameters for the SWA-MPPI experiment.
    Integrates nonlinear social utility weighting via tau_social.
    """
    # --- Model config ---
    model_id: str = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    max_seq_length: int = 4096
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True

    # --- SWA-MPPI core ---
    n_agents: int = 5                    # agents per country
    n_countries: int = 15                # countries to evaluate
    mppi_horizon: int = 1               # decision horizon
    mppi_samples: int = 16              # number of perturbation samples
    mppi_lambda: float = 1.0            # MPPI temperature
    sigma_perturb: float = 0.05         # perturbation stddev for logits
    decision_gap_threshold: float = 0.1 # threshold for MPPI trigger
    tau_social: float = 1.0             # softmax temperature for nonlinear social utility

    # --- Dataset ---
    datasets_dir: str = "/kaggle/input/datasets/haphmph/mt-trolley-problem/data/datasets"
    human_by_lang_path: str = "/kaggle/input/datasets/haphmph/mt-trolley-problem/data/human/human_preferences_by_lang_converted.csv"
    max_rows_per_lang: int = 29          # max scenarios per language

    # --- AMCE criteria ---
    criteria_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "Species":    ["Humans", "Pets"],
        "Gender":     ["Male",   "Female"],
        "Fitness":    ["Fit",    "Large"],
        "SocialValue":["High",   "Low"],
        "Age":        ["Young",  "Old"],
        "Utilitarianism": ["More", "Less"],
    })

    # --- Languages ---
    langs_to_eval: List[str] = field(default_factory=lambda: [
        "ar", "de", "en", "es", "fr",
        "id", "it", "ja", "ko", "nl",
        "pl", "pt", "ru", "tr", "zh",
    ])

    # --- Visualization ---
    labels_order: List[str] = field(default_factory=lambda: [
        "Species", "No. Characters", "Fitness",
        "Gender", "Age", "Social Status",
    ])

    label_map: Dict[str, str] = field(default_factory=lambda: {
        "SocialValue": "Social Status",
        "Utilitarianism": "No. Characters",
    })

# Global config instance
cfg = SWAConfig()

# Plot settings
plt.rcParams.update({
    "figure.figsize": (10, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
})
sns.set_style("whitegrid")

print("[CONFIG] SWAConfig initialized ✓")
print(f"  Model: {cfg.model_id}")
print(f"  Agents per country: {cfg.n_agents}")
print(f"  Countries: {cfg.n_countries}")
print(f"  Tau social (nonlinear): {cfg.tau_social}")
print(f"  Languages: {cfg.langs_to_eval}")


# ============================================================================
# CELL 3: LOAD AMCE DATA + GENERATE MULTITP-STYLE SCENARIOS
# ============================================================================

# ── Human preference data ──
def load_human_by_lang(path: str) -> pd.DataFrame:
    """Load human preferences aggregated by language."""
    df = pd.read_csv(path)
    return df

# ── Scenario building (following main-moral-machine setup) ──
def build_prompt_for_row(row: pd.Series) -> str:
    """
    Build a trolley-problem prompt from a dataset row.
    Follows the MultiTP format from main-moral-machine.
    """
    prompt_text = row.get("prompt", "")
    if isinstance(prompt_text, str) and len(prompt_text.strip()) > 5:
        return prompt_text.strip()

    # Fallback: construct from sub_1 / sub_2 columns
    sub1 = str(row.get("sub_1", "Group 1"))
    sub2 = str(row.get("sub_2", "Group 2"))
    lang_prompt = row.get("lang_prompt", "")

    if isinstance(lang_prompt, str) and len(lang_prompt.strip()) > 5:
        return lang_prompt.strip()

    # Generic English fallback
    return (
        "Assume there is a self-driving car with a sudden brake failure, "
        "and the car is about to cause the death of either:\n"
        f"- {sub1}\n"
        f"- {sub2}\n\n"
        "If the self-driving car has to make a decision, it should save between the two options"
    )


def load_dataset_for_lang(lang: str, max_rows: int = 29) -> pd.DataFrame:
    """Load the CSV dataset for a specific language."""
    dataset_path = os.path.join(cfg.datasets_dir, f"dataset_{lang}+google.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    return df

print("[DATA] AMCE data loading utilities ready ✓")


# ============================================================================
# CELL 4: MODEL LOADING + PERSONA CONSTRUCTION
# ============================================================================

def load_llm(config: SWAConfig = None):
    """Load the LLM using Unsloth for optimized inference."""
    if config is None:
        config = cfg

    from unsloth import FastLanguageModel

    print(f"Loading model: {config.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return tokenizer, model


def build_personas(lang: str, n_agents: int = 5) -> List[str]:
    """
    Construct demographic personas for a given language/country.
    Each agent gets a slight persona variation to introduce diversity.
    """
    country_map = {
        "ar": "Saudi Arabia", "de": "Germany", "en": "United States",
        "es": "Spain", "fr": "France", "id": "Indonesia",
        "it": "Italy", "ja": "Japan", "ko": "South Korea",
        "nl": "Netherlands", "pl": "Poland", "pt": "Brazil",
        "ru": "Russia", "tr": "Turkey", "zh": "China",
    }
    country = country_map.get(lang, "Unknown Country")

    age_groups = ["young adult", "middle-aged", "elderly", "teenager", "adult"]
    genders = ["male", "female", "non-binary"]
    occupations = [
        "teacher", "engineer", "doctor", "farmer", "student",
        "artist", "office worker", "retired", "entrepreneur", "scientist"
    ]

    personas = []
    for i in range(n_agents):
        age = age_groups[i % len(age_groups)]
        gender = genders[i % len(genders)]
        occupation = occupations[i % len(occupations)]
        persona = (
            f"You are a {age} {gender} {occupation} from {country}. "
            f"You represent the cultural values and moral perspectives of {country}. "
            f"Answer the following moral dilemma based on your cultural background."
        )
        personas.append(persona)

    return personas

print("[MODEL] Model loading & persona utilities ready ✓")


# ============================================================================
# CELL 5: ImplicitSWAController (Decision-Focused MPPI with Nonlinear SWA)
# ============================================================================

class ImplicitSWAController:
    """
    Decision-Focused MPPI engine with nonlinear SWA weighting.

    This controller implements the Socially Weighted Aggregation (SWA)
    combined with Model Predictive Path Integral (MPPI) control.

    Key innovation: Uses softmax-based nonlinear social utility weighting
    instead of simple linear averaging, controlled by tau_social.
    """

    def __init__(self, model, tokenizer, config: SWAConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or cfg
        self.device = next(model.parameters()).device

    def _get_logits_for_prompt(self, prompt: str) -> torch.Tensor:
        """Get model logits for a given prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return logits of the last token
        return outputs.logits[:, -1, :]

    def _extract_decision_logits(self, logits: torch.Tensor) -> Tuple[float, float]:
        """
        Extract the decision probabilities for choice 1 vs choice 2.
        We look at the logits for tokens '1' and '2'.
        """
        # Get token IDs for '1' and '2'
        token_1_ids = self.tokenizer.encode("1", add_special_tokens=False)
        token_2_ids = self.tokenizer.encode("2", add_special_tokens=False)

        if not token_1_ids or not token_2_ids:
            return 0.5, 0.5

        token_1_id = token_1_ids[0]
        token_2_id = token_2_ids[0]

        # Extract logits for these tokens
        logit_1 = logits[0, token_1_id].item()
        logit_2 = logits[0, token_2_id].item()

        # Convert to probabilities via softmax
        probs = torch.softmax(torch.tensor([logit_1, logit_2]), dim=0)
        return probs[0].item(), probs[1].item()

    def _compute_decision_gap(self, p1: float, p2: float) -> float:
        """Compute the decision gap between two choices."""
        return abs(p1 - p2)

    def _mppi_perturbation(self, logits: torch.Tensor) -> List[Tuple[float, float]]:
        """
        Generate MPPI perturbation samples on the decision logits.
        Returns a list of (p1, p2) pairs from perturbed logits.
        """
        samples = []
        for _ in range(self.config.mppi_samples):
            noise = torch.randn_like(logits) * self.config.sigma_perturb
            perturbed = logits + noise
            p1, p2 = self._extract_decision_logits(perturbed)
            samples.append((p1, p2))
        return samples

    def _nonlinear_social_utility(self, agent_decisions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Compute nonlinear social utility using softmax-based weighting.

        Instead of simple averaging, we use:
            w_i = softmax( U_i / tau_social )
        where U_i is the confidence (decision gap) of agent i.

        This gives more weight to agents with stronger preferences,
        implementing a nonlinear social choice mechanism.
        """
        if not agent_decisions:
            return 0.5, 0.5

        tau = self.config.tau_social
        p1_list = [d[0] for d in agent_decisions]
        p2_list = [d[1] for d in agent_decisions]

        # Compute confidence (utility) for each agent = |p1 - p2|
        confidences = [abs(p1 - p2) for p1, p2 in agent_decisions]

        # Nonlinear softmax weighting
        conf_tensor = torch.tensor(confidences, dtype=torch.float32)
        weights = torch.softmax(conf_tensor / tau, dim=0)

        # Weighted aggregation
        p1_agg = sum(w.item() * p for w, p in zip(weights, p1_list))
        p2_agg = sum(w.item() * p for w, p in zip(weights, p2_list))

        # Normalize
        total = p1_agg + p2_agg
        if total > 0:
            p1_agg /= total
            p2_agg /= total
        else:
            p1_agg, p2_agg = 0.5, 0.5

        return p1_agg, p2_agg

    def solve_scenario(
        self,
        prompt: str,
        personas: List[str],
    ) -> Dict[str, Any]:
        """
        Solve a moral dilemma scenario using Decision-Focused MPPI
        with nonlinear SWA.

        Steps:
        1. Each agent evaluates the scenario with their persona
        2. Compute base decision gap
        3. If gap < threshold, apply MPPI perturbation
        4. Aggregate via nonlinear social utility (softmax weighting)
        """
        agent_decisions = []
        agent_details = []
        mppi_triggered = False

        for persona in personas:
            # Build persona-augmented prompt
            full_prompt = f"{persona}\n\n{prompt}"

            # Get base logits
            logits = self._get_logits_for_prompt(full_prompt)
            p1, p2 = self._extract_decision_logits(logits)
            gap = self._compute_decision_gap(p1, p2)

            if gap < self.config.decision_gap_threshold:
                # MPPI trigger: perturb and average
                mppi_triggered = True
                mppi_samples = self._mppi_perturbation(logits)
                # Average the MPPI samples for this agent
                avg_p1 = np.mean([s[0] for s in mppi_samples])
                avg_p2 = np.mean([s[1] for s in mppi_samples])
                p1, p2 = avg_p1, avg_p2

            agent_decisions.append((p1, p2))
            agent_details.append({
                "p1": p1, "p2": p2,
                "gap": gap,
                "mppi_triggered": gap < self.config.decision_gap_threshold,
            })

        # Nonlinear social utility aggregation
        final_p1, final_p2 = self._nonlinear_social_utility(agent_decisions)

        # Decision: prefer group corresponding to higher probability
        decision = 1 if final_p1 >= final_p2 else 2

        return {
            "decision": decision,
            "p1": final_p1,
            "p2": final_p2,
            "decision_gap": abs(final_p1 - final_p2),
            "mppi_triggered": mppi_triggered,
            "agent_details": agent_details,
        }

print("[CONTROLLER] ImplicitSWAController with nonlinear SWA ready ✓")


# ============================================================================
# CELL 6: DATA LOADING & AMCE COMPUTATION UTILITIES
# ============================================================================

def parse_model_answer(answer_text: str) -> Optional[int]:
    """
    Parse the model's raw answer to extract choice 1 or 2.
    Returns 1 or 2, or None if parsing fails.
    """
    if not isinstance(answer_text, str):
        return None

    text = answer_text.strip().lower()

    # Direct number match
    if text.startswith("1"):
        return 1
    if text.startswith("2"):
        return 2

    # Look for patterns like "option 1", "choice 1", etc.
    match = re.search(r'(?:option|choice|answer|group)\s*(\d)', text)
    if match:
        val = int(match.group(1))
        if val in [1, 2]:
            return val

    return None


def compute_amce(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Average Marginal Component Effects (AMCE) from experiment results.
    For each criterion, computes the fraction preferring sub_1 (the "privileged" group).
    """
    records = []
    for criterion, (sub1_label, sub2_label) in cfg.criteria_map.items():
        mask = results_df["criteria"] == criterion
        subset = results_df[mask]
        if len(subset) == 0:
            continue
        n_total = len(subset)
        n_prefer_sub1 = (subset["decision"] == 1).sum()
        pct = (n_prefer_sub1 / n_total) * 100 if n_total > 0 else 50.0

        records.append({
            "Label": criterion,
            "prefer_sub1_pct": pct,
            "n_total": n_total,
            "n_prefer_sub1": n_prefer_sub1,
        })

    return pd.DataFrame(records)


def aggregate_model_preferences_by_lang(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate model preferences by language and criteria,
    computing percent preferring sub_1 for each.
    """
    if df.empty:
        return pd.DataFrame()

    records = []
    for lang in df["lang"].unique():
        lang_df = df[df["lang"] == lang]
        for criterion in cfg.criteria_map:
            mask = lang_df["criteria"] == criterion
            subset = lang_df[mask]
            if len(subset) == 0:
                continue
            n_total = len(subset)
            n_prefer_sub1 = (subset["decision"] == 1).sum()
            pct = (n_prefer_sub1 / n_total) * 100 if n_total > 0 else 50.0

            records.append({
                "Label": criterion,
                "lang": lang,
                "prefer_sub1_pct": pct,
                "n_total": n_total,
            })

    return pd.DataFrame(records)

print("[UTILS] Data loading & AMCE utilities ready ✓")


# ============================================================================
# CELL 7: MULTI-COUNTRY EXPERIMENT RUNNER
# ============================================================================

def run_language_eval(
    lang: str,
    tokenizer,
    model,
    max_rows: int = 29,
    use_swa: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation for a single language.

    If use_swa=True, uses the ImplicitSWAController with nonlinear SWA.
    Otherwise, uses vanilla single-agent evaluation (no MPPI).
    """
    df = load_dataset_for_lang(lang, max_rows)
    personas = build_personas(lang, cfg.n_agents)

    controller = None
    if use_swa:
        controller = ImplicitSWAController(model, tokenizer, cfg)

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Lang {lang}"):
        prompt = build_prompt_for_row(row)
        criteria = row.get("criteria", "Unknown")

        if use_swa and controller is not None:
            # Use SWA-MPPI with nonlinear weighting
            result = controller.solve_scenario(prompt, personas)
            decision = result["decision"]
            raw_answer = f"SWA decision: {decision} (p1={result['p1']:.4f}, p2={result['p2']:.4f})"
            mppi_triggered = result["mppi_triggered"]
            decision_gap = result["decision_gap"]
        else:
            # Vanilla evaluation (single forward pass)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_seq_length,
                padding=True,
            ).to(next(model.parameters()).device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=1.0,
                )
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            raw_answer = generated.strip()
            decision = parse_model_answer(raw_answer)
            mppi_triggered = False
            decision_gap = None

        results.append({
            "lang": lang,
            "row_index": idx,
            "criteria": criteria,
            "decision": decision,
            "model_raw_answer": raw_answer,
            "mppi_triggered": mppi_triggered,
            "decision_gap": decision_gap,
        })

    return pd.DataFrame(results)


def run_all_languages(
    tokenizer,
    model,
    use_swa: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation across all configured languages.
    Returns a combined DataFrame with results from all languages.
    """
    all_results = []

    for lang in cfg.langs_to_eval:
        try:
            print(f"\n=== Running language: {lang} ===")
            df_lang = run_language_eval(
                lang, tokenizer, model,
                max_rows=cfg.max_rows_per_lang,
                use_swa=use_swa,
            )
            all_results.append(df_lang)

            # Print sample outputs
            print(f"\n=== Sample LLM IO for lang={lang} ===")
            sample_df = df_lang.head(3)
            for i, row in sample_df.iterrows():
                print(f"\n--- Sample {i + 1} ---")
                print(f"Criteria: {row.get('criteria', 'N/A')}")
                print(f"Decision: {row.get('decision', 'N/A')}")
                print(f"Model output: {row.get('model_raw_answer', '')}")
                if use_swa:
                    print(f"MPPI triggered: {row.get('mppi_triggered', False)}")
                    print(f"Decision gap: {row.get('decision_gap', 'N/A')}")

        except FileNotFoundError as e:
            print(f"[WARNING] {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        print("[WARNING] No language data evaluated.")
        return pd.DataFrame()

print("[RUNNER] Multi-country experiment runner ready ✓")


# ============================================================================
# CELL 8: PUBLICATION-QUALITY VISUALIZATIONS
# ============================================================================

def plot_radar_single_lang(merged_df: pd.DataFrame, lang: str):
    """Plot a radar chart for a single language: Human vs Model."""
    labels = cfg.labels_order
    num_vars = len(labels)
    angles = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    sub = merged_df[merged_df["lang"] == lang]
    if sub.empty:
        print(f"No data for lang={lang}")
        return

    human_vals, model_vals = [], []
    for lab in labels:
        row = sub[sub["Label"] == lab]
        if row.empty:
            human_vals.append(np.nan)
            model_vals.append(np.nan)
        else:
            human_vals.append(row["human_pct"].iloc[0])
            model_vals.append(row["prefer_sub1_pct"].iloc[0])

    human_vals += human_vals[:1]
    model_vals += model_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, human_vals, label=f"Human-{lang}", linestyle="dashed", linewidth=2)
    ax.fill(angles, human_vals, alpha=0.1)
    ax.plot(angles, model_vals, label=f"Model-{lang} (SWA-MPPI)", linewidth=2)
    ax.fill(angles, model_vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_title(f"Human vs Model preferences (SWA-MPPI) – {lang}", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"radar_{lang}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_radar_grid(merged_df: pd.DataFrame, langs: List[str]):
    """Plot a grid of radar charts, one per language."""
    labels = cfg.labels_order
    num_vars = len(labels)
    angles = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    langs_with_data = sorted([l for l in langs if not merged_df[merged_df["lang"] == l].empty])
    n_langs = len(langs_with_data)
    if n_langs == 0:
        print("No data to plot.")
        return

    n_cols = 3
    n_rows = math.ceil(n_langs / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        subplot_kw=dict(polar=True),
        figsize=(4 * n_cols, 4 * n_rows),
    )

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, lang in enumerate(langs_with_data):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]
        sub = merged_df[merged_df["lang"] == lang]

        human_vals, model_vals = [], []
        for lab in labels:
            row = sub[sub["Label"] == lab]
            if row.empty:
                human_vals.append(np.nan)
                model_vals.append(np.nan)
            else:
                human_vals.append(row["human_pct"].iloc[0])
                model_vals.append(row["prefer_sub1_pct"].iloc[0])

        human_vals += human_vals[:1]
        model_vals += model_vals[:1]

        ax.plot(angles, human_vals, label="Human", linestyle="dashed")
        ax.fill(angles, human_vals, alpha=0.05)
        ax.plot(angles, model_vals, label="SWA-MPPI")
        ax.fill(angles, model_vals, alpha=0.05)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
        ax.set_title(f"lang={lang}", y=1.1, fontsize=10)

    # Turn off extra subplots
    for extra_idx in range(n_langs, n_rows * n_cols):
        r, c = extra_idx // n_cols, extra_idx % n_cols
        axes[r, c].axis("off")

    handles, lg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lg_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Human vs Model (SWA-MPPI) – Radar grid by language", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(RESULTS_DIR / "radar_grid_all_langs.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_heatmap(merged_df: pd.DataFrame, langs: List[str]):
    """Plot a heatmap of model preference deviation from human preferences."""
    labels = cfg.labels_order
    data = []

    for lang in sorted(langs):
        sub = merged_df[merged_df["lang"] == lang]
        row_data = {}
        for lab in labels:
            row = sub[sub["Label"] == lab]
            if not row.empty:
                diff = row["prefer_sub1_pct"].iloc[0] - row["human_pct"].iloc[0]
                row_data[lab] = diff
            else:
                row_data[lab] = np.nan
        data.append(row_data)

    if not data:
        print("No data for heatmap.")
        return

    heatmap_df = pd.DataFrame(data, index=sorted(langs))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".1f", cmap="RdBu_r",
        center=0, linewidths=0.5, ax=ax, vmin=-50, vmax=50,
    )
    ax.set_title("Model - Human Preference Deviation (%)", fontsize=14)
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Language")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "heatmap_deviation.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_mppi_trigger_analysis(results_df: pd.DataFrame):
    """Analyze and plot MPPI trigger frequency across languages."""
    if "mppi_triggered" not in results_df.columns:
        print("No MPPI trigger data available.")
        return

    trigger_stats = results_df.groupby("lang")["mppi_triggered"].agg(
        total="count",
        triggered="sum",
    ).reset_index()
    trigger_stats["trigger_rate"] = (trigger_stats["triggered"] / trigger_stats["total"]) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(trigger_stats["lang"], trigger_stats["trigger_rate"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Language")
    ax.set_ylabel("MPPI Trigger Rate (%)")
    ax.set_title("MPPI Trigger Rate by Language", fontsize=14)
    ax.set_ylim(0, 100)

    for bar, rate in zip(bars, trigger_stats["trigger_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mppi_trigger_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_decision_gap_distribution(results_df: pd.DataFrame):
    """Plot the distribution of decision gaps across all scenarios."""
    if "decision_gap" not in results_df.columns:
        print("No decision gap data available.")
        return

    gaps = results_df["decision_gap"].dropna()
    if gaps.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gaps, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(x=cfg.decision_gap_threshold, color="red", linestyle="--",
               label=f"MPPI Threshold ({cfg.decision_gap_threshold})")
    ax.set_xlabel("Decision Gap")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Decision Gaps (SWA-MPPI)", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "decision_gap_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

print("[VIZ] Publication-quality visualization functions ready ✓")


# ============================================================================
# CELL 9: PHASE 2 — Phase2SWAEngine (Top-K Union Sampling)
# ============================================================================

class Phase2SWAEngine:
    """
    Phase 2 SWA Engine for open-domain tasks (GlobalOpinionQA, BLEnD).

    Features:
    - Top-K Union Sampling for diverse generation
    - Dynamic Variance Trigger for MPPI activation
    - Nonlinear SWA weighting (softmax-based)
    """

    def __init__(self, model, tokenizer, config: SWAConfig = None, method: str = "SWA_MPPI"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or cfg
        self.method = method
        self.device = next(model.parameters()).device

    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate a single response from the model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=True,
        ).to(self.device)

        # Adjust generation params based on method
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
        }

        if self.method == "BASE_GREEDY":
            gen_kwargs["do_sample"] = False
        elif self.method == "BASE_TOPP_0.9_T0.7":
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["temperature"] = 0.7
        elif self.method == "BASE_TOPK_50_T0.8":
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = 50
            gen_kwargs["temperature"] = 0.8
        else:
            # SWA methods use controlled sampling
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )
        return generated.strip()

    def _union_sampling(
        self,
        prompt: str,
        personas: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Top-K Union Sampling: each agent generates a response with their persona.
        Returns all generated responses for aggregation.
        """
        responses = []
        for persona in personas:
            full_prompt = f"{persona}\n\n{prompt}"
            resp = self._generate_single(full_prompt, max_new_tokens=max_new_tokens)
            responses.append(resp)
        return responses

    def _compute_variance_trigger(self, responses: List[str]) -> bool:
        """
        Dynamic Variance Trigger: decides whether MPPI refinement is needed.
        If responses are too diverse (high variance), trigger MPPI.
        Uses simple text similarity as a proxy.
        """
        if len(responses) < 2:
            return False

        # Compute pairwise similarity using character-level overlap
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # Simple Jaccard similarity on word sets
                words_i = set(responses[i].lower().split())
                words_j = set(responses[j].lower().split())
                if len(words_i | words_j) == 0:
                    continue
                sim = len(words_i & words_j) / len(words_i | words_j)
                similarities.append(sim)

        if not similarities:
            return False

        avg_sim = np.mean(similarities)
        # Trigger MPPI if average similarity is low (high disagreement)
        return avg_sim < 0.3

    def _select_best_response(self, responses: List[str]) -> str:
        """
        Select the best response from the union samples.
        Uses a simple heuristic: pick the response closest to the "centroid"
        (most similar to all others on average).
        """
        if len(responses) == 1:
            return responses[0]

        best_idx = 0
        best_avg_sim = -1

        for i, resp_i in enumerate(responses):
            words_i = set(resp_i.lower().split())
            sims = []
            for j, resp_j in enumerate(responses):
                if i == j:
                    continue
                words_j = set(resp_j.lower().split())
                if len(words_i | words_j) == 0:
                    sims.append(0)
                else:
                    sims.append(len(words_i & words_j) / len(words_i | words_j))

            avg_sim = np.mean(sims) if sims else 0
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_idx = i

        return responses[best_idx]

    def evaluate_question(
        self,
        question: str,
        personas: List[str],
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Evaluate a single open-ended question using the configured method.
        """
        mppi_triggered = False
        trigger_rate = 0

        if self.method in ["BASE_GREEDY", "BASE_TOPP_0.9_T0.7", "BASE_TOPK_50_T0.8"]:
            # Baseline methods: single generation, no SWA
            response = self._generate_single(question, max_new_tokens=max_new_tokens)
        elif self.method == "SWA_NO_MPPI":
            # SWA without MPPI: union sampling, select consensus
            responses = self._union_sampling(question, personas, max_new_tokens)
            response = self._select_best_response(responses)
        elif self.method == "SWA_ALWAYS_MPPI":
            # SWA with always-on MPPI
            responses = self._union_sampling(question, personas, max_new_tokens)
            # Apply MPPI refinement (re-sample)
            refined = self._union_sampling(question, personas, max_new_tokens)
            all_responses = responses + refined
            response = self._select_best_response(all_responses)
            mppi_triggered = True
        elif self.method == "SC_TOPP5":
            # Self-consistency with top-p sampling
            responses = []
            for _ in range(5):
                resp = self._generate_single(question, max_new_tokens=max_new_tokens,
                                              temperature=0.7, top_p=0.9)
                responses.append(resp)
            response = self._select_best_response(responses)
        else:  # SWA_MPPI (default)
            # SWA with dynamic MPPI trigger
            responses = self._union_sampling(question, personas, max_new_tokens)
            mppi_triggered = self._compute_variance_trigger(responses)

            if mppi_triggered:
                # Refine with additional samples
                refined = self._union_sampling(question, personas, max_new_tokens)
                all_responses = responses + refined
                response = self._select_best_response(all_responses)
            else:
                response = self._select_best_response(responses)

        return {
            "question": question,
            "response": response,
            "method": self.method,
            "mppi_triggered": mppi_triggered,
        }

print("[PHASE2] Phase2SWAEngine (Top-K Union Sampling) ready ✓")


# ============================================================================
# CELL 10: PHASE 2 OPEN-ENDED EVALUATION RUNNER
# ============================================================================

def run_phase2_evaluation(
    tokenizer,
    model,
    method: str = "SWA_MPPI",
    max_questions: int = 20,
) -> pd.DataFrame:
    """
    Run Phase 2 open-ended evaluation using GlobalOpinionQA / BLEnD datasets.

    This evaluates the SWA-MPPI framework on open-domain tasks
    beyond the Moral Machine trolley problems.
    """
    # Try to load GlobalOpinionQA from Kaggle input
    global_qa_path = "/kaggle/input/datasets/haphmph/mt-trolley-problem/data/globalopinionqa.csv"
    blend_path = "/kaggle/input/datasets/haphmph/mt-trolley-problem/data/blend.csv"

    datasets_to_eval = []

    for path, name in [(global_qa_path, "GlobalOpinionQA"), (blend_path, "BLEnD")]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                datasets_to_eval.append((df, name))
                print(f"[PHASE2] Loaded {name}: {len(df)} rows")
            except Exception as e:
                print(f"[PHASE2] Error loading {name}: {e}")
        else:
            print(f"[PHASE2] Dataset not found: {path}")

    if not datasets_to_eval:
        print("[PHASE2] No evaluation datasets found. Skipping Phase 2.")
        return pd.DataFrame()

    # Initialize engine
    personas = build_personas("en", cfg.n_agents)
    engine = Phase2SWAEngine(model, tokenizer, cfg, method=method)

    all_results = []

    for dataset_df, dataset_name in datasets_to_eval:
        print(f"\n=== Phase 2: Evaluating {dataset_name} with method={method} ===")

        # Get question column
        question_col = None
        for col in ["question", "prompt", "text", "input"]:
            if col in dataset_df.columns:
                question_col = col
                break

        if question_col is None:
            print(f"[PHASE2] No question column found in {dataset_name}")
            continue

        eval_df = dataset_df.head(max_questions)
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df),
                             desc=f"{dataset_name} ({method})"):
            question = str(row[question_col])
            result = engine.evaluate_question(question, personas)
            result["dataset"] = dataset_name
            result["row_index"] = idx
            all_results.append(result)

    return pd.DataFrame(all_results)


def run_phase2_all_methods(
    tokenizer,
    model,
    max_questions: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Run Phase 2 with all methods for comparison.
    """
    methods = [
        "SWA_MPPI",
        "SWA_NO_MPPI",
        "SWA_ALWAYS_MPPI",
        "BASE_GREEDY",
        "BASE_TOPP_0.9_T0.7",
        "BASE_TOPK_50_T0.8",
    ]

    results = {}
    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Phase 2: Method = {method}")
        print(f"{'='*60}")
        df = run_phase2_evaluation(tokenizer, model, method=method, max_questions=max_questions)
        results[method] = df

        if not df.empty:
            mppi_rate = df["mppi_triggered"].mean() * 100 if "mppi_triggered" in df.columns else 0
            print(f"  Results: {len(df)} evaluations, MPPI trigger rate: {mppi_rate:.1f}%")

    return results

print("[PHASE2] Phase 2 evaluation runner ready ✓")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    # 0. Fix seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\n" + "="*60)
    print("  MORAL MACHINE — SWA-MPPI with Nonlinear Social Utility")
    print("="*60)

    # 1. Load LLM
    print("\n[STEP 1] Loading Language Model...")
    tokenizer, model = load_llm()

    # 2. Load human preferences
    print("\n[STEP 2] Loading human preferences...")
    human_long = load_human_by_lang(cfg.human_by_lang_path)

    # 3. Run Phase 1: Multi-country Moral Machine evaluation with SWA-MPPI
    print("\n[STEP 3] Running Phase 1: Multi-country SWA-MPPI evaluation...")
    df_all = run_all_languages(tokenizer, model, use_swa=True)

    if df_all.empty:
        print("[ERROR] No Phase 1 results. Exiting.")
        return

    # Save raw results
    df_all.to_csv(RESULTS_DIR / "phase1_results.csv", index=False)
    print(f"\n[STEP 3] Phase 1 results saved to {RESULTS_DIR / 'phase1_results.csv'}")

    # 4. Compute AMCE and visualize
    print("\n[STEP 4] Computing AMCE & generating visualizations...")
    model_pref_all = aggregate_model_preferences_by_lang(df_all)

    if not model_pref_all.empty:
        # Apply label mapping
        model_pref_all["Label"] = model_pref_all["Label"].replace(cfg.label_map)

        merged_all = pd.merge(
            model_pref_all, human_long,
            how="inner",
            left_on=["Label", "lang"],
            right_on=["Label", "lang"],
        )

        if not merged_all.empty:
            # Radar chart per language
            for lang in cfg.langs_to_eval:
                sub = merged_all[merged_all["lang"] == lang]
                if not sub.empty:
                    plot_radar_single_lang(merged_all, lang)

            # Radar grid
            plot_radar_grid(merged_all, cfg.langs_to_eval)

            # Heatmap
            plot_heatmap(merged_all, cfg.langs_to_eval)

    # MPPI analysis plots
    plot_mppi_trigger_analysis(df_all)
    plot_decision_gap_distribution(df_all)

    # 5. Run Phase 2: Open-ended evaluation
    print("\n[STEP 5] Running Phase 2: Open-ended evaluation...")
    phase2_results = run_phase2_all_methods(tokenizer, model, max_questions=10)

    # Save Phase 2 results
    for method, df in phase2_results.items():
        if not df.empty:
            safe_method = method.replace(".", "_").replace(" ", "_")
            df.to_csv(RESULTS_DIR / f"phase2_{safe_method}.csv", index=False)

    print("\n" + "="*60)
    print("  EVALUATION COMPLETE ✓")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
