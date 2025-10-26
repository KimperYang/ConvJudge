import random, copy, json, uuid, datetime
from typing import Dict, Any, List, Tuple
from azure_gpt_call import call_chat_completion  # your function

GENERAL_KEYS = [
    "Category 1: Universal Compliance / General Knowledge",
    "Category 3: Condition Triggered Guidelines",
]

# ---------------------------
# Sampling helpers
# ---------------------------

def sample_intent(oracle_category2: Dict[str, Any]) -> str:
    """
    Choose one intent that is a list (i.e., step-based flow).
    Raises ValueError if none are step-like.
    """
    if not isinstance(oracle_category2, dict) or not oracle_category2:
        raise ValueError("Category 2 dictionary cannot be empty.")
    intents = [k for k, v in oracle_category2.items() if isinstance(v, list)]
    if not intents:
        raise ValueError("No step-based intents found in Category 2.")
    return random.choice(intents)

def sample_intent_steps(
    oracle_intent_steps: List[str],
    violation_intent_steps: List[str],
    max_violations: int = 5
) -> Tuple[List[str], List[int]]:
    """
    Randomly replace 0..max_violations steps with violated versions (bounded by len(steps)).
    Returns (result_steps, violated_idx_list).
    """
    n = len(oracle_intent_steps)
    if n == 0:
        return [], []

    # BUGFIX: cap at len(steps)
    vcount = random.randint(1, min(max_violations, n))
    violated_idx = set(random.sample(range(n), vcount)) if vcount > 0 else set()

    result_steps = copy.deepcopy(oracle_intent_steps)
    for idx in violated_idx:
        # Defensive check: both lists must align in length
        if idx < len(violation_intent_steps):
            result_steps[idx] = violation_intent_steps[idx]
    return result_steps, sorted(list(violated_idx))

def sample_general_guidelines(
    oracle_guidelines: Dict[str, Any],
    violation_guidelines: Dict[str, Any],
    violation_num: int = 1
) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Sample general guidelines from Category 1 & 3, then flip 'violation_num' of them
    to their violated versions if available.

    Returns:
      must_follow_general: {category: {key: correct_text}}
      must_violate_general: [ {category: cat, guideline_key: key} ... ]
      presentation_general: {category: {key: text_used_in_prompt}} (mixture with replaced violated text)
    """
    # Flatten correct/violated under GENERAL_KEYS but preserve category/key names.
    correct_pool = []  # (cat, key, text)
    violated_pool_map = {}  # (cat, key) -> violated_text

    for cat in GENERAL_KEYS:
        if cat in oracle_guidelines and isinstance(oracle_guidelines[cat], dict):
            for k, v in oracle_guidelines[cat].items():
                correct_pool.append((cat, k, v))
        if cat in violation_guidelines and isinstance(violation_guidelines[cat], dict):
            for k, v in violation_guidelines[cat].items():
                violated_pool_map[(cat, k)] = v

    if not correct_pool:
        # nothing to sample; return empties
        return {}, [], {}

    # BUGFIX: cap violation_num by available keys
    k = min(violation_num, len(correct_pool))
    flip_pairs = set(random.sample(range(len(correct_pool)), k))

    must_follow_general: Dict[str, Dict[str, str]] = {}
    must_violate_general: List[Dict[str, str]] = []
    presentation_general: Dict[str, Dict[str, str]] = {}

    for i, (cat, key, correct_text) in enumerate(correct_pool):
        # By default: correct everywhere
        must_follow_general.setdefault(cat, {})[key] = correct_text
        presentation_general.setdefault(cat, {})[key] = correct_text

        # Flip 'k' items to their violated variant (if exists)
        if i in flip_pairs and (cat, key) in violated_pool_map:
            violated_text = violated_pool_map[(cat, key)]
            # presentation shows violated text (to ensure the generator knows which text to follow for violation)
            presentation_general[cat][key] = violated_text
            # remove from must_follow (since it's a must-violate this round)
            del must_follow_general[cat][key]
            if len(must_follow_general[cat]) == 0:
                del must_follow_general[cat]
            # record as must-violate label
            must_violate_general.append({"category": cat, "guideline_key": key})

    return must_follow_general, must_violate_general, presentation_general

# ---------------------------
# Prompt builders
# ---------------------------

def build_system_prompt() -> str:
    return (
        "You are a data generator that writes realistic airline contact-center conversations between a CUSTOMER and an AGENT (Celestar Air). The AGENT should respond correctly following some company guidelines. "
        "However, you need to intentionally make some mistakes to synthesize flawed conversations. "
        "Therefore, some guidelines are given already in their incorrect versions so you can directly follow them. These guidelines must be reflected in your generated conversation, so that the conversation has some mistakes.\n\n"
        "TASK OVERVIEW\n"
        "You will receive:\n"
        "1) A set of GENERAL guidelines (general behavior), with labels indicating which ones are '(Incorrect)'.\n"
        "2) A single user INTENT with its step-by-step guidelines (phase list), with labels indicating which steps are '(Incorrect)'.\n"
        "You must write a natural, coherent conversation that:\n"
        "- Realistically reflects the chosen INTENT (e.g., new booking, change booking, transfer steps).\n"
        "- Demonstrates ALL of the specified incorrect guidelines exactly once in AGENT turns. However, introduce subtle guideline violations to mimic realistic behavior, ensuring they are not overly salient or deliberate.\n"
        "- Respects correct guidelines (do not violate them).\n"
        "- Does not need to explicitly mention every correct guideline—only ensure no violations of them.\n"
        "- Uses English. Keep 10–18 total turns (USER+AGENT). Make the scenario plausible (airports, dates, etc.).\n"
        # "- The AGENT is \"Nova\". Keep airline brand consistent (Celestar Air).\n\n"
        "ANNOTATION REQUIREMENTS\n"
        "After the conversation, produce precise mistake annotations:\n"
        "- Each annotation must identify the exact turn index (0-based across the whole conversation) where the AGENT violates the original guideline.\n"
        "- Each annotation must specify: guideline_type (if this is an intent guideline, use intent name), and guideline_phase (only apply for intent guidelines, only include a Phase number here (only number without Phase as prefix). Use -1 for general guidelines) for the INTENT.\n"
        "- Include a short evidence snippet (quoted text) from the mistaken AGENT message.\n\n"
        "OUTPUT FORMAT (strict JSON)\n"
        "{\n"
        "  \"meta\": {\n"
        "    \"intent\": \"<intent_name>\",\n"
        "    \"mistakes_planned\": [\n"
        "      {\"guideline_type\": \"...\", \"guideline_phase\": \"...\"}\n"
        "    ]\n"
        "  },\n"
        "  \"conversation\": [\n"
        "    {\"turn_index\": 0,\"role\":\"user\",\"content\":\"...\"},\n"
        "    {\"turn_index\": 1,\"role\":\"assistant\",\"content\":\"...\"}\n"
        "  ],\n"
        "  \"mistakes\": [\n"
        "    {\n"
        "      \"turn_index\": 5,\n"
        "      \"guideline_type\": \"...\",\n"
        "      \"guideline_phase\": 2,\n"
        "      \"evidence\": \"<short quote from the violating assistant message>\"\n"
        "      \"incorrect guideline\": \"<how the incorrect guideline is followed>\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "RULES\n"
        "- Ensure mistakes occur on AGENT (assistant) turns.\n"
        "- Ensure mistake should only occur once and corresponds to one specific Agent turn.\n"
        "- Indexing starts at 0 for the first item in the \"conversation\" array.\n"
        "- Be concise yet realistic; keep flight details plausible.\n"
    )

def build_user_prompt(
    must_follow_general: Dict[str, Dict[str, str]],
    must_violate_general: List[Dict[str, str]],
    presentation_general: Dict[str, Dict[str, str]],
    intent_key: str,
    intent_steps: List[str],
    violated_step_idx: List[int]
) -> str:
    def fmt_dict(d: Dict[str, Dict[str, str]]) -> str:
        lines = []
        for cat, m in d.items():
            for key, text in m.items():
                tag = "(Incorrect)" if  {'category': cat, 'guideline_key': key} in must_violate_general else "(Correct)"
                lines.append(f"- type: {key}\n{text} {tag}")
        return "\n".join(lines) if lines else "- (none)"

    general_text_block = fmt_dict(presentation_general)

    # Build intent steps block
    intent_lines = []
    for idx, step in enumerate(intent_steps):
        tag = "(Incorrect)" if idx in violated_step_idx else "(Correct)"
        intent_lines.append(f"- {step} {tag}")

    intent_block = "\n".join(intent_lines)

    prompt = (
        "GENERAL GUIDELINES:\n"
        f"{general_text_block}\n\n"
        f"USER INTENT: {intent_key}\n\n"
        "INTENT WORKFLOW GUIDELINES (each step labeled):\n"
        f"{intent_block}\n\n"
        "INSTRUCTIONS\n"
        "- Generate a realistic conversation satisfying INTENT and the step labels.\n"
        "- Insert AGENT violations only for the GENERAL MUST-VIOLATE items and intent workflow steps labeled (Incorrect).\n"
        "- Ensure all violations appear in AGENT turns and report exact turn indices in the output JSON. Each violation should be reflected and annotated in exact only one turn.\n"
        "- Keep the conversation between 10 and 18 turns inclusive.\n"
    )
    return prompt

# ---------------------------
# Main generator
# ---------------------------

def generate_conversation(
    oracle_data_path: str,
    violation_data_path: str,
    system_prompt: str,
    model: str = "gpt-5",
    max_violation_num: int = 3,
    seed: int = None
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    with open(oracle_data_path, 'r', encoding='utf-8') as f:
        oracle_data = json.load(f)
    with open(violation_data_path, 'r', encoding='utf-8') as f:
        violation_data = json.load(f)

    # 1) Pick one intent from Category 2 that is a list
    cat2 = oracle_data.get("Category 2: Intent Triggered Guidelines", {})
    intent_key = sample_intent(cat2)

    # 2) Build intent steps, possibly injecting violations
    has_violated_intent_guidelines = random.choice([True, False])
    if has_violated_intent_guidelines:
        intent_steps, violated_idx = sample_intent_steps(
            oracle_data["Category 2: Intent Triggered Guidelines"][intent_key],
            violation_data["Category 2: Intent Triggered Guidelines"][intent_key],
            max_violations=5
        )
    else:
        intent_steps = oracle_data["Category 2: Intent Triggered Guidelines"][intent_key]
        violated_idx = []

    # 3) Sample general guidelines and flip some to violated, 1..max_violation_num
    violation_num = random.randint(1, max_violation_num)
    must_follow_general, must_violate_general, presentation_general = sample_general_guidelines(
        oracle_data,
        violation_data,
        violation_num=violation_num
    )

    # 4) Build messages
    user_prompt = build_user_prompt(
        must_follow_general=must_follow_general,
        must_violate_general=must_violate_general,
        presentation_general=presentation_general,
        intent_key=intent_key,
        intent_steps=intent_steps,
        violated_step_idx=violated_idx
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 5) Call model (must return JSON per spec)
    response_text = call_chat_completion(model, messages)
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # If the model responded with text, wrap it for debugging
        data = {
            "meta": {"intent": intent_key, "parse_error": True},
            "raw_response": response_text
        }

    # 6) Attach planned violations & bookkeeping for audit
    data.setdefault("meta", {})
    data["meta"].update({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "intent": intent_key,
        # "global_must_violate": must_violate_general,
        "has_violated_intent_guidelines": "Yes" if has_violated_intent_guidelines else "No",
        "intent_step_violated_idx": violated_idx,
    })

    # Add "violations_planned" to meta in the format the system prompt allows
    planned = [{"category": x["category"], "guideline_key": x["guideline_key"], "phase_or_step": None}
               for x in must_violate_general]
    for idx in violated_idx:
        planned.append({
            "category": "Category 2: Intent Triggered Guidelines",
            "guideline_key": intent_key,
            "phase_or_step": f"step_index_{idx}"
        })

    return data

# ---------------------------
# Example runner (optional)
# ---------------------------

def save_sample(sample: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Example usage:
    sys_prompt = build_system_prompt()

    num_samples = 100
    for i in range(num_samples):
        sample = generate_conversation(
            oracle_data_path="guidelines/airlines/oracle.json",
            violation_data_path="guidelines/airlines/violation.json",
            system_prompt=sys_prompt,
            model="gpt-5",
            max_violation_num=3,
            seed=None
        )
        file_name = f"data/sample_conversation_{i+1}.json"
        save_sample(sample, file_name)
        print(f"Saved: {file_name}")
