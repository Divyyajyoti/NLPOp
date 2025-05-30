import streamlit as st
import openai
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize, linear_sum_assignment
from itertools import permutations
import logging
import json
import re
import os
from typing import Dict, List, Optional, Tuple
import json

OPENAI_API_KEY = "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"
OPENAI_PROJECT_ID = "proj_bb1YRYbP5wd4P8quGgIrXOsd"
OPENAI_ORG_ID = "org-adVrGVntY5ftjNinDvOf53Ku"

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    st.stop()

try:
    openai.api_key = OPENAI_API_KEY
    openai.organization = OPENAI_ORG_ID
    openai.project = OPENAI_PROJECT_ID
except Exception as e:
    st.error(f"Failed to initialize OpenAI: {e}")
    st.stop()

# --- Logging Setup ---
try:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='app.log',
        filemode='a'
    )
except PermissionError:
    st.error("Cannot write to app.log. Check directory permissions.")
    st.stop()

st.set_page_config(page_title="Optimizer & Finance Chatbot", layout="wide")

# --- Optimization Type Detection ---
OPT_TYPES = {
    "linear_programming": False,
    "integer_programming": False,
    "nonlinear_programming": False,
    "quadratic_programming": False,
    "convex_programming": False,
    "combinatorial_optimization": False,
    "dynamic_programming": False,
    "stochastic_programming": False,
    "multi_objective_optimization": False,
    "set_covering": False
}

token_count = []

# --- Initialize Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "session_problem" not in st.session_state:
    st.session_state.session_problem = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = ""
if "user_input_history" not in st.session_state:
    st.session_state.user_input_history = []
if "excel_handled" not in st.session_state:
    st.session_state.excel_handled = False

def detect_optimization_type(problem_statement: str) -> Dict[str, bool]:
    """Detect the type of optimization problem from the problem statement."""
    prompt = f"""
    You are an expert in optimization theory. Read the following optimization problem statement and determine which types of optimization problems it involves.

    ---
    {problem_statement}
    ---

    In addition to standard optimization categories, pay special attention to Set Covering Problems, even if not explicitly named.

    Set covering problems typically involve:
    - A universe of required elements (e.g., skills, cities).
    - A collection of sets covering parts of the universe.
    - Objective: select minimal sets to cover the universe.

    If detected, set 'set_covering' to True.

    For stochastic_programming, look for:
    - Uncertainty (random demand, costs, probabilities).
    - Scenarios, expected values, or chance constraints.

    If detected, set 'stochastic_programming' to True.

    Return a dictionary:
    {json.dumps(OPT_TYPES)}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            detected_types = json.loads(match.group(0))
            if re.search(r'[a-zA-Z]\^\d+|[a-zA-Z]+\*[a-zA-Z]+|sin\(|cos\(|exp\(|log\(', problem_statement):
                detected_types["nonlinear_programming"] = True
            if "set cover" in problem_statement.lower() or "set covering" in problem_statement.lower():
                detected_types["set_covering"] = True
            return detected_types
        return OPT_TYPES.copy()
    except Exception as e:
        logging.error(f"Error in detect_optimization_type: {e}")
        st.error(f"Optimization type detection failed: {e}")
        return OPT_TYPES.copy()

def humanize_response(technical_output: str, problem_type: str = "optimization") -> str:
    """Convert technical output to natural language."""
    prompt = f"""
    You are a formal mathematical assistant. The following is a technical {problem_type} solution:

    ---
    {technical_output}
    ---

    Rewrite in simple, conversational language, highlighting optimal value, key variables, and takeaways.
    Be brief and helpful.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        token_count.append(response.usage.total_tokens)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in humanize_response: {e}")
        return f"(⚠️ Could not humanize: {e})\n\n{technical_output}"

def simplify_math_expressions(text: str) -> str:
    """Simplify mathematical expressions (e.g., 2*3 -> 6)."""
    pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)\b')
    return pattern.sub(lambda m: str(float(m.group(1)) * float(m.group(2))), text)

def extract_lpp_from_text(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Extract Linear Programming Problem (LPP) components."""
    prompt = f"""
    You are an expert in optimization. Extract LPP components from the following text and return **only** a valid JSON dictionary with double-quoted property names, no markdown, no extra text, and no comments.

    ---
    {text}
    ---

    Stages:
    1. Convert to SI units for processing, revert to original units for output.
    2. Extract:
       - Objective coefficients (`c_max`, `c_min`)
       - Inequality constraints (`A_ub`, `b_ub`)
       - Equality constraints (`A_eq`, `b_eq`)
       - Bounds: list of (lower, upper) tuples, default (0, None)
       - Variable names
       - Constraint names
       - Objective: "maximize", "minimize", or "mixed"
    3. Verify matrices five times for consistency.

    Return a JSON dictionary exactly in this format:
    {{
        "c_max": [float, ...],
        "c_min": [float, ...],
        "A_ub": [[float, ...], ...],
        "b_ub": [float, ...],
        "A_eq": [[float, ...], ...] or null,
        "b_eq": [float, ...] or null,
        "bounds": [[float, float or null], ...],
        "objective": "maximize" or "minimize" or "mixed",
        "variable_names": ["x1", "x2", ...],
        "constraint_names": ["Constraint 1", ...]
    }}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        # Stricter regex to match a single JSON object, handling nested structures
        match = re.search(r'^\s*\{[\s\S]*?\}\s*$', response.choices[0].message.content, re.DOTALL)
        if match:
            raw_json = match.group(0)
            try:
                parsed_dict = json.loads(raw_json)
                if not isinstance(parsed_dict, dict):
                    logging.error(f"Parsed response is not a dictionary: {raw_json}")
                    return None, "Parsed response is not a valid JSON dictionary."
                # Validate required fields
                required_fields = ["c_max", "c_min", "A_ub", "b_ub", "objective"]
                missing_fields = [f for f in required_fields if f not in parsed_dict]
                if missing_fields:
                    logging.error(f"Missing required fields {missing_fields} in JSON: {raw_json}")
                    return None, f"Missing required fields: {', '.join(missing_fields)}"
                parsed_dict = ask_user_to_fill_missing_values(parsed_dict)
                st.session_state.history.append(("assistant", parsed_dict))
                return parsed_dict, None
            except json.JSONDecodeError as json_err:
                logging.error(f"JSON parsing error: {json_err}, Raw response: {response.choices[0].message.content}")
                return None, f"Failed to parse JSON response: {json_err}"
        else:
            logging.error(f"No valid JSON object found in response: {response.choices[0].message.content}")
            return None, "No valid JSON object found in API response."
    except Exception as e:
        logging.error(f"Error parsing LPP: {e}, Raw response: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
        return None, f"Error parsing LPP: {e}"


def extract_dynamic_programming_problem(problem_text: str) -> Optional[Dict]:
    """Extract Dynamic Programming components."""
    prompt = f"""
    Extract from:

    ---
    {problem_text}
    ---

    Return a JSON dictionary:
    {{
        "states": [...],
        "transitions": "...",
        "recurrence_relation": "...",
        "base_cases": {{...}},
        "goal": "maximize" or "minimize"
    }}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except Exception as e:
        logging.error(f"Error extracting DP: {e}")
        return {"error": f"Failed to extract DP: {e}"}

def solve_dp(dp_dict: Dict) -> Tuple[str, Optional[float]]:
    """Solve Dynamic Programming problem."""
    try:
        if "knapsack" in dp_dict.get("description", "").lower():
            return solve_knapsack(dp_dict['values'], dp_dict['weights'], dp_dict['capacity'])
        return "DP not implemented.", None
    except Exception as e:
        logging.error(f"Error solving DP: {e}")
        return f"Error solving DP: {e}", None

def ask_user_to_fill_missing_values(problem_dict: Dict) -> Dict:
    """Prompt user to fill missing values."""
    logging.info("Checking for missing values.")
    updated = False
    missing_fields = []

    for key, value in problem_dict.items():
        if value is None or (isinstance(value, list) and not value):
            missing_fields.append(key)
            logging.debug(f"Missing field: {key}")

    for field in missing_fields:
        prompt = f"What value for '{field}'?"
        try:
            if field in ["A_eq", "A_ub"]:
                num_vars = len(problem_dict.get("c_max") or problem_dict.get("c_min") or [])
                num_constraints = st.number_input(f"How many {field} constraints?", min_value=1, value=1, key=f"num_{field}")
                matrix = []
                for i in range(int(num_constraints)):
                    row = []
                    for j in range(num_vars):
                        var_name = problem_dict.get("variable_names", [f"x{k+1}" for k in range(num_vars)])[j]
                        val = st.number_input(
                            f"Coefficient for {var_name} in {field} constraint {i+1}",
                            value=0.0,
                            key=f"{field}_{i}_{j}"
                        )
                        row.append(float(val))
                    matrix.append(row)
                problem_dict[field] = matrix
                updated = True
                logging.info(f"Updated '{field}': {matrix}")
            elif field in ["b_eq", "b_ub"]:
                num_constraints = len(problem_dict.get("A_eq" if field == "b_eq" else "A_ub", []))
                values = []
                for i in range(num_constraints):
                    val = st.number_input(
                        f"RHS for {'equality' if field == 'b_eq' else 'inequality'} constraint {i+1}",
                        value=0.0,
                        key=f"{field}_{i}"
                    )
                    values.append(float(val))
                problem_dict[field] = values
                updated = True
                logging.info(f"Updated '{field}': {values}")
            else:
                user_input = st.text_input(prompt, key=f"missing_{field}")
                if user_input:
                    try:
                        problem_dict[field] = json.loads(user_input)
                    except:
                        problem_dict[field] = user_input
                    updated = True
                    logging.info(f"Updated '{field}': {user_input}")
        except Exception as e:
            logging.error(f"Error processing '{field}': {e}")

    if updated:
        st.success(humanize_response("Missing values updated."))
        logging.info("Humanized response displayed.")

    return problem_dict

def solve_lpp(lpp_dict: Dict, alpha: float = 0.5) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
    """Solve Linear Programming Problem."""
    c_max = lpp_dict.get('c_max')
    c_min = lpp_dict.get('c_min')
    A_ub = lpp_dict.get('A_ub')
    b_ub = lpp_dict.get('b_ub')
    A_eq = lpp_dict.get('A_eq')
    b_eq = lpp_dict.get('b_eq')
    bounds = lpp_dict.get('bounds')
    objective = lpp_dict.get('objective')

    num_vars = len(c_max) if c_max else len(c_min) if c_min else 0
    if not bounds or len(bounds) != num_vars:
        bounds = [(0, None) for _ in range(num_vars)]

    if objective == 'maximize' and c_max:
        c = [-val for val in c_max]
    elif objective == 'minimize' and c_min:
        c = c_min
    elif objective == 'mixed' and c_max and c_min:
        c = [(alpha * -x) + ((1 - alpha) * y) for x, y in zip(c_max, c_min)]
    else:
        return "Invalid objective or coefficients.", None, None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            if objective == 'maximize':
                res.fun = -res.fun
            return None, res.fun, res.x
        return f"LPP failed: {res.message}", None, None
    except ValueError as e:
        logging.error(f"Error solving LPP: {e}")
        return f"Error solving LPP: {e}", None, None

def display_constraints(lpp_dict: Dict) -> str:
    """Format LPP constraints."""
    details = []
    var_names = lpp_dict.get('variable_names') or [f"x{i+1}" for i in range(len(lpp_dict.get('c_max', lpp_dict.get('c_min', []))))]
    con_names = lpp_dict.get('constraint_names', [])

    if lpp_dict.get('A_ub'):
        for idx, (row, b) in enumerate(zip(lpp_dict['A_ub'], lpp_dict['b_ub'])):
            constraint = " + ".join(f"{round(coef, 2)}{var_names[i]}" for i, coef in enumerate(row) if coef != 0)
            name = con_names[idx] if idx < len(con_names) else f"Constraint {idx+1}"
            details.append(f"{name}: {constraint} <= {b}")
    if lpp_dict.get('A_eq'):
        start_idx = len(lpp_dict.get('A_ub') or [])
        for i, (row, b) in enumerate(zip(lpp_dict['A_eq'], lpp_dict['b_eq'])):
            constraint = " + ".join(f"{round(coef, 2)}{var_names[j]}" for j, coef in enumerate(row) if coef != 0)
            idx = start_idx + i
            name = con_names[idx] if idx < len(con_names) else f"Constraint {idx+1}"
            details.append(f"{name}: {constraint} = {b}")
    return "\n".join(details) or "No constraints."

def format_solution(opt_val: float, opt_vars: np.ndarray, objective: str, lpp_dict: Dict) -> str:
    """Format LPP solution."""
    if opt_val is None or opt_vars is None:
        return "No feasible solution."

    var_names = lpp_dict.get('variable_names') or [f"x{i+1}" for i in range(len(opt_vars))]
    var_details = "\n".join([f"  - {name}: {val:.2f}" for name, val in zip(var_names, opt_vars)])

    summary = ""
    if objective == 'maximize' and lpp_dict.get('c_max'):
        terms = [f"{round(coef, 2)}x{i+1}" for i, coef in enumerate(lpp_dict['c_max']) if coef != 0]
        summary += "**Objective:** Maximize Z = " + " + ".join(terms) + "\n"
    elif objective == 'minimize' and lpp_dict.get('c_min'):
        terms = [f"{round(coef, 2)}x{i+1}" for i, coef in enumerate(lpp_dict['c_min']) if coef != 0]
        summary += "**Objective:** Minimize Z = " + " + ".join(terms) + "\n"
    elif objective == 'mixed' and lpp_dict.get('c_max') and lpp_dict.get('c_min'):
        terms = [f"(α*-{x} + (1-α)*{y})x{i+1}" for i, (x, y) in enumerate(zip(lpp_dict['c_max'], lpp_dict['c_min'])) if x != 0 or y != 0]
        summary += "**Objective:** Mixed = " + " + ".join(terms) + "\n"

    summary += "\n**Constraints:**\n" + display_constraints(lpp_dict) + "\n\n"
    result_text = f"Optimal Value: **{opt_val:.2f}**\n\nVariables:\n{var_details}"
    return summary + result_text

def modify_lpp(session_problem: Dict, user_input: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Modify an existing LPP."""
    prompt = f"""
    Modify LPP:

    {json.dumps(session_problem, indent=2)}

    Based on:
    "{user_input}"

    Update only necessary fields.

    Return a JSON dictionary.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        return None, "Failed to parse modified LPP."
    except Exception as e:
        logging.error(f"Error parsing modified LPP: {e}")
        return None, f"Error parsing modified LPP: {e}"

def extract_nlp_components(problem_statement: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Extract Non-Linear Programming (NLP) components."""
    prompt = f"""
    Extract NLP components:

    ---
    {problem_statement}
    ---

    Return a JSON dictionary:
    {{
        "objective_function": "lambda x: ...",
        "objective_type": "minimize" or "maximize",
        "constraints": [
            {{'type': 'eq' or 'ineq', 'fun': "lambda x: ..."}}
        ],
        "initial_guess": [float, ...],
        "variable_names": ["x1", ...],
        "constraint_descriptions": ["desc1", ...]
    }}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        return None, "Invalid NLP format."
    except Exception as e:
        logging.error(f"Error parsing NLP: {e}")
        return None, f"Error parsing NLP: {e}"

def solve_nlp(nlp_dict: Dict) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
    """Solve Non-Linear Programming problem."""
    objective_func_str = nlp_dict.get('objective_function')
    objective_type = nlp_dict.get('objective_type', 'minimize')
    constraints_data = nlp_dict.get('constraints', [])
    initial_guess = nlp_dict.get('initial_guess')

    if not objective_func_str or not initial_guess:
        return "Objective or initial guess missing.", None, None

    try:
        objective_function = eval(objective_func_str, {"__builtins__": {}}, {"np": np})
    except Exception as e:
        logging.error(f"Error evaluating objective: {e}")
        return f"Error evaluating objective: {e}", None, None

    formatted_constraints = []
    if constraints_data:
        for con in constraints_data:
            try:
                func = eval(con['fun'], {"__builtins__": {}}, {"np": np})
                formatted_constraints.append({'type': con['type'], 'fun': func})
            except Exception as e:
                logging.error(f"Error evaluating constraint: {e}")
                return f"Error evaluating constraint: {e}", None, None

    if objective_type == 'maximize':
        def negative_objective(x):
            return -objective_function(x)
        solver_objective = negative_objective
    else:
        solver_objective = objective_function

    try:
        result = minimize(solver_objective, initial_guess, constraints=formatted_constraints, method='SLSQP')
        if result.success:
            optimal_value = result.fun if objective_type == 'minimize' else -result.fun
            return None, optimal_value, result.x
        return f"NLP failed: {result.message}", None, None
    except Exception as e:
        logging.error(f"Error solving NLP: {e}")
        return f"Error solving NLP: {e}", None, None

def format_nlp_solution(opt_val: float, opt_vars: np.ndarray, objective: str, nlp_dict: Dict) -> str:
    """Format NLP solution."""
    if opt_val is None or opt_vars is None:
        return "No feasible solution."

    var_names = nlp_dict.get('variable_names') or [f"x{i+1}" for i in range(len(opt_vars))]
    var_details = "\n".join([f"  - {name}: {val:.4f}" for name, val in zip(var_names, opt_vars)])
    objective_str = nlp_dict.get('objective_function_original', nlp_dict.get('objective_function', "Objective"))

    summary = f"**Objective:** {objective_str} ({objective})\n\n"
    if nlp_dict.get('constraint_descriptions'):
        summary += "**Constraints:**\n" + "\n".join([f"- {desc}" for desc in nlp_dict['constraint_descriptions']]) + "\n\n"

    result_text = f"Optimal Value: **{opt_val:.4f}**\n\nVariables:\n{var_details}"
    return summary + result_text

def modify_nlp(session_nlp: Dict, user_input: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Modify an existing NLP."""
    prompt = f"""
    Modify NLP:

    {json.dumps(session_nlp, indent=2)}

    Based on:
    "{user_input}"

    Update only necessary fields. Keep lambda functions.

    Return a JSON dictionary.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        return None, "Failed to parse modified NLP."
    except Exception as e:
        logging.error(f"Error parsing modified NLP: {e}")
        return None, f"Error parsing modified NLP: {e}"

def solve_knapsack(values: List[float], weights: List[float], capacity: float, item_names: Optional[List[str]] = None) -> Tuple[int, List[str]]:
    """Solve Knapsack problem."""
    if not isinstance(capacity, (int, float)) or capacity != int(capacity):
        raise ValueError("Capacity must be integer.")
    if not all(float(v).is_integer() for v in values):
        raise ValueError("Values must be integers.")
    if not all(float(w).is_integer() for w in weights):
        raise ValueError("Weights must be integers.")

    values = [int(v) for v in values]
    weights = [int(w) for w in weights]
    capacity = int(capacity)
    n = len(values)

    if item_names is None:
        item_names = [f"Item {i+1}" for i in range(n)]
    elif len(item_names) != n:
        raise ValueError("Item names length mismatch.")

    dp = np.zeros((n + 1, capacity + 1), dtype=int)
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    optimal_value = int(dp[n][capacity])
    chosen_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen_items.append(item_names[i - 1])
            w -= weights[i - 1]

    return optimal_value, chosen_items[::-1]

def display_knapsack_solution(values: List[float], weights: List[float], item_names: List[str], chosen_items: List[str]):
    """Display Knapsack solution."""
    data = [{"Item": name, "Value": val, "Weight": wt, "Selected": "✅ Yes" if name in chosen_items else "❌ No"}
            for name, val, wt in zip(item_names, values, weights)]
    df = pd.DataFrame(data)
    st.subheader("📋 Knapsack Table")
    st.dataframe(df, use_container_width=True)

def solve_tsp(distance_matrix: np.ndarray) -> Tuple[Optional[float], Optional[List[int]]]:
    """Solve Traveling Salesman Problem."""
    n = len(distance_matrix)
    if len(distance_matrix) != len(distance_matrix[0]):
        return "Error: Non-square matrix.", None
    if n == 0:
        return "Error: Empty matrix.", None
    if n == 1:
        return 0, [0]
    if n > 10:
        return "Warning: TSP too large.", None

    cities = list(range(n))
    min_distance = float('inf')
    best_path = None

    for perm in permutations(cities):
        current_distance = 0
        valid_path = True
        for i in range(n):
            current_city = perm[i]
            next_city = perm[(i + 1) % n]
            distance = distance_matrix[current_city][next_city]
            if np.isinf(distance):
                valid_path = False
                break
            current_distance += distance

        if valid_path and current_distance < min_distance:
            min_distance = current_distance
            best_path = perm

    if min_distance == float('inf'):
        return "Error: No valid tour.", None

    return min_distance, list(best_path)

def solve_set_covering(sets: List[List], universe: List) -> Tuple[Optional[List[int]], Optional[int]]:
    """Solve Set Covering problem."""
    uncovered = set(universe)
    cover = []
    set_indices = list(range(len(sets)))

    while uncovered:
        best_set_index = -1
        max_elements_covered = 0

        for i, s in enumerate(sets):
            elements_covered = len(set(s) & uncovered)
            if elements_covered > max_elements_covered:
                max_elements_covered = elements_covered
                best_set_index = i

        if best_set_index == -1:
            return "Error: Cannot cover all.", None

        cover.append(best_set_index)
        uncovered -= set(sets[best_set_index])

    return cover, len(cover)

def solve_assignment_problem(cost_matrix: np.ndarray) -> Tuple[Optional[float], Optional[List[Tuple[int, int]]]]:
    """Solve Assignment problem."""
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        optimal_cost = cost_matrix[row_ind, col_ind].sum()
        assignment = list(zip(row_ind, col_ind))
        return optimal_cost, assignment
    except Exception as e:
        logging.error(f"Error solving assignment: {e}")
        return f"Error solving assignment: {e}", None

def format_tsp_solution(distance: float, path: List[int]) -> str:
    """Format TSP solution."""
    path_str = " -> ".join(map(str, path))
    return f"Optimal Path: {path_str}\nDistance: {distance:.2f}"

def format_assignment_solution(cost: float, assignment: List[Tuple[int, int]]) -> str:
    """Format Assignment solution."""
    assignment_str = "\n".join([f"Worker {w} to Job {j}" for w, j in assignment])
    return f"Optimal Assignment:\n{assignment_str}\nCost: {cost:.2f}"

def format_set_covering_solution(cover_indices: List[int], cost: int, sets: List[List]) -> str:
    """Format Set Covering solution."""
    if isinstance(cover_indices, str):
        return cover_indices
    covered_sets = [f"Set {i + 1}: {sets[i]}" for i in cover_indices]
    sets_used = ", ".join(map(str, [i + 1 for i in cover_indices]))
    return f"Sets: {sets_used}\nCovered:\n" + "\n".join(covered_sets) + f"\nTotal: {cost}"

def extract_combinatorial_data(problem_statement: str) -> Dict:
    """Extract combinatorial optimization data."""
    prompt = f"""
    Identify combinatorial problem:

    ---
    {problem_statement}
    ---

    Return a JSON dictionary:
    {{
        "problem_type": "tsp" or "assignment" or "knapsack" or "set_covering" or "unknown",
        "data": {{
            "distance_matrix": [[...], ...],
            "cost_matrix": [[...], ...],
            "values": [...], "weights": [...], "capacity": ...,
            "item_names": [...],
            "sets": [[...], ...],
            "universe": [...]
        }},
        "error": null
    }}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"problem_type": "unknown", "data": {}, "error": "Failed to parse."}
    except Exception as e:
        logging.error(f"Error extracting combinatorial data: {e}")
        return {"problem_type": "unknown", "data": {}, "error": f"Error: {e}"}

def modify_combinatorial(session_com: Dict, user_input: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Modify combinatorial problem."""
    prompt = f"""
    Modify combinatorial problem:

    {json.dumps(session_com, indent=2)}

    Based on:
    "{user_input}"

    Update only necessary fields.

    Return a JSON dictionary.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        return None, "Failed to parse."
    except Exception as e:
        logging.error(f"Error modifying combinatorial: {e}")
        return None, f"Error: {e}"

def extract_stochastic_components(problem_text: str) -> Dict:
    """Extract stochastic programming components."""
    prompt = f"""
    Extract stochastic components:

    ---
    {problem_text}
    ---

    Return a JSON dictionary:
    {{
      "first_stage_variables": ["x1", ...],
      "second_stage_variables": ["y1", ...],
      "scenarios": [
        {{
          "name": "...",
          "probability": float,
          "cost_coefficients": [float, ...],
          "constraints": ["...", ...]
        }},
        ...
      ],
      "objective": "maximize" or "minimize"
    }}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"error": "Failed to parse."}
    except Exception as e:
        logging.error(f"Error extracting stochastic components: {e}")
        return {"error": f"Exception: {e}"}

def solve_two_stage_stochastic(stochastic_data: Dict) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """Solve two-stage stochastic problem."""
    try:
        first_stage_vars = stochastic_data["first_stage_variables"]
        second_stage_vars = stochastic_data["second_stage_variables"]
        scenarios = stochastic_data["scenarios"]
        objective = stochastic_data["objective"]

        num_x = len(first_stage_vars)
        num_y = len(second_stage_vars)
        num_scenarios = len(scenarios)

        total_vars = num_x + (num_y * num_scenarios)

        def expected_total_value(z):
            x = z[:num_x]
            total = 0.0
            for i, scen in enumerate(scenarios):
                prob = scen["probability"]
                cost_coeffs = scen["cost_coefficients"]
                y_s = z[num_x + i*num_y : num_x + (i+1)*num_y]
                scen_cost = sum(c * xi for c, xi in zip(cost_coeffs[:len(x)], x)) + \
                            sum(c * yi for c, yi in zip(cost_coeffs[len(x):], y_s))
                total += prob * scen_cost
            return -total if objective == "maximize" else total

        bounds = [(0, None) for _ in range(total_vars)]
        result = minimize(expected_total_value, [1.0]*total_vars, bounds=bounds, method='SLSQP')

        if result.success:
            optimal_value = -result.fun if objective == "maximize" else result.fun
            optimal_x = result.x[:num_x]
            optimal_ys = [result.x[num_x + i*num_y : num_x + (i+1)*num_y] for i in range(num_scenarios)]
            return None, optimal_value, optimal_x, optimal_ys
        return f"Stochastic failed: {result.message}", None, None, None
    except Exception as e:
        logging.error(f"Error solving stochastic: {e}")
        return f"Error solving stochastic: {e}", None, None, None

def classify_user_input(user_input: str, session_problem: Optional[Dict] = None) -> str:
    """Classify user input as new or follow-up."""
    prompt = f"""
    Classify user input:

    Current problem:
    ```{json.dumps(session_problem, indent=2) if session_problem else 'None'}```

    User input:
    ```{user_input}```

    Return:
    "new" or "followup"
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        answer = response.choices[0].message.content.strip().lower()
        if "new" in answer:
            return ""
        elif "followup" in answer:
            return st.session_state.problem_type
        return "unknown"
    except Exception as e:
        logging.error(f"Error classifying input: {e}")
        return f"error: {e}"

def contains_compound_interest_terms(text: str) -> bool:
    """Check for compound interest terms."""
    interest_keywords = ["interest", "returns", "compounded", "growth", "rate"]
    return any(keyword in text.lower() for keyword in interest_keywords)

def extract_growth_coefficients_with_llm(text: str) -> List[Dict]:
    """Extract compound growth coefficients."""
    prompt = f"""
    Extract compound growth coefficients:

    ---
    {text}
    ---

    Return a JSON list:
    [
        {{"name": "asset", "rate": float, "years": float}},
        ...
    ]
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        match = re.search(r'\[.*\]', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except Exception as e:
        logging.error(f"Error extracting growth coefficients: {e}")
        return []

def inject_compound_growth_coefficients(text: str) -> str:
    """Inject compound growth coefficients."""
    if not contains_compound_interest_terms(text):
        return text

    growth_entries = extract_growth_coefficients_with_llm(text)
    growth_map = {}
    for entry in growth_entries:
        name = entry['name'].lower().replace(" ", "_")
        rate = entry['rate']
        years = entry['years']
        growth_factor = round((1 + rate) ** years, 4)
        growth_map[name] = growth_factor

    summary = "\n\n# Growth Coefficients:\n"
    for asset, factor in growth_map.items():
        summary += f"{asset} => {factor}\n"
    return text + summary

def check_problem_completeness(problem_type: str, session_problem: Optional[Dict], user_input: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Check if problem is complete."""
    problem_type_description = str(problem_type).replace('_', ' ') if problem_type else "unknown"
    prompt = f"""
    Check completeness for {problem_type_description}:

    Problem:
    {json.dumps(session_problem, indent=2) if session_problem else 'None'}

    Input:
    "{user_input}"

    Return a JSON dictionary:
    {{
        "is_complete": true or false,
        "missing_fields": ["field1", ...]
    }}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        token_count.append(response.usage.total_tokens)
        # Improved regex to match JSON object, accounting for whitespace and potential surrounding text
        match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            raw_json = match.group(0)
            try:
                completeness_check = json.loads(raw_json)
                if not isinstance(completeness_check, dict):
                    logging.error(f"Parsed response is not a dictionary: {raw_json}")
                    return None, "Parsed response is not a valid JSON dictionary."
                if not completeness_check.get("is_complete", False):
                    missing_fields = completeness_check.get("missing_fields", [])
                    error_message = f"Missing: {', '.join(missing_fields)}"
                    return completeness_check, error_message
                return completeness_check, None
            except json.JSONDecodeError as json_err:
                logging.error(f"JSON parsing error: {json_err}, Raw response: {response.choices[0].message.content}")
                return None, f"Failed to parse JSON response: {json_err}"
        else:
            logging.error(f"No JSON found in response: {response.choices[0].message.content}")
            return None, "No valid JSON found in API response."
    except Exception as e:
        logging.error(f"Error checking completeness: {e}, Raw response: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
        return None, f"Error checking completeness: {e}"

def handle_missing_information(problem_type: str, session_problem: Optional[Dict], user_input: str):
    """Handle missing information."""
    st.session_state.history.append(("user", user_input))
    completeness_check, error_message = check_problem_completeness(problem_type, session_problem, user_input)
    if completeness_check is None:
        st.session_state.history.append(("assistant", f"❌ Failed to check completeness: {error_message}"))
        return
    if error_message:
        st.session_state.history.append(("assistant", f"⚠️ {error_message}"))
        missing_fields = completeness_check.get("missing_fields", [])
        questions = [f"What value for '{field}'?" for field in missing_fields]
        st.session_state.history.append(("assistant", humanize_response("\n".join(questions))))


# --- Streamlit App ---
def run_app():
    st.title("🔢 Optimizer")
    st.markdown("Describe an optimization problem ")

    # --- Optimization Interface ---
    st.subheader("📄 Optimization Input")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

    if st.session_state.get("excel_handled"):
        if st.button("🔄 Clear Excel"):
            st.session_state.pop('excel_handled', None)
            st.session_state.session_problem = None
            st.session_state.problem_type = ""
            st.session_state.user_input_history = []
            st.success("Excel data cleared.")
            st.rerun()  # Rerun to update UI after clearing

    if uploaded_file and not st.session_state.excel_handled:
        try:
            df_dict = pd.read_excel(uploaded_file, sheet_name=None)
            st.success("✅ Excel uploaded.")

            def describe_excel_for_ai(df_dict):
                content = ""
                for sheet_name, df in df_dict.items():
                    content += f"Sheet: {sheet_name}\n{df.to_string(index=False)}\n\n"
                return content

            excel_text = describe_excel_for_ai(df_dict)
            opt_types = detect_optimization_type(excel_text)
            user_input = inject_compound_growth_coefficients(excel_text)

            if opt_types.get("linear_programming"):
                lpp_data, error = extract_lpp_from_text(user_input)
                if error:
                    st.error(f"❌ {error}")
                else:
                    err2, opt_val, opt_vars = solve_lpp(lpp_data)
                    if err2:
                        st.error(f"❌ {err2}")
                    else:
                        formatted = format_solution(opt_val, opt_vars, lpp_data.get("objective"), lpp_data)
                        human_response = humanize_response(formatted)
                        st.session_state.session_problem = lpp_data
                        st.session_state.problem_type = "linear"
                        st.session_state.history.append(("assistant", "📋 **Excel: Linear Programming**"))
                        st.session_state.history.append(("assistant", human_response))
                        st.session_state.excel_handled = True
                        st.rerun()  # Rerun to display response immediately

            elif opt_types.get("nonlinear_programming"):
                nlp_data, error = extract_nlp_components(user_input)
                if error:
                    st.error(f"❌ {error}")
                else:
                    err2, opt_val, opt_vars = solve_nlp(nlp_data)
                    if err2:
                        st.error(f"❌ {err2}")
                    else:
                        nlp_data['objective_function_original'] = \
                        re.search(r"lambda x: (.+)", nlp_data.get('objective_function', ''))[1] if nlp_data.get(
                            'objective_function') else "Objective"
                        formatted = format_nlp_solution(opt_val, opt_vars, nlp_data.get("objective_type"), nlp_data)
                        human_response = humanize_response(formatted, "nonlinear")
                        st.session_state.session_problem = nlp_data
                        st.session_state.problem_type = "nonlinear"
                        st.session_state.history.append(("assistant", "📊 **Excel: Nonlinear Programming**"))
                        st.session_state.history.append(("assistant", human_response))
                        st.session_state.excel_handled = True
                        st.rerun()  # Rerun to display response immediately

            elif opt_types.get("stochastic_programming"):
                stochastic_dict = extract_stochastic_components(user_input)
                if "error" in stochastic_dict:
                    st.error(f"❌ {stochastic_dict['error']}")
                else:
                    err2, optimal_value, optimal_x, optimal_ys = solve_two_stage_stochastic(stochastic_dict)
                    if err2:
                        st.error(f"❌ {err2}")
                    else:
                        var_names = stochastic_dict["first_stage_variables"]
                        second_var_names = stochastic_dict["second_stage_variables"]
                        num_scenarios = len(stochastic_dict["scenarios"])
                        var_details = "\n".join([f"  - {name}: {val:.2f}" for name, val in zip(var_names, optimal_x)])
                        second_stage_details = ""
                        for i in range(num_scenarios):
                            scen_name = stochastic_dict["scenarios"][i]["name"]
                            second_decisions = optimal_ys[i]
                            decisions_text = ", ".join(
                                [f"{var}: {val:.2f}" for var, val in zip(second_var_names, second_decisions)])
                            second_stage_details += f"\n**Scenario {i + 1} ({scen_name}):** {decisions_text}"
                        summary = f"**Optimal Value:** {optimal_value:.2f}\n\n**First-Stage:**\n{var_details}\n\n**Second-Stage:**{second_stage_details}"
                        human_response = humanize_response(summary, "stochastic")
                        st.session_state.session_problem = stochastic_dict
                        st.session_state.problem_type = "stochastic_programming"
                        st.session_state.history.append(("assistant", "🎲 **Excel: Stochastic**"))
                        st.session_state.history.append(("assistant", human_response))
                        st.session_state.excel_handled = True
                        st.rerun()  # Rerun to display response immediately

            elif opt_types.get("combinatorial_optimization"):
                combinatorial_data = extract_combinatorial_data(user_input)
                if combinatorial_data["problem_type"] == "tsp":
                    st.session_state.problem_type = "combinatorial_tsp"
                    distance_matrix = np.array(combinatorial_data["data"].get("distance_matrix", []))
                    result = solve_tsp(distance_matrix)
                    if isinstance(result[0], str):
                        st.error(f"❌ {result[0]}")
                    else:
                        distance, path = result
                        formatted = format_tsp_solution(distance, path)
                        human_response = humanize_response(formatted, "TSP")
                        st.session_state.session_problem = combinatorial_data
                        st.session_state.history.append(("assistant", "🗺️ **Excel: TSP**"))
                        st.session_state.history.append(("assistant", human_response))
                        st.session_state.excel_handled = True
                        st.rerun()  # Rerun to display response immediately

                elif combinatorial_data["problem_type"] == "assignment":
                    st.session_state.problem_type = "combinatorial_assignment"
                    cost_matrix = np.array(combinatorial_data["data"].get("cost_matrix", []))
                    result = solve_assignment_problem(cost_matrix)
                    if isinstance(result[0], str):
                        st.error(f"❌ {result[0]}")
                    else:
                        cost, assignment = result
                        formatted = format_assignment_solution(cost, assignment)
                        human_response = humanize_response(formatted, "assignment")
                        st.session_state.session_problem = combinatorial_data
                        st.session_state.history.append(("assistant", "👷 **Excel: Assignment**"))
                        st.session_state.history.append(("assistant", human_response))
                        st.session_state.excel_handled = True
                        st.rerun()  # Rerun to display response immediately

                elif combinatorial_data["problem_type"] == "knapsack":
                    st.session_state.problem_type = "combinatorial_knapsack"
                    values = combinatorial_data["data"].get("values", [])
                    weights = combinatorial_data["data"].get("weights", [])
                    capacity = combinatorial_data["data"].get("capacity")
                    item_names = combinatorial_data["data"].get("item_names",
                                                                [f"Item {i + 1}" for i in range(len(values))])
                    optimal_value, chosen_items = solve_knapsack(values, weights, capacity, item_names)
                    formatted = f"Optimal Value: {optimal_value}\nItems: {chosen_items}"
                    human_response = humanize_response(formatted, "knapsack")
                    st.session_state.session_problem = combinatorial_data
                    st.session_state.history.append(("assistant", "🎒 **Knapsack**"))
                    st.session_state.history.append(("assistant", human_response))
                    st.success(f"🎯 Value: {optimal_value}")
                    display_knapsack_solution(values, weights, item_names, chosen_items)
                    st.session_state.excel_handled = True
                    st.rerun()  # Rerun to display response immediately

                elif combinatorial_data["problem_type"] == "set_covering":
                    st.session_state.problem_type = "set_covering"
                    sets = combinatorial_data["data"].get("sets", [])
                    universe = combinatorial_data["data"].get("universe", [])
                    result = solve_set_covering(sets, universe)
                    if isinstance(result[0], str):
                        st.error(f"❌ {result[0]}")
                    else:
                        cover_indices, cost = result
                        formatted = format_set_covering_solution(cover_indices, cost, sets)
                        human_response = humanize_response(formatted, "set covering")
                        st.session_state.session_problem = combinatorial_data
                        st.session_state.history.append(("assistant", "🛡️ **Set Covering**"))
                        st.session_state.history.append(("assistant", human_response))
                        st.session_state.excel_handled = True
                        st.rerun()  # Rerun to display response immediately

                else:
                    st.warning(f"⚠️ Unrecognized: {combinatorial_data['error']}")
                    st.session_state.excel_handled = True
                    st.rerun()  # Rerun to display warning

        except Exception as e:
            logging.error(f"Error processing Excel: {e}")
            st.error(f"❌ Error processing Excel: {e}")
            st.session_state.excel_handled = True
            st.rerun()  # Rerun to display error

    user_input = st.chat_input("Enter optimization problem or modification...")
    if user_input:
        with st.spinner("Processing..."):
            st.session_state.problem_type = classify_user_input(user_input, st.session_state.session_problem)
            st.session_state.user_input_history.append(user_input)
            latest_input = "\n".join(st.session_state.user_input_history)

            if st.session_state.problem_type == "":
                opt_types = detect_optimization_type(latest_input)
                if not any(opt_types.values()):
                    handle_missing_information("", st.session_state.session_problem, latest_input)
                    st.rerun()  # Rerun to display missing information prompt
                else:
                    st.session_state.problem_type = next(k for k, v in opt_types.items() if v)
                    completeness_check, error = check_problem_completeness(st.session_state.problem_type,
                                                                           st.session_state.session_problem,
                                                                           latest_input)
                    if error:
                        handle_missing_information(st.session_state.problem_type, st.session_state.session_problem,
                                                   latest_input)
                        st.rerun()  # Rerun to display missing information prompt
                    else:
                        st.success("✅ Problem complete!")
                        process_new_problem(latest_input, user_input, opt_types)
                        st.rerun()  # Rerun to display response immediately

            else:
                process_followup(latest_input, user_input, st.session_state.session_problem)
                st.rerun()  # Rerun to display follow-up response immediately


def process_new_problem(latest_input: str, user_input: str, opt_types: Dict[str, bool]):
    """Handle new optimization problems."""
    st.session_state.history.append(("user", user_input))
    if opt_types.get("linear_programming"):
        st.session_state.problem_type = "linear"
        user_input = inject_compound_growth_coefficients(latest_input)
        lpp_data, error = extract_lpp_from_text(user_input)
        if error:
            st.session_state.history.append(("assistant", f"❌ {error}"))
            st.rerun()  # Rerun to display error
        else:
            err2, opt_val, opt_vars = solve_lpp(lpp_data)
            if err2:
                st.session_state.history.append(("assistant", f"❌ {err2}"))
                st.rerun()  # Rerun to display error
            else:
                formatted = format_solution(opt_val, opt_vars, lpp_data.get("objective"), lpp_data)
                human_response = humanize_response(formatted)
                st.session_state.session_problem = lpp_data
                st.session_state.history.append(("assistant", human_response))
                true_types = [k.replace("_", " ").title() for k, v in opt_types.items() if v]
                st.session_state.history.append(("assistant", f"📚 **Types:** {', '.join(true_types)}"))
                st.rerun()  # Rerun to display response

    elif opt_types.get("nonlinear_programming"):
        st.session_state.problem_type = "nonlinear"
        nlp_data, error = extract_nlp_components(latest_input)
        if error:
            st.session_state.history.append(("assistant", f"❌ {error}"))
            st.rerun()  # Rerun to display error
        else:
            err2, opt_val, opt_vars = solve_nlp(nlp_data)
            if err2:
                st.session_state.history.append(("assistant", f"❌ {err2}"))
                st.rerun()  # Rerun to display error
            else:
                nlp_data['objective_function_original'] = re.search(r"lambda x: (.+)", nlp_data.get('objective_function', ''))[1] if nlp_data.get('objective_function') else "Objective"
                formatted = format_nlp_solution(opt_val, opt_vars, nlp_data.get("objective_type"), nlp_data)
                human_response = humanize_response(formatted, "nonlinear")
                st.session_state.session_problem = nlp_data
                st.session_state.history.append(("assistant", human_response))
                true_types = [k.replace('_', ' ').title() for k, v in opt_types.items() if v]
                st.session_state.history.append(("assistant", f"📚 Types: {', '.join(true_types)}"))
                st.rerun()  # Rerun to display response

    elif opt_types.get("combinatorial_optimization"):
        combinatorial_data = extract_combinatorial_data(latest_input)
        st.session_state.session_problem = combinatorial_data
        if combinatorial_data["problem_type"] == "tsp":
            st.session_state.problem_type = "tsp"
            distance_matrix = np.array(combinatorial_data["data"].get("distance_matrix", []))
            result = solve_tsp(distance_matrix)
            if isinstance(result[0], str):
                st.session_state.history.append(("assistant", f"❌ {result[0]}"))
                st.rerun()  # Rerun to display error
            else:
                distance, path = result
                formatted = format_tsp_solution(distance, path)
                human_response = humanize_response(formatted, "TSP")
                st.session_state.history.append(("assistant", human_response))
                st.rerun()  # Rerun to display response

        elif combinatorial_data["problem_type"] == "assignment":
            st.session_state.problem_type = "assignment"
            cost_matrix = np.array(combinatorial_data["data"].get("cost_matrix", []))
            result = solve_assignment_problem(cost_matrix)
            if isinstance(result[0], str):
                st.session_state.history.append(("assistant", f"❌ {result[0]}"))
                st.rerun()  # Rerun to display error
            else:
                cost, assignment = result
                formatted = format_assignment_solution(cost, assignment)
                human_response = humanize_response(formatted, "assignment")
                st.session_state.history.append(("assistant", human_response))
                st.rerun()  # Rerun to display response

        elif combinatorial_data["problem_type"] == "knapsack":
            st.session_state.problem_type = "knapsack"
            values = combinatorial_data["data"].get("values", [])
            weights = combinatorial_data["data"].get("weights", [])
            capacity = combinatorial_data["data"].get("capacity")
            item_names = combinatorial_data["data"].get("item_names", [f"Item {i+1}" for i in range(len(values))])
            try:
                optimal_value, chosen_items = solve_knapsack(values, weights, capacity, item_names)
                formatted = f"Value: {optimal_value}\nItems: {chosen_items}"
                human_response = humanize_response(formatted, "knapsack")
                st.session_state.history.append(("assistant", human_response))
                st.success(f"Value: {optimal_value}")
                display_knapsack_solution(values, weights, item_names, chosen_items)
                st.rerun()  # Rerun to display response
            except Exception as e:
                logging.error(f"Knapsack error: {e}")
                st.session_state.history.append(("assistant", f"❌ Knapsack error: {e}"))
                st.rerun()  # Rerun to display error

        elif combinatorial_data["problem_type"] == "set_covering":
            st.session_state.problem_type = "set_covering"
            sets = combinatorial_data["data"].get("sets", [])
            universe = combinatorial_data["data"].get("universe", [])
            if not sets or not universe:
                st.session_state.history.append(("assistant", "❌ Set Covering requires sets and universe."))
                st.rerun()  # Rerun to display error
            else:
                result = solve_set_covering(sets, universe)
                if isinstance(result[0], str):
                    st.session_state.history.append(("assistant", f"❌ {result[0]}"))
                    st.rerun()  # Rerun to display error
                else:
                    cover_indices, cost = result
                    formatted = format_set_covering_solution(cover_indices, cost, sets)
                    human_response = humanize_response(formatted, "set covering")
                    st.session_state.session_problem = combinatorial_data
                    st.session_state.history.append(("assistant", human_response))
                    st.rerun()  # Rerun to display response

    elif opt_types.get("stochastic_programming"):
        st.session_state.problem_type = "stochastic_programming"
        stochastic_data = extract_stochastic_components(latest_input)
        if "error" in stochastic_data:
            st.session_state.history.append(("assistant", f"❌ {stochastic_data['error']}"))
            st.rerun()  # Rerun to display error
        else:
            err2, optimal_value, optimal_x, optimal_ys = solve_two_stage_stochastic(stochastic_data)
            if err2:
                st.session_state.history.append(("assistant", f"❌ {err2}"))
                st.rerun()  # Rerun to display error
            else:
                var_names = stochastic_data["first_stage_variables"]
                second_var_names = stochastic_data["second_stage_variables"]
                num_scenarios = len(stochastic_data["scenarios"])
                var_details = "\n".join([f"  - {name}: {val:.2f}" for name, val in zip(var_names, optimal_x)])
                second_stage_details = ""
                for i in range(num_scenarios):
                    scen_name = stochastic_data["scenarios"][i]["name"]
                    second_decisions = optimal_ys[i]
                    decisions_text = ", ".join([f"{var}: {val:.2f}" for var, val in zip(second_var_names, second_decisions)])
                    second_stage_details += f"\n**Scenario {i+1} ({scen_name}):** {decisions_text}"
                summary = f"**Value:** {optimal_value:.2f}\n\n**First-Stage:**\n{var_details}\n\n**Second-Stage:**{second_stage_details}"
                human_response = humanize_response(summary, "stochastic")
                st.session_state.session_problem = stochastic_data
                st.session_state.history.append(("assistant", human_response))
                true_types = [k.replace('_', ' ').title() for k, v in opt_types.items() if v]
                st.session_state.history.append(("assistant", f"📚 **Types:** {', '.join(true_types)}"))
                st.rerun()  # Rerun to display response

def process_followup(latest_input: str, user_input: str, session_problem: Optional[Dict]):
    """Handle follow-up modifications."""
    st.session_state.history.append(("user", user_input))
    try:
        if st.session_state.problem_type == "linear":
            user_input = inject_compound_growth_coefficients(latest_input)
            modified_lpp, error = modify_lpp(session_problem, user_input)
            if error:
                st.session_state.history.append(("assistant", f"❌ {error}"))
                st.rerun()  # Rerun to display error
            else:
                err2, opt_val, opt_vars = solve_lpp(modified_lpp)
                if err2:
                    st.session_state.history.append(("assistant", f"❌ {err2}"))
                    st.rerun()  # Rerun to display error
                else:
                    formatted = format_solution(opt_val, opt_vars, modified_lpp.get("objective"), modified_lpp)
                    human_response = humanize_response(formatted, "linear")
                    st.session_state.session_problem = modified_lpp
                    st.session_state.history.append(("assistant", human_response))
                    st.rerun()  # Rerun to display response

        elif st.session_state.problem_type == "nonlinear":
            modified_nlp, error = modify_nlp(session_problem, user_input)
            if error:
                st.session_state.history.append(("assistant", f"❌ {error}"))
                st.rerun()  # Rerun to display error
            else:
                err2, opt_val, opt_vars = solve_nlp(modified_nlp)
                if err2:
                    st.session_state.history.append(("assistant", f"❌ {err2}"))
                    st.rerun()  # Rerun to display error
                else:
                    formatted = format_nlp_solution(opt_val, opt_vars, modified_nlp.get("objective_type", "minimize"), modified_nlp)
                    human_response = humanize_response(formatted, "nonlinear")
                    st.session_state.session_problem = modified_nlp
                    st.session_state.history.append(("assistant", human_response))
                    st.rerun()  # Rerun to display response

        elif st.session_state.problem_type == "knapsack":
            modified_combinatorial, error = modify_combinatorial(session_problem, user_input)
            if error or not modified_combinatorial:
                st.session_state.history.append(("assistant", f"❌ {error or 'No modified problem'}"))
                st.rerun()  # Rerun to display error
            else:
                values = modified_combinatorial["data"].get("values", [])
                weights = modified_combinatorial["data"].get("weights", [])
                capacity = modified_combinatorial["data"].get("capacity")
                item_names = modified_combinatorial["data"].get("item_names", [f"Item {i+1}" for i in range(len(values))])
                try:
                    optimal_value, chosen_items = solve_knapsack(values, weights, capacity, item_names)
                    formatted = f"Value: {optimal_value}\nItems: {chosen_items}"
                    human_response = humanize_response(formatted, "knapsack")
                    st.session_state.session_problem = modified_combinatorial
                    st.session_state.history.append(("assistant", human_response))
                    st.success(f"🎯 Value: {optimal_value}")
                    display_knapsack_solution(values, weights, item_names, chosen_items)
                    st.rerun()  # Rerun to display response
                except Exception as e:
                    logging.error(f"Knapsack error: {e}")
                    st.session_state.history.append(("assistant", f"❌ Knapsack error: {e}"))
                    st.rerun()  # Rerun to display error

        elif st.session_state.problem_type == "assignment":
            modified_combinatorial, error = modify_combinatorial(session_problem, user_input)
            if error or not modified_combinatorial:
                st.session_state.history.append(("assistant", f"❌ {error or 'No modified assignment'}"))
                st.rerun()  # Rerun to display error
            else:
                cost_matrix = np.array(modified_combinatorial["data"].get("cost_matrix", []))
                result = solve_assignment_problem(cost_matrix)
                if isinstance(result[0], str):
                    st.session_state.history.append(("assistant", f"❌ {result[0]}"))
                    st.rerun()  # Rerun to display error
                else:
                    cost, assignment = result
                    formatted = format_assignment_solution(cost, assignment)
                    human_response = humanize_response(formatted, "assignment")
                    st.session_state.session_problem = modified_combinatorial
                    st.session_state.history.append(("assistant", human_response))
                    st.rerun()  # Rerun to display response

        elif st.session_state.problem_type == "tsp":
            modified_combinatorial, error = modify_combinatorial(session_problem, user_input)
            if error or not modified_combinatorial:
                st.session_state.history.append(("assistant", f"❌ {error or 'No modified TSP'}"))
                st.rerun()  # Rerun to display error
            else:
                distance_matrix = np.array(modified_combinatorial["data"].get("distance_matrix", []))
                result = solve_tsp(distance_matrix)
                if isinstance(result[0], str):
                    st.session_state.history.append(("assistant", f"❌ {result[0]}"))
                    st.rerun()  # Rerun to display error
                else:
                    distance, path = result
                    formatted = format_tsp_solution(distance, path)
                    human_response = humanize_response(formatted, "TSP")
                    st.session_state.session_problem = modified_combinatorial
                    st.session_state.history.append(("assistant", human_response))
                    st.rerun()  # Rerun to display response

        elif st.session_state.problem_type == "set_covering":
            modified_combinatorial, error = modify_combinatorial(session_problem, user_input)
            if error or not modified_combinatorial:
                st.session_state.history.append(("assistant", f"❌ {error or 'No modified set covering'}"))
                st.rerun()  # Rerun to display error
            else:
                sets = modified_combinatorial["data"].get("sets", [])
                universe = modified_combinatorial["data"].get("universe", [])
                if not sets or not universe:
                    st.session_state.history.append(("assistant", "❌ Set Covering requires sets/universe."))
                    st.rerun()  # Rerun to display error
                else:
                    result = solve_set_covering(sets, universe)
                    if isinstance(result[0], str):
                        st.session_state.history.append(("assistant", f"❌ {result[0]}"))
                        st.rerun()  # Rerun to display error
                    else:
                        cover_indices, cost = result
                        formatted = format_set_covering_solution(cover_indices, cost, sets)
                        human_response = humanize_response(formatted, "set covering")
                        st.session_state.session_problem = modified_combinatorial
                        st.session_state.history.append(("assistant", human_response))
                        st.rerun()  # Rerun to display response

        elif st.session_state.problem_type == "stochastic_programming":
            stochastic_data = extract_stochastic_components(latest_input)
            if "error" in stochastic_data:
                st.session_state.history.append(("assistant", f"❌ {stochastic_data['error']}"))
                st.rerun()  # Rerun to display error
            else:
                err2, optimal_value, optimal_x, optimal_ys = solve_two_stage_stochastic(stochastic_data)
                if err2:
                    st.session_state.history.append(("assistant", f"❌ {err2}"))
                    st.rerun()  # Rerun to display error
                else:
                    var_names = stochastic_data["first_stage_variables"]
                    second_var_names = stochastic_data["second_stage_variables"]
                    num_scenarios = len(stochastic_data["scenarios"])
                    var_details = "\n".join([f"  - {name}: {val:.2f}" for name, val in zip(var_names, optimal_x)])
                    second_stage_details = ""
                    for i in range(num_scenarios):
                        scen_name = stochastic_data["scenarios"][i]["name"]
                        second_decisions = optimal_ys[i]
                        decisions_text = ", ".join([f"{var}: {val:.2f}" for var, val in zip(second_var_names, second_decisions)])
                        second_stage_details += f"\n**Scenario {i+1} ({scen_name}):** {decisions_text}"
                    summary = f"**Value:** {optimal_value:.2f}\n\n**First-Stage:**\n{var_details}\n\n**Second-Stage:**{second_stage_details}"
                    human_response = humanize_response(summary, "stochastic")
                    st.session_state.session_problem = stochastic_data
                    st.session_state.history.append(("assistant", human_response))
                    st.rerun()  # Rerun to display response

        else:
            st.session_state.history.append(("assistant", "❌ Unknown operation."))
            st.rerun()  # Rerun to display error

    except Exception as e:
        logging.error(f"Error in follow-up: {e}")
        st.session_state.history.append(("assistant", f"❌ Error in follow-up: {e}"))
        st.rerun()  # Rerun to display error



# --- Display Chat History ---
for sender, message in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(message)

# --- Display Token Count ---
st.markdown(f"Total tokens used: {sum(token_count)}")

# --- Run the App ---
if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        logging.error(f"Error running app: {e}")
        st.error(f"❌ Failed to start app: {e}")