import streamlit as st
from openai import OpenAI
from scipy.optimize import linprog, minimize, linear_sum_assignment
from itertools import permutations
import os
import re
import numpy as np
import pandas as pd
import math
import random
import ast
import logging
import requests  # Added for API access
import requests
import logging
import openai
import requests
import logging


def openai_chat_completion(messages, model="gpt-4", temperature=0):
    """Handle OpenAI API calls with both official client and fallback to requests"""
    API_KEY = "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"

    try:
        # First try with official OpenAI client
        openai.api_key = API_KEY
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        logging.warning(f"Official client failed, falling back to requests: {e}")
        try:
            # Fallback using requests
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API request failed: {response.text}")
        except Exception as e:
            logging.error(f"Both API methods failed: {e}")
            return f"❌ OpenAI call failed: {e}"



# --- Optimization Type Detection ---
OPT_TYPES = {
    "linear_programming": False,
    "integer_programming": False,
    "nonlinear_programming": False,
    "quadratic_programming": False,
    "convex_programming": False,
    "combinatorial_optimization": False,
    "dynamic_programming": False,
    "stochastic_optimization": False,
    "multi_objective_optimization": False,
    "set_covering": False,
    "stochastic_programming": False
}

def detect_optimization_type(problem_statement):
    prompt = f"""
    You are an expert in optimization theory. Read the following problem statement and determine which types of optimization problems it involves.

    ---
    {problem_statement}
    ---

    In addition to standard categories (linear programming, integer programming, combinatorial optimization, etc.), **pay special attention to problems resembling Set Covering Problems**, even if they are not explicitly named.

    Set covering problems typically involve:
    - A universe of required elements (e.g., skills, cities, resources).
    - A collection of sets (e.g., candidates, facilities, resources) covering parts of the universe.
    - The objective is to select the minimal number of sets to cover the entire universe.

    If you detect this structure, **toggle 'set_covering' to True** even if the problem does not use the words "set covering."

    Also, if the problem involves uncertainty (for example, random demand, uncertain costs, probabilistic events), detect it as a **stochastic_programming** problem.

    Typical signs:
    - Demand, costs, or other parameters have **different scenarios**.
    - Decisions have to be made **before** knowing the actual scenario.
    - The objective involves **expected values** or **chance constraints**.

    If such uncertainty is present, set `"stochastic_programming": True.

    Return a Python dictionary in the following format, setting only the relevant types to True:

    {OPT_TYPES}

    Respond ONLY with the dictionary. Do not include explanation.
    """

    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            detected_types = ast.literal_eval(match.group(0))
            if re.search(r'[a-zA-Z]\^\d+|[a-zA-Z]+\*[a-zA-Z]+|sin\(|cos\(|exp\(|log\(', problem_statement):
                detected_types["nonlinear_programming"] = True
            if "set cover" in problem_statement.lower() or "set covering" in problem_statement.lower():
                detected_types["set_covering"] = True
            return detected_types
        return OPT_TYPES
    except Exception as e:
        logging.error(f"Error in detect_optimization_type: {e}")
        return OPT_TYPES

# --- Utility Functions ---
def humanize_response(technical_output, problem_type="optimization"):
    prompt = f"""
    You are a formal mathematical assistant. The following is a technical explanation of a {problem_type} solution:

    ---
    {technical_output}
    ---

    Rewrite this in natural language to help a user understand what the solution means in simple terms.
    Highlight the optimal value, key variables, and what they should take away from it. Be brief, helpful, and conversational.
    """

    try:
        response =openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(⚠️ Could not generate humanized response: {e})\n\n{technical_output}"

def simplify_math_expressions(text):
    pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)\b')
    return pattern.sub(lambda m: str(float(m.group(1)) * float(m.group(2))), text)

# --- Linear Programming Functions ---
def extract_lpp_from_text(text):
    prompt = f"""
    You are a world-class mathematical assistant designed to extract structured Linear Programming Problems (LPPs) from natural language.

    ---

    Your task involves **three stages**:

    ### 🔁 Stage 1: Unit Standardization
    1. Convert all quantities into **SI units**.
    2. **Reframe the problem text using SI units** for internal processing and validation.
    3. **Before final output**, convert all variables and constraints **back into their original units** as used in the question to avoid confusing the user.

    ---

    ### 🧠 Stage 2: LPP Extraction
    From the question below, extract the following LPP components:
    - Objective function coefficients for **both** `maximize` and `minimize` objectives:
      - If only one is mentioned, infer the other by negating or mirroring as needed.
    - Inequality constraints: matrix `A_ub`, vector `b_ub`
    - Equality constraints: matrix `A_eq`, vector `b_eq`
    - Variable bounds: list of `(lower, upper)` tuples. Default: `(0, None)` if not mentioned.
    - Variable names: e.g., `["x1", "x2", ...]` (generate meaningful names when possible).
    - Constraint names: e.g., `["Raw Material Constraint", "Budget Constraint", ...]`
    - Objective type: `"maximize"`, `"minimize"`, or `"mixed"`
    Ensure all values (coefficients, RHS, bounds) are floats.

    ---

    ### 🔁 Stage 3: Matrix Verification (5-Pass Loop)
    Double-check the integrity of all matrices:
    - **Verify variable-to-column alignment**
    - **Ensure shape consistency** between `A_ub`, `b_ub`, `A_eq`, `b_eq`
    - **Confirm that all constraints, objective coefficients, and bounds reflect the original logic**
    - Repeat this validation logic **five times** before finalizing the dictionary to ensure consistency and correctness.

    ---

    ### 📝 Input:
    \"\"\"{text}\"\"\"

    ---

    ### ✅ Final Output Format
    Output ONLY a **valid Python dictionary** (no explanation, no markdown, no comments), strictly following this schema:

    {{
        "c_max": [float, ...],
        "c_min": [float, ...],
        "A_ub": [[float, ...], ...],
        "b_ub": [float, ...],
        "A_eq": [[float, ...], ...] or None,
        "b_eq": [float, ...] or None,
        "bounds": [(float, float or None), ...],
        "objective": "maximize" or "minimize" or "mixed",
        "variable_names": ["x1", "x2", ...],
        "constraint_names": ["Constraint 1", "Constraint 2", ...]
    }}

    Return only the dictionary. Do not include code blocks, comments, or any other content.
    ---

    ### 🚨 Missing Values Handling
    If any value (such as `b_eq`, `A_eq`, or variable bounds) is inferred as `None`, include a placeholder and **clearly mark the corresponding variable or constraint name**.

    The downstream logic will prompt the user to optionally fill in the missing values. So do not guess — just flag them with `None`.

    Only return the final LPP dictionary as described earlier.
    """

    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            raw_dict = match.group(0)
            raw_dict = simplify_math_expressions(raw_dict)
            parsed_dict = ast.literal_eval(raw_dict)
            parsed_dict = ask_user_to_fill_missing_values(parsed_dict)
            st.session_state.history.append(("assistant", parsed_dict))
            return parsed_dict, None
        return None, "Invalid LPP format returned (no dictionary match)."
    except Exception as e:
        return None, f"❌ Error parsing LPP: {e}"

def ask_user_to_fill_missing_values(problem_dict):
    logging.info("Starting to check for missing values in the problem dictionary.")
    updated = False
    missing_fields = []

    for key, value in problem_dict.items():
        if value is None or (isinstance(value, list) and not value):
            missing_fields.append(key)
            logging.debug(f"Missing field detected: {key}")

    for field in missing_fields:
        prompt = f"What value should be assigned to '{field}'? Please provide a valid input."
        try:
            response = openai_chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            question = response.choices[0].message.content.strip()
            logging.info(f"Generated question for missing field '{field}': {question}")
            user_input = st.text_input(question, key=f"missing_{field}")
            if user_input:
                if isinstance(problem_dict[field], list):
                    problem_dict[field] = ast.literal_eval(user_input)
                else:
                    problem_dict[field] = user_input
                updated = True
                logging.info(f"Updated field '{field}' with user input: {user_input}")
        except Exception as e:
            logging.error(f"Error generating question for '{field}': {e}")

    if updated:
        humanized_message = humanize_response("Thank you! The missing values have been updated.")
        st.success(humanized_message)
        logging.info("Humanized response displayed to the user.")

    return problem_dict

def solve_lpp(lpp_dict, alpha=0.5):
    c_max = lpp_dict.get('c_max')
    c_min = lpp_dict.get('c_min')
    A_ub = lpp_dict.get('A_ub')
    b_ub = lpp_dict.get('b_ub')
    A_eq = lpp_dict.get('A_eq')
    b_eq = lpp_dict.get('b_eq')
    bounds = lpp_dict.get('bounds')
    objective = lpp_dict.get('objective')

    num_vars = len(c_max) if c_max else len(c_min)
    if not bounds or len(bounds) != num_vars:
        bounds = [(0, None) for _ in range(num_vars)]

    if objective == 'maximize':
        c = [-val for val in c_max]
    elif objective == 'minimize':
        c = c_min
    elif objective == 'mixed':
        c = [(alpha * -x) + ((1 - alpha) * y) for x, y in zip(c_max, c_min)]
    else:
        return "Unknown objective type.", None, None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            if objective == 'maximize':
                res.fun = -res.fun
            return None, res.fun, res.x
        return f"LPP solving failed: {res.message}", None, None
    except ValueError as e:
        return f"Error solving LPP: {e}", None, None

def display_constraints(lpp_dict):
    details = []
    var_names = lpp_dict.get('variable_names') or [f"x{i+1}" for i in range(len(lpp_dict.get('c_max', lpp_dict.get('c_min', []))))]
    con_names = lpp_dict.get('constraint_names', [])

    if lpp_dict.get('A_ub'):
        for idx, (row, b) in enumerate(zip(lpp_dict['A_ub'], lpp_dict['b_ub'])):
            constraint = " + ".join(f"{round(coef, 2)}{var_names[i]}" for i, coef in enumerate(row))
            name = con_names[idx] if idx < len(con_names) else f"Constraint {idx+1}"
            details.append(f"{name}: {constraint} <= {b}")
    if lpp_dict.get('A_eq'):
        start_idx = len(lpp_dict.get('A_ub') or [])
        for i, (row, b) in enumerate(zip(lpp_dict['A_eq'], lpp_dict['b_eq'])):
            constraint = " + ".join(f"{round(coef, 2)}{var_names[j]}" for j, coef in enumerate(row))
            idx = start_idx + i
            name = con_names[idx] if idx < len(con_names) else f"Constraint {idx+1}"
            details.append(f"{name}: {constraint} = {b}")
    return "\n".join(details) or "No constraints found."

def format_solution(opt_val, opt_vars, objective, lpp_dict):
    if opt_val is None or opt_vars is None:
        return "No feasible solution found."

    var_names = lpp_dict.get('variable_names') or [f"x{i+1}" for i in range(len(opt_vars))]
    var_details = "\n".join([f"  - {name}: {val:.2f}" for name, val in zip(var_names, opt_vars)])

    summary = ""
    if objective == 'maximize' and lpp_dict.get('c_max'):
        terms = [f"{round(coef, 2)}x{i+1}" for i, coef in enumerate(lpp_dict['c_max'])]
        summary += "**Objective Function:** Maximize Z = " + " + ".join(terms) + "\n"
    elif objective == 'minimize' and lpp_dict.get('c_min'):
        terms = [f"{round(coef, 2)}x{i+1}" for i, coef in enumerate(lpp_dict['c_min'])]
        summary += "**Objective Function:** Minimize Z = " + " + ".join(terms) + "\n"
    elif objective == 'mixed':
        c_max = lpp_dict.get('c_max')
        c_min = lpp_dict.get('c_min')
        if c_max and c_min:
            terms = [f"(α*-{x} + (1-α)*{y})x{i+1}" for i, (x, y) in enumerate(zip(c_max, c_min))]
            summary += "**Objective Function:** Mixed = " + " + ".join(terms) + "\n"

    summary += "\n**Constraints**:\n" + display_constraints(lpp_dict) + "\n\n"
    result_text = f"Optimal Value: **{opt_val:.2f}**\n\nVariable Values:\n{var_details}"
    return summary + result_text

def modify_lpp(session_problem, user_input):
    prompt = f"""
    You are assisting in modifying a Linear Programming Problem (LPP). Here is the existing LPP in dictionary format:

    {session_problem}

    Based on this user instruction:
    "{user_input}"

    Return an updated version of the dictionary **with only the necessary changes made**.
    DO NOT remove or omit any fields from the original unless asked explicitly. Maintain structure integrity.

    Return ONLY the Python dictionary with changes implemented. No explanation or extra text.
    """
    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0)), None
        return None, "Failed to parse modified LPP."
    except Exception as e:
        return None, f"Error parsing modified LPP: {e}"

# --- Nonlinear Programming Functions ---
def extract_nlp_components(problem_statement):
    prompt = f"""
    You are an expert in extracting components of a Non-Linear Programming (NLP) problem from natural language.

    ---
    From the following problem statement, identify and extract:

    - **Objective Function (as a Python string representing a lambda function with variables 'x[0]', 'x[1]', ...):** Express the function to be minimized or maximized.
    - **Objective Type:** "minimize" or "maximize".
    - **Constraints (a list of dictionaries, each with 'type' ('eq' or 'ineq') and 'fun' (as a Python string representing a lambda function with 'x')):** Represent the equality and inequality constraints.
    - **Initial Guess (a list of initial values for the variables):** Provide a reasonable starting point for the optimization.
    - **Variable Names (a list of names for the variables):**
    - **Constraint Descriptions (a list of human-readable descriptions for the constraints):**

    If any component cannot be reliably extracted, indicate with None.

    For the given problem:
    "Nestlé produces two types of chocolates—Milk Chocolate and Dark Chocolate—and wants to maximize its profit while considering non-linear production costs. The selling price per unit is ₹200 for Milk Chocolate and ₹250 for Dark Chocolate. However, the cost per unit follows a non-linear pattern: ₹(50 + 0.01M²) for Milk Chocolate and ₹(70 + 0.008D²) for Dark Chocolate. The company has 10,000 kg of cocoa, with each Milk Chocolate requiring 3 kg and each Dark Chocolate needing 5 kg. The goal is to determine the optimal number of chocolates to produce while ensuring total cocoa usage does not exceed the limit."

    A potential output structure would be (adjust variable names and function definitions based on your interpretation):

    {{
        "objective_function": "lambda x: (200 * x[0]) + (250 * x[1]) - (50 + 0.01 * x[0]**2) - (70 + 0.008 * x[1]**2)",
        "objective_type": "maximize",
        "constraints": [
            {{'type': 'ineq', 'fun': "lambda x: 10000 - (3 * x[0] + 5 * x[1])" }}
        ],
        "initial_guess": [100, 100],
        "variable_names": ["Milk Chocolate", "Dark Chocolate"],
        "constraint_descriptions": ["Total cocoa usage within limit"]
    }}

    ---
    Problem Statement:
    \"\"\"{problem_statement}\"\"

    ---
    Return ONLY a Python dictionary in the specified format. Ensure that 'objective_function' and the 'fun' part of 'constraints' are strings representing lambda functions. Do not include any explanation or extra text.
    """
    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0)), None
        return None, "Invalid NLP format returned."
    except Exception as e:
        return None, f"Error parsing NLP: {e}"

def solve_nlp(nlp_dict):
    objective_func_str = nlp_dict.get('objective_function')
    objective_type = nlp_dict.get('objective_type', 'minimize')
    constraints_data = nlp_dict.get('constraints', [])
    initial_guess = nlp_dict.get('initial_guess')

    if not objective_func_str or not initial_guess:
        return "Objective function or initial guess not provided.", None, None

    try:
        objective_function = eval(objective_func_str)
    except Exception as e:
        return f"Error evaluating objective function: {e}", None, None

    formatted_constraints = []
    if constraints_data:
        for con in constraints_data:
            try:
                func = eval(con['fun'])
                formatted_constraints.append({'type': con['type'], 'fun': func})
            except Exception as e:
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
        return f"NLP solving failed: {result.message}", None, None
    except Exception as e:
        return f"Error solving NLP: {e}", None, None

def set_nlp_objective_original(nlp_data):
    objective_function = nlp_data.get('objective_function', '')
    if objective_function:
        match = re.search(r'lambda x: (.+)', objective_function)
        nlp_data['objective_function_original'] = match.group(1) if match else 'Objective Function'
    else:
        nlp_data['objective_function_original'] = 'Objective Function'
    return nlp_data

def format_nlp_solution(opt_val, opt_vars, objective, nlp_dict):
    if opt_val is None or opt_vars is None:
        return "No feasible solution found."

    var_names = nlp_dict.get('variable_names') or [f"x{i+1}" for i in range(len(opt_vars))]
    var_details = "\n".join([f"  - {name}: {val:.4f}" for name, val in zip(var_names, opt_vars)])
    objective_str = nlp_dict.get('objective_function_original', "Objective Function")

    summary = f"**Objective Function:** {objective_str} ({objective})\n\n"
    if nlp_dict.get('constraint_descriptions'):
        summary += "**Constraints**:\n" + "\n".join([f"- {desc}" for desc in nlp_dict['constraint_descriptions']]) + "\n\n"

    result_text = f"Optimal Value: **{opt_val:.4f}**\n\nVariable Values:\n{var_details}"
    return summary + result_text

def modify_nlp(session_nlp, user_input):
    prompt = f"""
    You are assisting in modifying a Non-Linear Programming (NLP). Here is the existing NLP in dictionary format:

    {session_nlp}

    Based on this user instruction:
    "{user_input}"

    Return an updated version of the dictionary **with only the necessary changes made**.
    Ensure that the 'objective_function' and constraint 'fun' values remain as Python-executable strings (lambda functions).
    DO NOT remove or omit any fields from the original unless asked explicitly. Maintain structure integrity.

    Return ONLY the Python dictionary with changes implemented. No explanation or extra text.
    """
    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0)), None
        return None, "Failed to parse modified NLP."
    except Exception as e:
        return None, f"Error parsing modified NLP: {e}"

# --- Combinatorial Optimization Functions ---
def solve_knapsack(values, weights, capacity, item_names=None):
    if not isinstance(capacity, (int, float)) or capacity != int(capacity):
        raise ValueError("Capacity must be an integer.")
    if not all(isinstance(v, (int, float)) and float(v).is_integer() for v in values):
        raise ValueError("All values must be integers.")
    if not all(isinstance(w, (int, float)) and float(w).is_integer() for w in weights):
        raise ValueError("All weights must be integers.")

    values = [int(v) for v in values]
    weights = [int(w) for w in weights]
    capacity = int(capacity)
    n = len(values)

    if item_names is None:
        item_names = [f"Item {i+1}" for i in range(n)]
    elif len(item_names) != n:
        raise ValueError("Length of item_names must match number of items.")

    dp = np.zeros((n + 1, capacity + 1), dtype=int)

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    optimal_value = dp[n][capacity]
    chosen_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen_items.append(item_names[i - 1])
            w -= weights[i - 1]

    return optimal_value, chosen_items[::-1]

def display_knapsack_solution(values, weights, item_names, chosen_items):
    data = []
    for name, val, wt in zip(item_names, values, weights):
        selected = "✅ Yes" if name in chosen_items else None
        data.append({"Item": name, "Value": val, "Weight": wt, "Selected": selected})

    df = pd.DataFrame(data)
    st.subheader("📋 Knapsack Decision Table")
    st.dataframe(df, use_container_width=True)

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if len(distance_matrix) != len(distance_matrix[0]):
        return "Error: Distance matrix must be square.", None
    if n == 0:
        return "Error: Distance matrix cannot be empty.", None
    if n == 1:
        return 0, [0]
    if n > 10:
        return "Warning: TSP instance is relatively large for brute force.", None

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
        return None, "Error: No valid TSP tour found."
    return min_distance, list(best_path)

def solve_set_covering(sets, universe):
    uncovered = set(universe)
    cover_indices = []
    set_indices = list(range(len(sets)))

    while uncovered:
        best_set_index = -1
        max_elements_covered = -1

        for i, s in enumerate(sets):
            elements_covered = len(set(s) & uncovered)
            if elements_covered > max_elements_covered:
                max_elements_covered = elements_covered
                best_set_index = i

        if best_set_index == -1:
            return None, "Error: No set covers the remaining elements."
        cover_indices.append(best_set_index)
        uncovered -= set(sets[best_set_index])

    return cover_indices, len(cover_indices)

def solve_assignment_problem(cost_matrix):
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        optimal_cost = cost_matrix[row_ind, col_ind].sum()
        assignment = list(zip(row_ind, col_ind))
        return {"cost": optimal_cost, "assignment": assignment}, None
    except Exception as e:
        return None, f"Error solving assignment problem: {e}"

def format_tsp_solution(distance, path):
    path_str = " -> ".join(str(x + 1) for x in path)
    return f"Optimal Path: {path_str}\nTotal Distance: {distance:.2f}"

def format_assignment_solution(result):
    cost, assignment = result["cost"], result["assignment"]
    assignment_str = "\n".join(f"Worker {w + 1} assigned to Job {j + 1}" for w, j in assignment)
    return f"Optimal Assignment:\n{assignment_str}\nTotal Cost: {cost:.2f}"

def format_set_covering_solution(cover_indices, cost, sets):
    if cover_indices is None:
        return "No feasible solution found."
    covered_sets = [f"Set {i + 1}: {sets[i]}" for i in cover_indices]
    return f"Sets Chosen: {', '.join(str(i + 1) for i in cover_indices)}\nCovered Sets:\n" + "\n".join(covered_sets) + f"\nTotal Sets Used: {cost}"

def extract_combinatorial_data(problem_statement):
    prompt = f"""
    You are an expert in combinatorial optimization. Analyze the following problem statement and:

    1. Identify the problem type: Traveling Salesman Problem (TSP), Assignment Problem, Knapsack Problem, or Set Covering Problem.
    2. Extract the data: Distance matrix (TSP), cost matrix (Assignment), values/weights/capacity/item_names (Knapsack), or sets/universe (Set Covering).

    ---
    Problem Statement:
    \"\"\"{problem_statement}\"\"\"
    ---

    Respond with a Python dictionary:
    {{
        "problem_type": "tsp|assignment|knapsack|set_covering|unknown",
        "data": {{
            "distance_matrix": [[float, ...], ...],
            "cost_matrix": [[float, ...], ...],
            "values": [float, ...], "weights": [float, ...], "capacity": float, "item_names": [str, ...],
            "sets": [[str, ...], ...], "universe": [str, ...]
        }},
        "error": null
    }}

    Return ONLY the dictionary.
    """
    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0)), None
        return None, "Invalid combinatorial format."
    except Exception as e:
        logging.error(f"Error in extract_combinatorial_data: {e}")
        return None, f"Error parsing combinatorial data: {e}"

def modify_combinatorial(session_problem, user_input):
    prompt = f"""
    You are assisting in modifying a combinatorial optimization problem. Here is the existing problem in dictionary format:

    {session_problem}

    Based on this user instruction:
    "{user_input}"

    Return an updated version of the dictionary with only the necessary changes made.
    DO NOT remove or omit any fields from the original unless explicitly asked. Maintain structure integrity.

    Return ONLY the Python dictionary. No explanation or extra text.
    """
    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0)), None
        return None, "Failed to parse modified combinatorial problem."
    except Exception as e:
        return None, f"Error parsing modified combinatorial problem: {e}"

# --- Stochastic Programming Functions ---
def extract_stochastic_components(problem_text):
    prompt = f"""
    You are an expert in stochastic optimization.

    From the following problem statement, extract:
    - **First-stage variables**: Variables decided before uncertainty is realized.
    - **Second-stage variables**: Variables decided after uncertainty is known.
    - **Scenarios**: Each scenario should include:
      - Name
      - Probability
      - Scenario-specific cost coefficients
      - Scenario-specific constraints

    ---
    Problem:
    \"\"\"{problem_text}\"\"\"
    ---

    Format the output strictly as a Python dictionary:
    {{
        "first_stage_variables": ["x1", "x2", ...],
        "second_stage_variables": ["y1", "y2", ...],
        "scenarios": [
            {{
                "name": "High Demand",
                "probability": 0.6,
                "cost_coefficients": [float, float, ...],
                "constraints": ["constraint 1 description", "constraint 2 description", ...]
            }},
            ...
        ],
        "objective": "maximize" or "minimize"
    }}
    Return ONLY the dictionary. No explanations.
    """
    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0)), None
        return None, "Failed to parse stochastic components: No valid dictionary found."
    except (SyntaxError, ValueError) as e:
        logging.error(f"Error parsing stochastic components: {e}")
        return None, f"Failed to parse stochastic components: Invalid dictionary format ({e})."
    except Exception as e:
        logging.error(f"Error processing stochastic components: {e}")
        return None, f"Failed to extract stochastic components: Unexpected error ({e})."

def solve_two_stage_stochastic(stochastic_data):
    try:
        first_stage_vars = stochastic_data.get("first_stage_variables", [])
        second_stage_vars = stochastic_data.get("second_stage_variables", [])
        scenarios = stochastic_data.get("scenarios", [])
        objective = stochastic_data.get("objective", "minimize")

        num_x = len(first_stage_vars)
        num_y = len(second_stage_vars)
        num_scenarios = len(scenarios)

        if not all([num_x, num_y, num_scenarios]):
            return "Error: Missing variables or scenarios.", None, None, None

        total_vars = num_x + (num_y * num_scenarios)

        def expected_total_value(z):
            x = z[:num_x]
            total = 0.0
            for i, scen in enumerate(scenarios):
                prob = scen["probability"]
                cost_coeffs = scen["cost_coefficients"]
                y_s = z[num_x + i * num_y: num_x + (i + 1) * num_y]
                scen_cost = sum(c * xi for c, xi in zip(cost_coeffs[:len(x)], x)) + \
                            sum(c * yi for c, yi in zip(cost_coeffs[len(x):], y_s))
                total += prob * scen_cost
            return -total if objective == "maximize" else total

        bounds = [(0, None) for _ in range(total_vars)]
        result = minimize(expected_total_value, [1.0] * total_vars, bounds=bounds, method='SLSQP')
        if result.success:
            optimal_value = -result.fun if objective == "maximize" else result.fun
            optimal_value = result
            optimal_x = result.x[:num_x]
            optimal_ys = optimal_x[num_x + i * num_y: num_x + (i + 1) * num_y]
            return None, optimal_value, optimal_x, optimal_y
        return optimal_value, optimal_x, None, None
    except Exception as e:
        return None, f"Error solving two-stage stochastic problem: {e}", None, None, None

# --- Additional Functions ---
def classify_user_input(user_input, user_input_history=None):
    prompt = f"""
    You are an intelligent assistant that classifies user instructions in the context of optimization problems.

    Given the following problem history (if any):
    \"\"\"{user_input_history or 'None'}\"\"

    And the user's latest input:
    \"\"\"{user_input}\"\"

    Decide whether the user's input is:
    - A completely new optimization problem
    - A modification or follow-up to the existing one.

    Return only one word:
    - "new\" if it is a new problem
    - "" if it is a follow-up to the existing one
    """
    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        return "" if "new" in answer else st.session_state.problem_type
    except Exception as e:
        logging.error(f"Error classifying user input: {e}")
        return None, "unknown"

def contains_compound_interest_terms(text):
    interest_keywords = ["interest", "returns", "compounded", "compounding", "growth"]
    return any(keyword.lower() in text.lower() for keyword in interest_keywords)

def extract_growth_coefficients_with_llm(text):
    prompt = f"""
    From the following optimization problem statement, extract compound growth coefficients for each asset class:

    ---
    {text}
    ---

    For each asset type mentioned (e.g., equity mutual funds, real estate, fixed deposits), return a dictionary:
    {{
        "name": "<asset_name>",
        "rate": <annual return as decimal>,
        "years": <time horizon in years>
    }}

    Return a Python list of dictionaries, e.g.:
    [
        {{"name": "equity", "rate": 0.12, "years": 7}},
        {{"name": "real_estate", "rate": 0.10, "years": 7}},
        {{"name": "fixed", "rate": 0.055, "years": 7}}
    ]
    """
    try:
        response = copenai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\[.*?\](?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
        return []
    except Exception as e:
        logging.error(f"Error extracting growth coefficients: {e}")
        return []

def inject_compound_gold_coefficients(excel_text):
    if not contains_compound_interest_terms(excel_text):
        return excel_text

    growth_coefficients = extract_growth_coefficients_with_llm(excel_text)
    growth_map = {}
    for entry in growth_coefficients:
        name = entry['name'].lower().replace(' ', '_')
        rate = entry['rate']
        years = entry['years']
        growth_factor = round((1 + rate) ** years, 4)
        growth_map[name] = growth_factor

    summary = "\n\n# Calculated Growth Coefficients:\n"
    for name, factor in growth_map.items():
        summary += f"{name}: {factor:.4f}\n"
    return excel_text + summary

def check_problem_completeness(problem_type, session_problem, user_input):
    problem_type_description = str(problem_type).replace('_', ' ') if problem_type else "unknown optimization"
    prompt = f"""
    You are an expert in {problem_type_description} optimization.

    Check if the following problem statement provides sufficient information to solve a {problem_type_description} problem:
    - Current problem dictionary: {session_problem}
    - User input: "{user_input}"

    Return a Python dictionary:
    {{
        "is_complete": bool,
        "missing_fields": ["field1", "field2", ...]
    }}
    """
    try:
        response = openai_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        match = re.search(r'\{.*?\}(?=\s*$|\s*\n)', response.choices[0].message.content, re.DOTALL)
        if match:
            completeness_check = ast.literal_eval(match.group(0))
            if not completeness_check.get("is_complete", False):
                missing_fields = completeness_check.get("missing_fields", [])
                error_message = f"Missing information: {', '.join(missing_fields)}"
                return completeness_check, error_message
            return completeness_check, None
        return None, "Failed to parse completeness check response."
    except Exception as e:
        logging.error(f"Error during completeness check: {e}")
        return None, f"Error checking completeness: {e}"

def handle_missing_information(problem_type, session_problem, user_input):
    st.session_state.history.append(("user", user_input))
    opt_types = detect_optimization_type(user_input)
    problem_type_description = str(problem_type).replace('_', ' ').title() if problem_type else "Unknown Optimization"

    completeness_check, error_message = check_problem_completeness(problem_type, session_problem, user_input)

    if completeness_check is None:
        st.session_state.history.append(("assistant", f"❌ Failed to check problem completeness: {error_message}"))
        return None, error_message

    if error_message:
        st.session_state.history.append(("assistant", f"⚠️ {error_message}"))
        missing_fields = completeness_check.get("missing_fields", [])
        questions = []
        for field in missing_fields:
            prompt = f"What value should be assigned to '{field}' for the {problem_type_description} problem? Please provide a valid input."
            try:
                response = openai_chat_completion(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                question = response.choices[0].message.content.strip()
                questions.append(question)
            except Exception as e:
                questions.append(f"Error generating question for '{field}': {e}")
        humanized_questions = humanize_response("\n".join(questions))
        st.session_state.history.append(("assistant", humanized_questions))
        return questions, None
    return None, None

# --- Streamlit App ---
st.set_page_config(page_title="Optimization Solver", layout="wide")
st.title("🔢 Optimization Problem Solver")
st.markdown("Describe your optimization problem in natural language or upload an Excel file, and let AI solve it interactively.")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'session_problem' not in st.session_state:
    st.session_state.session_problem = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'user_input_history' not in st.session_state:
    st.session_state.user_input_history = []

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)

# --- Excel File Upload ---
st.subheader("📄 Optional: Upload an Excel File")
uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if st.session_state.get("excel_history"):
    if st.button("🔄 Clear Uploaded File and Start New Problem"):
        st.session_state.pop("excel_history", None)
        st.session_state.session_problem = None
        st.session_state.problem_type = None
        st.session_state.history = []

if uploaded_file and "excel_history" not in st.session_state:
    try:
        xl = pd.ExcelFile(uploaded_file)
        df_dict = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xl.sheet_names}
        st.success("✅ Excel file uploaded successfully.")

        def describe_excel(excel_dict):
            content = ""
            for sheet_name, df in excel_dict.items():
                content += f"Sheet: {sheet_name}\n{df.to_string(index=False)}\n\n"
            return content

        excel_text = describe_excel(df_dict)
        opt_types = detect_optimization_type(excel_text)

        if opt_types.get("linear_programming", False):
            st.session_state.problem_type = "linear"
            user_input = inject_compound_gold_coefficients(excel_text)
            lpp_data, error = extract_lpp_from_text(user_input)
            if error:
                st.error(f"❌ {error}")
            else:
                err2, opt_val, opt_vars = solve_lpp(lpp_data)
                if err2:
                    st.error(f"❌ {err2}")
                else:
                    formatted = format_solution(opt_val, opt_vars, lpp_data.get("objective"), lpp_data)
                    human_response = humanize_response(formatted, "linear programming")
                    st.session_state.session_problem = lpp_data
                    st.session_state.history.append(("assistant", "📜 **Excel file processed as Linear Programming!**"))
                    st.session_state.history.append(("assistant", human_response))

        elif opt_types.get("nonlinear_programming", False):
            st.session_state.problem_type = "nonlinear"
            nlp_data, error = extract_nlp_components(excel_text)
            if error:
                st.error(f"❌ {error}")
            else:
                err2, opt_val, opt_vars = solve_nlp(nlp_data)
                if err2:
                    st.error(f"❌ {err2}")
                else:
                    nlp_data = set_nlp_objective_original(nlp_data)
                    formatted = format_nlp_solution(opt_val, opt_vars, nlp_data.get("objective_type", "minimize"), nlp_data)
                    human_response = humanize_response(formatted, "nonlinear programming")
                    st.session_state.session_problem = nlp_data
                    st.session_state.history.append(("assistant", "📜 **Excel file processed as Nonlinear Programming!**"))
                    st.session_state.history.append(("assistant", human_response))

        elif opt_types.get("stochastic_programming", False):
            st.session_state.problem_type = "stochastic_programming"
            stochastic_data, error = extract_stochastic_components(excel_text)
            if error:
                st.error(f"❌ {error}")
            else:
                err2, optimal_value, optimal_x, optimal_ys = solve_two_stage_stochastic(stochastic_data)
                if err2:
                    st.error(f"❌ {err2}")
                else:
                    var_names = stochastic_data.get("first_stage_variables", [])
                    second_var_names = stochastic_data.get("second_stage_variables", [])
                    scenarios = stochastic_data.get("scenarios", [])
                    var_details = "\n".join([f"  - {name}: {val:.2f}" for name, val in zip(var_names, optimal_x)])
                    second_stage_details = ""
                    for i, (scen, y) in enumerate(zip(scenarios, optimal_ys)):
                        scen_name = scen["name"]
                        decisions_text = ", ".join([f"{var}: {val:.2f}" for var, val in zip(second_var_names, y)])
                        second_stage_details += f"\n**Scenario {i+1} ({scen_name}):** {decisions_text}"
                    summary = f"**Optimal Expected Value:** {optimal_value:.2f}\n\n**First-Stage Decisions:**\n{var_details}\n\n**Second-Stage Decisions:**{second_stage_details}"
                    human_response = humanize_response(summary, "stochastic programming")
                    st.session_state.session_problem = stochastic_data
                    st.session_state.history.append(("assistant", "📜 **Excel file processed as Stochastic Programming!**"))
                    st.session_state.history.append(("assistant", human_response))

        elif opt_types.get("combinatorial_optimization", False):
            combinatorial_data, error = extract_combinatorial_data(excel_text)
            if error:
                st.error(f"❌ {error}")
            elif combinatorial_data["problem_type"] == "tsp":
                st.session_state.problem_type = "tsp_combinatorial"
                distance_matrix = np.array(combinatorial_data["data"].get("distance_matrix", []))
                distance, path = solve_tsp(distance_matrix)
                if isinstance(path, str):  # Error case
                    st.error(f"❌ {path}")
                else:
                    formatted = format_tsp_solution(distance, path)
                    human_response = humanize_response(formatted, "traveling salesman problem")
                    st.session_state.session_problem = combinatorial_data
                    st.session_state.history.append(("assistant", "📜 **Excel file processed as TSP!**"))
                    st.session_state.history.append(("assistant", human_response))

            elif combinatorial_data["problem_type"] == "assignment":
                st.session_state.problem_type = "assignment_combinatorial"
                cost_matrix = np.array(combinatorial_data["data"].get("cost_matrix", []))
                result, error = solve_assignment_problem(cost_matrix)
                if error:
                    st.error(f"❌ {error}")
                else:
                    formatted = format_assignment_solution(result)
                    human_response = humanize_response(formatted, "assignment problem")
                    st.session_state.session_problem = combinatorial_data
                    st.session_state.history.append(("assistant", "📜 **Excel file processed as Assignment Problem!**"))
                    st.session_state.history.append(("assistant", human_response))

            elif combinatorial_data["problem_type"] == "knapsack":
                st.session_state.problem_type = "knapsack_combinatorial"
                values = np.array(combinatorial_data["data"].get("values", []))
                weights = np.array(combinatorial_data["data"].get("weights", []))
                capacity = np.array(combinatorial_data["data"].get("capacity"))
                item_names = np.array(combinatorial_data["data"].get("item_names", []))
                try:
                    optimal_value, chosen_items = optimal_value_knapsack(values, weights, capacity, item_names)
                    formatted = f"Optimal Knapsack Value: {optimal_value}\nChosen Items: {', '.join(chosen_items)}"
                    human_response = humanize_response(formatted_solution, "knapsack problem")
                    st.session_state.session_problem = combinatorial_data["history"]
                    st.session_state["history"].append(("assistant", "📜 **Excel file processed as Knapsack Problem!**"))
                    st.session_state_history.append(("history", human_response))
                    st.success(f"✅ Optimal Value: {optimal_value}")
                    display_knapsack_solution(values, weights, item_names, chosen_items)
                except Exception as e:
                    st.error(f"❌ Error solving knapsack problem: {e}")

            elif combinatorial_data.get["problem_type"] == "set_covering":
                st.session_state_problem_type = "set_covering"
                sets = np.array(combinatorial_data["data"].get("sets", []))
                universe = np.array(combinatorial_data["data"].get("universe", []))
                if not sets or not universe:
                    st.error("Error: Set covering problem requires both 'sets' and 'universe' data.")
                else:
                    st.error(f"Error: {err} cover_indices, error")
                    human_response = humanize_response(formatted_response, "set covering problem")
                    st.session_state.session_problem = combinatorial_data
                    st.session_state.history.append(("assistant", "📜 **Excel file processed as Set Covering Problem!**"))
                    st.session_state.history.append(("assistant", human_response))

            else:
                st.warning("⚠️ Could not classify Excel file as a supported optimization problem.")

        st.session_state.excel_history = True

    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        st.error(f"❌ {e}")


# --- Chat Interface ---
user_input = st.chat_input("Enter a new problem description or a follow-up modification...")
if user_input:
    with st.spinner("Processing..."):
        try:
            # Update user input history (string concatenation as in PrototypeMindY)
            st.session_state.user_input_history += (
                ". " + user_input if st.session_state.user_input_history else user_input)

            # Classify input as new or follow-up
            st.session_state.problem_type = classify_user_input(
                user_input,
                st.session_state.session_problem
            )

            if st.session_state.problem_type == '':
                # New problem
                opt_types = detect_optimization_type(user_input)
                if not any(opt_types.values()):
                    questions, error_message = handle_missing_information(
                        '',
                        st.session_state.session_problem,
                        user_input
                    )
                    if error_message:
                        st.error(f"Error: {error_message}")
                    st.session_state.history.append(("user", user_input))
                    if questions:
                        st.session_state.history.append(("assistant", "\n".join(questions)))
                else:
                    st.session_state.problem_type = next(key for key, val in opt_types.items() if val)
                    completeness_check, error_message = check_problem_completeness(
                        st.session_state.problem_type,
                        st.session_state.session_problem,
                        user_input
                    )
                    if error_message:
                        handle_missing_information(
                            st.session_state.problem_type,
                            st.session_state.session_problem,
                            user_input
                        )
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"⚠️ {error_message}"))
                    else:
                        st.success("✅ Problem statement is complete!")
                        st.session_state.history.append(("user", user_input))

                        if opt_types.get("linear_programming", False):
                            st.session_state.problem_type = "linear"
                            user_input = inject_compound_growth_coefficients(user_input)
                            lpp_data, error = extract_lpp_from_text(user_input)
                            if error:
                                st.session_state.history.append(("assistant", f"❌ {error}"))
                            else:
                                err2, opt_val, opt_vars = solve_lpp(lpp_data)
                                if err2:
                                    st.session_state.history.append(("assistant", f"❌ {err2}"))
                                else:
                                    formatted = format_solution(opt_val, opt_vars, lpp_data.get("objective"), lpp_data)
                                    human_response = humanize_response(formatted, "linear programming")
                                    st.session_state.session_problem = lpp_data
                                    st.session_state.history.append(("assistant", human_response))

                        elif opt_types.get("nonlinear_programming", False):
                            st.session_state.problem_type = "nonlinear"
                            nlp_data, error = extract_nlp_components(user_input)
                            if error:
                                st.session_state.history.append(("assistant", f"❌ {error}"))
                            else:
                                err2, opt_val, opt_vars = solve_nlp(nlp_data)
                                if err2:
                                    st.session_state.history.append(("assistant", f"❌ {err2}"))
                                else:
                                    nlp_data['objective_function_original'] = (
                                        re.search(r"lambda x: (.+)", nlp_data.get('objective_function', ''))[1]
                                        if nlp_data.get('objective_function') else "Objective Function"
                                    )
                                    formatted = format_nlp_solution(
                                        opt_val, opt_vars, nlp_data.get("objective_type", "minimize"), nlp_data
                                    )
                                    human_response = humanize_response(formatted, "nonlinear programming")
                                    st.session_state.session_problem = nlp_data
                                    st.session_state.history.append(("assistant", human_response))

                        elif opt_types.get("combinatorial_optimization", False):
                            combinatorial_data = extract_combinatorial_data(user_input)
                            st.session_state.session_problem = combinatorial_data

                            if combinatorial_data["problem_type"] == "tsp":
                                st.session_state.problem_type = "combinatorial_tsp"
                                distance_matrix = np.array(combinatorial_data["data"].get("distance_matrix", []))
                                result = solve_tsp(distance_matrix)
                                if isinstance(result, str):
                                    st.session_state.history.append(("assistant", f"❌ {result}"))
                                else:
                                    distance, path = result
                                    formatted = format_tsp_solution(distance, path)
                                    human_response = humanize_response(formatted, "traveling salesman problem")
                                    st.session_state.history.append(("assistant", human_response))

                            elif combinatorial_data["problem_type"] == "assignment":
                                st.session_state.problem_type = "combinatorial_assignment"
                                cost_matrix = np.array(combinatorial_data["data"].get("cost_matrix", []))
                                result = solve_assignment_problem(cost_matrix)
                                if isinstance(result, str):
                                    st.session_state.history.append(("assistant", f"❌ {result}"))
                                else:
                                    cost, assignment = result
                                    formatted = format_assignment_solution(cost, assignment)
                                    human_response = humanize_response(formatted, "assignment problem")
                                    st.session_state.history.append(("assistant", human_response))

                            elif combinatorial_data["problem_type"] == "knapsack":
                                st.session_state.problem_type = "combinatorial_knapsack"
                                values = np.array(combinatorial_data["data"].get("values", []))
                                weights = np.array(combinatorial_data["data"].get("weights", []))
                                capacity = combinatorial_data["data"].get("capacity")
                                item_names = combinatorial_data["data"].get("item_names", [f"Item {i + 1}" for i in
                                                                                           range(len(values))])
                                try:
                                    optimal_value, chosen_items = solve_knapsack(values, weights, capacity, item_names)
                                    formatted = f"Optimal Knapsack Value: {optimal_value}\nChosen Items: {', '.join(chosen_items)}"
                                    human_response = humanize_response(formatted, "knapsack problem")
                                    st.session_state.history.append(("assistant", human_response))
                                    st.success(f"🎯 Optimal Value: {optimal_value}")
                                    display_knapsack_solution(values, weights, item_names, chosen_items)
                                except Exception as e:
                                    st.session_state.history.append(("assistant", f"❌ Error solving knapsack: {e}"))

                            elif combinatorial_data["problem_type"] == "set_covering":
                                st.session_state.problem_type = "set_covering"
                                sets = combinatorial_data["data"].get("sets", [])
                                universe = combinatorial_data["data"].get("universe", [])
                                if not sets or not universe:
                                    st.session_state.history.append(("assistant",
                                                                     "❌ Error: Set covering problem requires both 'sets' and 'universe' data."))
                                else:
                                    result = solve_set_covering(sets, universe)
                                    if isinstance(result, str):
                                        st.session_state.history.append(("assistant", f"❌ {result}"))
                                    else:
                                        cover_indices, cost = result
                                        formatted = format_set_covering_solution(cover_indices, cost, sets)
                                        human_response = humanize_response(formatted, "set covering problem")
                                        st.session_state.history.append(("assistant", human_response))

                        elif opt_types.get("stochastic_programming", False):
                            st.session_state.problem_type = "stochastic_programming"
                            stochastic_data = extract_stochastic_components(user_input)
                            if "error" in stochastic_data:
                                st.session_state.history.append(("assistant", f"❌ {stochastic_data['error']}"))
                            else:
                                err2, optimal_value, optimal_x, optimal_ys = solve_two_stage_stochastic(stochastic_data)
                                if err2:
                                    st.session_state.history.append(("assistant", f"❌ {err2}"))
                                else:
                                    var_names = stochastic_data.get("first_stage_variables", [])
                                    second_var_names = stochastic_data.get("second_stage_variables", [])
                                    scenarios = stochastic_data.get("scenarios", [])
                                    var_details = "\n".join(
                                        [f"  - {name}: {val:.2f}" for name, val in zip(var_names, optimal_x)])
                                    second_stage_details = ""
                                    for i, (scen, y) in enumerate(zip(scenarios, optimal_ys)):
                                        scen_name = scen["name"]
                                        decisions_text = ", ".join(
                                            [f"{var}: {val:.2f}" for var, val in zip(second_var_names, y)])
                                        second_stage_details += f"\n**Scenario {i + 1} ({scen_name}):** {decisions_text}"
                                    summary = (
                                        f"**Optimal Expected Value:** {optimal_value:.2f}\n\n"
                                        f"**First-Stage Decisions:**\n{var_details}\n\n"
                                        f"**Second-Stage Decisions:**{second_stage_details}"
                                    )
                                    human_response = humanize_response(summary, "stochastic programming")
                                    st.session_state.session_problem = stochastic_data
                                    st.session_state.history.append(("assistant", human_response))

                        true_types = [key.replace("_", " ").title() for key, val in opt_types.items() if val]
                        st.session_state.history.append(("assistant",
                                                         f"📚 **Detected Optimization Types:** {', '.join(true_types) or 'None detected'}"))

            else:
                # Follow-up problem
                opt_types = detect_optimization_type(st.session_state.user_input_history)
                st.session_state.history.append(("user", user_input))

                if st.session_state.problem_type == "linear":
                    user_input = inject_compound_growth_coefficients(user_input)
                    modified_lpp, error = modify_lpp(st.session_state.session_problem, user_input)
                    if error:
                        st.session_state.history.append(("assistant", f"❌ {error}"))
                    else:
                        err2, opt_val, opt_vars = solve_lpp(modified_lpp)
                        if err2:
                            st.session_state.history.append(("assistant", f"❌ {err2}"))
                        else:
                            formatted = format_solution(opt_val, opt_vars, modified_lpp.get("objective"), modified_lpp)
                            human_response = humanize_response(formatted, "linear programming")
                            st.session_state.session_problem = modified_lpp
                            st.session_state.history.append(("assistant", human_response))

                elif st.session_state.problem_type == "nonlinear":
                    modified_nlp, error = modify_nlp(st.session_state.session_problem, user_input)
                    if error:
                        st.session_state.history.append(("assistant", f"❌ {error}"))
                    else:
                        err2, opt_val, opt_vars = solve_nlp(modified_nlp)
                        if err2:
                            st.session_state.history.append(("assistant", f"❌ {err2}"))
                        else:
                            formatted = format_nlp_solution(opt_val, opt_vars,
                                                            modified_nlp.get("objective_type", "minimize"),
                                                            modified_nlp)
                            human_response = humanize_response(formatted, "nonlinear programming")
                            st.session_state.session_problem = modified_nlp
                            st.session_state.history.append(("assistant", human_response))

                elif st.session_state.problem_type == "combinatorial_knapsack":
                    modified_knapsack, error = modify_combinatorial(st.session_state.session_problem, user_input)
                    if error or modified_knapsack is None:
                        st.session_state.history.append(("assistant", f"❌ {error}"))
                    else:
                        values = np.array(modified_knapsack["data"].get("values", []))
                        weights = np.array(modified_knapsack["data"].get("weights", []))
                        capacity = modified_knapsack["data"].get("capacity")
                        item_names = modified_knapsack["data"].get("item_names",
                                                                   [f"Item {i + 1}" for i in range(len(values))])
                        try:
                            optimal_value, chosen_items = solve_knapsack(values, weights, capacity, item_names)
                            formatted = f"Optimal Knapsack Value: {optimal_value}\nChosen Items: {', '.join(chosen_items)}"
                            human_response = humanize_response(formatted, "knapsack problem")
                            st.session_state.session_problem = modified_knapsack
                            st.session_state.history.append(("assistant", human_response))
                            st.success(f"🎯 Optimal Value: {optimal_value}")
                            display_knapsack_solution(values, weights, item_names, chosen_items)
                        except Exception as e:
                            st.session_state.history.append(("assistant", f"❌ Error solving knapsack: {e}"))

                elif st.session_state.problem_type == "combinatorial_assignment":
                    modified_assignment, error = modify_combinatorial(st.session_state.session_problem, user_input)
                    if error or modified_assignment is None:
                        st.session_state.history.append(("assistant", f"❌ {error}"))
                    else:
                        cost_matrix = np.array(modified_assignment["data"].get("cost_matrix", []))
                        result = solve_assignment_problem(cost_matrix)
                        if isinstance(result, str):
                            st.session_state.history.append(("assistant", f"❌ {result}"))
                        else:
                            cost, assignment = result
                            formatted = format_assignment_solution(cost, assignment)
                            human_response = humanize_response(formatted, "assignment problem")
                            st.session_state.session_problem = modified_assignment
                            st.session_state.history.append(("assistant", human_response))

                elif st.session_state.problem_type == "combinatorial_tsp":
                    modified_tsp, error = modify_combinatorial(st.session_state.session_problem, user_input)
                    if error or modified_tsp is None:
                        st.session_state.history.append(("assistant", f"❌ {error}"))
                    else:
                        distance_matrix = np.array(modified_tsp["data"].get("distance_matrix", []))
                        result = solve_tsp(distance_matrix)
                        if isinstance(result, str):
                            st.session_state.history.append(("assistant", f"❌ {result}"))
                        else:
                            distance, path = result
                            formatted = format_tsp_solution(distance, path)
                            human_response = humanize_response(formatted, "traveling salesman problem")
                            st.session_state.session_problem = modified_tsp
                            st.session_state.history.append(("assistant", human_response))

                elif st.session_state.problem_type == "set_covering":
                    modified_set_covering, error = modify_combinatorial(st.session_state.session_problem, user_input)
                    if error or modified_set_covering is None:
                        st.session_state.history.append(("assistant", f"❌ {error}"))
                    else:
                        sets = modified_set_covering["data"].get("sets", [])
                        universe = modified_set_covering["data"].get("universe", [])
                        if not sets or not universe:
                            st.session_state.history.append(("assistant",
                                                             "❌ Error: Set covering problem requires both 'sets' and 'universe' data."))
                        else:
                            result = solve_set_covering(sets, universe)
                            if isinstance(result, str):
                                st.session_state.history.append(("assistant", f"❌ {result}"))
                            else:
                                cover_indices, cost = result
                                formatted = format_set_covering_solution(cover_indices, cost, sets)
                                human_response = humanize_response(formatted, "set covering problem")
                                st.session_state.session_problem = modified_set_covering
                                st.session_state.history.append(("assistant", human_response))

        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            st.session_state.history.append(("user", user_input))
            st.session_state.history.append(("assistant", f"❌ OpenAI Error: {e}"))
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            st.session_state.history.append(("user", user_input))
            st.session_state.history.append(("assistant", f"❌ Unexpected error: {e}"))
# --- Display Chat History ---
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)


