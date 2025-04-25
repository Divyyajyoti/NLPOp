# Prototype_10043_extended.py
# Humanised output, optimization type identification, and support for linear and non-linear optimization

import streamlit as st
import google.generativeai as genai
from scipy.optimize import linprog, minimize, linear_sum_assignment
import numpy as np
import os
import re
from itertools import permutations
from collections import defaultdict

# --- Configure Google GenAI ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Set it as an environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

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
    "set_covering": False  # Added set covering
}


def detect_optimization_type(problem_statement):
    prompt = f"""
    You are an expert in optimization theory. Read the following problem statement and determine which types of optimization problems it involves.

    ---
    {problem_statement}
    ---

    Return a Python dictionary in the following format, setting only the relevant types to True:

    {OPT_TYPES}

    Respond ONLY with the dictionary. Do not include explanation.
    """
    model = genai.GenerativeModel('models/learnlm-1.5-pro-experimental')
    try:
        response = model.generate_content(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            detected_types = eval(match.group(0))
            # Basic heuristic: if the problem mentions non-linear terms. Improve this for better accuracy.
            if re.search(r'[a-zA-Z]\^\d+|[a-zA-Z]+\*[a-zA-Z]+|sin\(|cos\(|exp\(|log\(', problem_statement):
                detected_types["nonlinear_programming"] = True
            if "set cover" in problem_statement.lower() or "set covering" in problem_statement.lower():
                detected_types["set_covering"] = True
            return detected_types
    except Exception as e:
        print(f"Error in detect_optimization_type: {e}")  # Add error logging
        pass
    return OPT_TYPES  # fallback


# --- Humanize Response Function ---
def humanize_response(technical_output, problem_type="optimization"):
    model = genai.GenerativeModel('models/learnlm-1.5-pro-experimental')
    prompt = f"""
    You are a formal mathematical assistant. The following is a technical explanation of a {problem_type} solution:

    ---
    {technical_output}
    ---

    Rewrite this in natural language to help a user understand what the solution means in simple terms.
    Highlight the optimal value, key variables, and what they should take away from it. Be brief, helpful, and conversational.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error in humanize_response: {e}")  # Add error logging
        return f"(⚠️ Could not generate humanized response: {e})\n\n{technical_output}"


# --- Combinatorial Optimization Functions ---
def solve_knapsack(values, weights, capacity):
    """
    Solves the 0/1 knapsack problem using dynamic programming.
    values: List of item values
    weights: List of item weights
    capacity: Maximum weight capacity of the knapsack
    """
    n = len(values)
    dp = np.zeros((n + 1, capacity + 1))

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # Extract the optimal solution
    optimal_value = dp[n][capacity]
    chosen_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen_items.append(i - 1)
            w -= weights[i - 1]

    return optimal_value, chosen_items[::-1]


def solve_tsp(distance_matrix):
    """
    Solves the Traveling Salesman Problem (TSP) using a brute force approach.
    Represents no direct path with infinity (np.inf).
    distance_matrix: A 2D array where distance_matrix[i][j] is the distance between city i and city j.
                     Use np.inf to represent no direct path.
    """
    n = len(distance_matrix)

    # Check if the distance matrix is square
    if len(distance_matrix) != len(distance_matrix[0]):
        return "Error: Distance matrix must be square.", None
    # Check if the distance matrix is empty
    if n == 0:
        return "Error: Distance matrix cannot be empty.", None

    # Check if the distance matrix has only one city
    if n == 1:
        return 0, [0]

    if n > 10:
        return "Warning: TSP instance is relatively large for brute force. Solution might take a very long time.", None

    cities = list(range(n))
    min_distance = float('inf')
    best_path = None

    for perm in permutations(cities):
        current_distance = 0
        valid_path = True
        for i in range(n):
            current_city = perm[i]
            next_city = perm[(i + 1) % n]  # Cycle back to the start

            distance = distance_matrix[current_city][next_city]
            if np.isinf(distance):
                valid_path = False
                break
            current_distance += distance

        if valid_path and current_distance < min_distance:
            min_distance = current_distance
            best_path = perm

    if min_distance == float('inf'):
        return "Error: No valid TSP tour found (possibly due to disconnected cities or all paths being infinite).", None

    return min_distance, list(best_path)


def solve_assignment_problem(cost_matrix):
    """
    Solves the Assignment Problem using the Hungarian algorithm (implemented in scipy).
    cost_matrix: A 2D array where cost_matrix[i][j] is the cost of assigning worker i to job j.
    """
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        optimal_cost = cost_matrix[row_ind, col_ind].sum()
        assignment = list(zip(row_ind, col_ind))  # List of (worker, job) pairs
        return optimal_cost, assignment
    except Exception as e:
        return f"Error solving assignment problem: {e}", None


def solve_set_covering(sets, universe):
    """
    Solves the Set Covering problem using a greedy approximation algorithm.

    Args:
        sets: A list of sets, where each set is a list of elements from the universe.
        universe: The set of all elements that need to be covered.

    Returns:
        A tuple containing:
            - A list of the indices of the sets chosen for the cover.
            - The cost of the cover (number of sets chosen).
    """
    uncovered = set(universe)
    cover = []
    set_indices = list(range(len(sets)))  # Keep track of original indices

    while uncovered:
        best_set_index = -1
        max_elements_covered = 0

        for i, s in enumerate(sets):
            elements_covered = len(set(s) & uncovered)
            if elements_covered > max_elements_covered:
                max_elements_covered = elements_covered
                best_set_index = i

        if best_set_index == -1:
            return "Error: No set covers the remaining elements.", None

        cover.append(best_set_index)
        uncovered -= set(sets[best_set_index])

    return cover, len(cover)


# --- Linear Programming Functions ---
def extract_lpp_from_text(text):
    model = genai.GenerativeModel('models/learnlm-1.5-pro-experimental')
    prompt = f"""
    You are a world-class mathematical assistant designed to extract structured Linear Programming Problems (LPPs) from natural language.

    ---
    Your task involves **three stages**: ... (rest of the LPP extraction prompt)
    """
    response = model.generate_content(prompt)
    try:
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return eval(match.group(0)), None
        else:
            return None, "Invalid LPP format returned."
    except Exception as e:
        print(f"Error in extract_lpp_from_text: {e}")  # Add error logging
        return None, f"Error parsing LPP: {e}"


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
        else:
            return f"LPP solving failed: {res.message}", None, None
    except ValueError as e:
        return f"Error solving LPP: {e}", None, None


def format_lpp_solution(opt_val, opt_vars, objective, lpp_dict):
    """Formats the LPP solution into a human-readable string."""
    var_names = lpp_dict.get('variable_names') or [f"x{i + 1}" for i in range(len(opt_vars))]
    var_details = "\n".join([f"  - {name}: {val:.4f}" for name, val in zip(var_names, opt_vars)])

    summary = f"**Objective:** {objective}\n\n"
    result_text = f"Optimal Value: **{opt_val:.4f}**\n\nVariable Values:\n{var_details}"
    return summary + result_text


# --- Non-Linear Programming Functions ---

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
    \"\"\"{problem_statement}\"\"\"

    ---
    Return ONLY a Python dictionary in the specified format. Ensure that 'objective_function' and the 'fun' part of 'constraints' are strings representing lambda functions. Do not include any explanation or extra text.
    """
    model = genai.GenerativeModel('models/learnlm-1.5-pro-experimental')
    try:
        response = model.generate_content(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return eval(match.group(0)), None
        else:
            return None, "Invalid NLP format returned."
    except Exception as e:
        print(f"Error in extract_nlp_components: {e}")  # Add error logging
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
        else:
            return f"NLP solving failed: {result.message}", None, None
    except Exception as e:
        return f"Error solving NLP: {e}", None, None


def format_nlp_solution(opt_val, opt_vars, objective, nlp_dict):
    if opt_val is None or opt_vars is None:
        return "No feasible solution found."

    var_names = nlp_dict.get('variable_names') or [f"x{i + 1}" for i in range(len(opt_vars))]
    var_details = "\n".join([f"  - {name}: {val:.4f}" for name, val in zip(var_names, opt_vars)])
    objective_str = nlp_dict.get('objective_function_original',
                                 "Objective Function")  # Consider storing original string

    summary = f"**Objective Function:** {objective_str} ({objective})\n\n"
    if nlp_dict.get('constraint_descriptions'):
        summary += "**Constraints:**\n" + "\n".join(
            [f"- {desc}" for desc in nlp_dict['constraint_descriptions']]) + "\n\n"

    result_text = f"Optimal Value: **{opt_val:.4f}**\n\nVariable Values:\n{var_details}"
    return summary + result_text


def modify_nlp(session_nlp, user_input):
    model = genai.GenerativeModel('models/learnlm-1.5-pro-experimental')
    prompt = f"""
    You are assisting in modifying a Non-Linear Programming (NLP) problem. Here is the existing NLP in dictionary format:

    {session_nlp}

    Based on this user instruction:
    "{user_input}"

    Return an updated version of the dictionary **with only the necessary changes made**.
    Ensure that the 'objective_function' and constraint 'fun' values remain as Python-executable strings (lambda functions).
    DO NOT remove or omit any fields from the original unless asked explicitly. Maintain structure integrity.

    Return ONLY the Python dictionary with changes implemented. No explanation or extra text.
    """
    response = model.generate_content(prompt)
    try:
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return eval(match.group(0)), None
        else:
            return None, "Failed to parse modified NLP."
    except Exception as e:
        print(f"Error in modify_nlp: {e}")  # Add error logging
        return None, f"Error parsing modified NLP: {e}"


# --- Streamlit App ---
st.set_page_config(page_title="Optimizer Chat", layout="wide")
st.title("⚙️ Optimization Problem Solver (Linear & Non-Linear)")
st.markdown(
    "Describe your optimization problem in natural language and let AI extract, solve, and refine it interactively.")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'session_problem' not in st.session_state:
    st.session_state.session_problem = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None  # Initialize problem_type

user_input = st.chat_input("Enter a new optimization problem description or a follow-up modification...")


def format_tsp_solution(distance, path):
    """Formats the TSP solution into a human-readable string."""
    path_str = " -> ".join(map(str, path))
    return f"Optimal TSP Path: {path_str}\nTotal Distance: {distance:.2f}"


def format_assignment_solution(cost, assignment):
    """Formats the Assignment Problem solution into a human-readable string."""
    assignment_str = "\n".join([f"Worker {w} assigned to Job {j}" for w, j in assignment])
    return f"Optimal Assignment:\n{assignment_str}\nTotal Cost: {cost:.2f}"


def format_set_covering_solution(cover_indices, cost, sets):
    """Formats the Set Covering Problem solution into a human-readable string."""
    if isinstance(cover_indices, str):
        return cover_indices  # Return the error message

    covered_sets = [f"Set {i + 1}: {sets[i]}" for i in cover_indices]  # +1 for 1-based indexing
    sets_used = ", ".join(map(str, [i + 1 for i in cover_indices]))
    return f"Sets chosen: {sets_used}\nCovered Sets:\n" + "\n".join(covered_sets) + f"\nTotal Sets (Cost): {cost}"


def extract_combinatorial_data(problem_statement):
    """
    Uses AI to extract the problem type and data for combinatorial optimization.
    """
    model = genai.GenerativeModel('models/learnlm-1.5-pro-experimental')
    prompt = f"""
    You are an expert in combinatorial optimization. Your task is to analyze the following problem statement and:

    1.  Identify the problem type: Determine if it's a Traveling Salesman Problem (TSP), Assignment Problem, Knapsack Problem, or Set Covering Problem.
    2.  Extract the data: Extract the relevant data for the identified problem type. This might include a distance matrix (for TSP), a cost matrix (for Assignment), 
        values, weights, and capacity (for Knapsack), or a list of sets and the universe (for Set Covering).

    ---
    Problem Statement:
    \"\"\"{problem_statement}\"\"\"

    ---

    Respond with a Python dictionary in the following format:

    {{
        "problem_type": "tsp" or "assignment" or "knapsack" or "set_covering" or "unknown",
        "data": {{
            "distance_matrix": [[...], [...], ...],
            "cost_matrix": [[...], [...], ...],
            "values": [...], "weights": [...], "capacity": ...,
            "sets": [[...], [...], ...],  # List of sets
            "universe": [...]             # The universe of elements
        }},
        "error": None
    }}

    If you cannot reliably identify the problem type or extract the data, set "problem_type" to "unknown" and provide an error message.  
    The data should be a valid Python list of lists (for matrices) or a list/number as appropriate.  Do not include any explanation or extra text.
    """
    try:
        response = model.generate_content(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return eval(match.group(0))
        else:
            return {"problem_type": "unknown", "data": {}, "error": "Failed to parse AI response."}
    except Exception as e:
        print(f"Error in extract_combinatorial_data: {e}")  # Add error logging
        return {"problem_type": "unknown", "data": {}, "error": f"Error during AI processing: {e}"}


if user_input:
    with st.spinner("Processing..."):
        opt_types = detect_optimization_type(user_input)
        is_nonlinear = opt_types.get("nonlinear_programming", False)
        is_combinatorial = opt_types.get("combinatorial_optimization", False)
        is_set_covering = opt_types.get("set_covering", False)  # check for set cover

        if st.session_state.session_problem is None:
            if is_combinatorial:
                # Use AI to extract the problem type and data
                combinatorial_data = extract_combinatorial_data(user_input)

                if combinatorial_data["problem_type"] == "tsp":
                    distance_matrix = np.array(combinatorial_data["data"].get("distance_matrix", []))
                    result = solve_tsp(distance_matrix)

                    if isinstance(result, str):
                        error = result
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"❌ {error}"))
                    else:
                        distance, path = result
                        formatted_solution = format_tsp_solution(distance, path)
                        human_response = humanize_response(formatted_solution, "traveling salesman problem")
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))

                elif combinatorial_data["problem_type"] == "assignment":
                    cost_matrix = np.array(combinatorial_data["data"].get("cost_matrix", []))
                    result = solve_assignment_problem(cost_matrix)
                    if isinstance(result, str):
                        error = result
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"❌ {error}"))
                    else:
                        cost, assignment = result
                        formatted_solution = format_assignment_solution(cost, assignment)
                        human_response = humanize_response(formatted_solution, "assignment problem")
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))

                elif combinatorial_data["problem_type"] == "knapsack":
                    values = combinatorial_data["data"].get("values", [])
                    weights = combinatorial_data["data"].get("weights", [])
                    capacity = combinatorial_data["data"].get("capacity")

                    try:
                        optimal_value, chosen_items = solve_knapsack(values, weights, capacity)
                        formatted_solution = f"Optimal Knapsack Value: {optimal_value}\nChosen Items: {chosen_items}"
                        human_response = humanize_response(formatted_solution, "knapsack problem")
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))
                    except Exception as e:
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"❌ Error solving knapsack problem: {e}"))

                elif combinatorial_data["problem_type"] == "set_covering":
                    sets = combinatorial_data["data"].get("sets", [])
                    universe = combinatorial_data["data"].get("universe", [])

                    if not sets or not universe:
                        error_message = "Error: Set Covering Problem requires both 'sets' and 'universe' data."
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"❌ {error_message}"))
                    else:
                        result = solve_set_covering(sets, universe)
                        if isinstance(result, str):
                            error_message = result
                            st.session_state.history.append(("user", user_input))
                            st.session_state.history.append(("assistant", f"❌ {error_message}"))
                        else:
                            cover_indices, cost = result
                            formatted_solution = format_set_covering_solution(cover_indices, cost, sets)
                            human_response = humanize_response(formatted_solution, "set covering problem")
                            st.session_state.history.append(("user", user_input))
                            st.session_state.history.append(("assistant", human_response))

                else:
                    st.session_state.history.append(("user", user_input))
                    st.session_state.history.append(("assistant",
                                                     f"❌ Could not identify combinatorial problem type.  Error: {combinatorial_data['error']}"))

            elif not is_nonlinear:
                st.session_state.problem_type = "linear"
                lpp_data, error = extract_lpp_from_text(user_input)
                if error:
                    st.session_state.history.append(("user", user_input))
                    st.session_state.history.append(("assistant", f"❌ {error}"))
                else:
                    err2, opt_val, opt_vars = solve_lpp(lpp_data)
                    if err2:
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"❌ {err2}"))
                    else:
                        formatted = format_lpp_solution(opt_val, opt_vars, lpp_data.get("objective"), lpp_data)
                        human_response = humanize_response(formatted, "linear programming")
                        st.session_state.session_problem = lpp_data
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))
                        true_types = [key.replace("_", " ").title() for key, val in opt_types.items() if val]
                        if true_types:
                            st.session_state.history.append(
                                ("assistant", f"📚 **Detected Optimization Types:** {', '.join(true_types)}"))
                        else:
                            st.session_state.history.append(
                                ("assistant", "📚 **Detected Optimization Types:** None detected"))

            else:
                st.session_state.problem_type = "nonlinear"
                nlp_data, error = extract_nlp_components(user_input)
                if error:
                    st.session_state.history.append(("user", user_input))
                    st.session_state.history.append(("assistant", f"❌ {error}"))
                else:
                    err2, opt_val, opt_vars = solve_nlp(nlp_data)
                    if err2:
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"❌ {err2}"))
                    else:
                        # Store the original objective function string for better output
                        nlp_data['objective_function_original'] = \
                        re.search(r"lambda x: (.+)", nlp_data.get('objective_function', ''))[1] if nlp_data.get(
                            'objective_function') else "Objective Function"
                        formatted = format_nlp_solution(opt_val, opt_vars, nlp_data.get("objective_type", "minimize"),
                                                        nlp_data)
                        human_response = humanize_response(formatted, "non-linear programming")
                        st.session_state.session_problem = nlp_data
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))
                        true_types = [key.replace("_", " ").title() for key, val in opt_types.items() if val]
                        if true_types:
                            st.session_state.history.append(
                                ("assistant", f"📚 **Detected Optimization Types:** {', '.join(true_types)}"))
                        else:
                            st.session_state.history.append(
                                ("assistant", "📚 **Detected Optimization Types:** None detected"))

        else:
            # Modification logic (not implemented for combinatorial yet)
            if st.session_state.problem_type == "nonlinear":
                modified_nlp, error = modify_nlp(st.session_state.session_problem, user_input)
                if error:
                    st.session_state.history.append(("user", user_input))
                    st.session_state.history.append(("assistant", f"❌ {error}"))
                else:
                    err2, opt_val, opt_vars = solve_nlp(modified_nlp)
                    if err2:
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"❌ {err2}"))
                    else:
                        modified_nlp['objective_function_original'] = \
                        re.search(r"lambda x: (.+)", modified_nlp.get('objective_function', ''))[1] if modified_nlp.get(
                            'objective_function') else "Objective Function"
                        formatted = format_nlp_solution(opt_val, opt_vars,
                                                        modified_nlp.get("objective_type", "minimize"), modified_nlp)
                        human_response = humanize_response(formatted, "non-linear programming")
                        st.session_state.session_problem = modified_nlp
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))
            elif st.session_state.problem_type == "linear":
                # Implement modification logic for linear problems here
                st.session_state.history.append(("user", user_input))
                st.session_state.history.append(
                    ("assistant", "❌ Modification of linear problems is not yet fully implemented."))
            else:
                st.session_state.history.append(("user", user_input))
                st.session_state.history.append(
                    ("assistant", "❌ Modification of combinatorial problems is not yet implemented."))

# --- Chat Display ---
for sender, message in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(message)
