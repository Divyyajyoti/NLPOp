#Prototype_10043_extended.py
#Humanised output, optimization type identification, and support for linear and non-linear optimization

import streamlit as st
import google.generativeai as genai
from scipy.optimize import linprog, minimize
import numpy as np
import os
import re

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
    "multi_objective_optimization": False
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
            return detected_types
    except:
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
        return f"(⚠️ Could not generate humanized response: {e})\n\n{technical_output}"

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

def display_constraints_lpp(lpp_dict):
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

def format_lpp_solution(opt_val, opt_vars, objective, lpp_dict):
    if opt_val is None or opt_vars is None:
        return "No feasible solution found."

    var_names = lpp_dict.get('variable_names') or [f"x{i+1}" for i in range(len(opt_vars))]
    var_details = "\n".join([f"  - {name}: {val:.2f}" for name, val in zip(var_names, opt_vars)])

    summary = ""
    if objective == 'maximize' and lpp_dict.get('c_max'):
        terms = [f"{round(coef, 2)}{var_names[i]}" for i, coef in enumerate(lpp_dict['c_max'])]
        summary += "**Objective Function:** Maximize Z = " + " + ".join(terms) + "\n"
    elif objective == 'minimize' and lpp_dict.get('c_min'):
        terms = [f"{round(coef, 2)}{var_names[i]}" for i, coef in enumerate(lpp_dict['c_min'])]
        summary += "**Objective Function:** Minimize Z = " + " + ".join(terms) + "\n"
    elif objective == 'mixed':
        c_max = lpp_dict.get('c_max')
        c_min = lpp_dict.get('c_min')
        if c_max and c_min:
            terms = [f"(α*-{x} + (1-α)*{y}){var_names[i]}" for i, (x, y) in enumerate(zip(c_max, c_min))]
            summary += "**Objective Function:** Mixed = " + " + ".join(terms) + "\n"

    summary += "\n**Constraints:**\n" + display_constraints_lpp(lpp_dict) + "\n\n"
    result_text = f"Optimal Value: **{opt_val:.2f}**\n\nVariable Values:\n{var_details}"

    return summary + result_text

def modify_lpp(session_lpp, user_input):
    model = genai.GenerativeModel('models/learnlm-1.5-pro-experimental')
    prompt = f"""
    You are assisting in modifying a Linear Programming Problem (LPP). Here is the existing LPP in dictionary format:

    {session_lpp}

    Based on this user instruction:
    "{user_input}"

    Return an updated version of the dictionary **with only the necessary changes made**.
    DO NOT remove or omit any fields from the original unless asked explicitly. Maintain structure integrity.

    Return ONLY the Python dictionary with changes implemented. No explanation or extra text.
    """
    response = model.generate_content(prompt)
    try:
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return eval(match.group(0)), None
        else:
            return None, "Failed to parse modified LPP."
    except Exception as e:
        return None, f"Error parsing modified LPP: {e}"

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

    var_names = nlp_dict.get('variable_names') or [f"x{i+1}" for i in range(len(opt_vars))]
    var_details = "\n".join([f"  - {name}: {val:.4f}" for name, val in zip(var_names, opt_vars)])
    objective_str = nlp_dict.get('objective_function_original', "Objective Function") # Consider storing original string

    summary = f"**Objective Function:** {objective_str} ({objective})\n\n"
    if nlp_dict.get('constraint_descriptions'):
        summary += "**Constraints:**\n" + "\n".join([f"- {desc}" for desc in nlp_dict['constraint_descriptions']]) + "\n\n"

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
        return None, f"Error parsing modified NLP: {e}"

# --- Streamlit App ---
st.set_page_config(page_title="Optimizer Chat", layout="wide")
st.title("⚙️ Optimization Problem Solver (Linear & Non-Linear)")
st.markdown("Describe your optimization problem in natural language and let AI extract, solve, and refine it interactively.")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'session_problem' not in st.session_state:
    st.session_state.session_problem = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None  # Initialize problem_type

user_input = st.chat_input("Enter a new optimization problem description or a follow-up modification...")

if user_input:
    with st.spinner("Processing..."):
        opt_types = detect_optimization_type(user_input)
        is_nonlinear = opt_types.get("nonlinear_programming", False)

        if st.session_state.session_problem is None:
            if not is_nonlinear:
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
                            st.session_state.history.append(("assistant", f"📚 **Detected Optimization Types:** {', '.join(true_types)}"))
                        else:
                            st.session_state.history.append(("assistant", "📚 **Detected Optimization Types:** None detected"))

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
                        nlp_data['objective_function_original'] = re.search(r"lambda x: (.+)", nlp_data.get('objective_function', ''))[1] if nlp_data.get('objective_function') else "Objective Function"
                        formatted = format_nlp_solution(opt_val, opt_vars, nlp_data.get("objective_type", "minimize"), nlp_data)
                        human_response = humanize_response(formatted, "non-linear programming")
                        st.session_state.session_problem = nlp_data
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))
                        true_types = [key.replace("_", " ").title() for key, val in opt_types.items() if val]
                        if true_types:
                            st.session_state.history.append(("assistant", f"📚 **Detected Optimization Types:** {', '.join(true_types)}"))
                        else:
                            st.session_state.history.append(("assistant", "📚 **Detected Optimization Types:** None detected"))

        else:
            if st.session_state.problem_type == "linear":
                modified_lpp, error = modify_lpp(st.session_state.session_problem, user_input)
                if error:
                    st.session_state.history.append(("user", user_input))
                    st.session_state.history.append(("assistant", f"❌ {error}"))
                else:
                    err2, opt_val, opt_vars = solve_lpp(modified_lpp)
                    if err2:
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", f"❌ {err2}"))
                    else:
                        formatted = format_lpp_solution(opt_val, opt_vars, modified_lpp.get("objective"), modified_lpp)
                        human_response = humanize_response(formatted, "linear programming")
                        st.session_state.session_problem = modified_lpp
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))
            elif st.session_state.problem_type == "nonlinear":
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
                        formatted = format_nlp_solution(opt_val, opt_vars, modified_nlp.get("objective_type", "minimize"), modified_nlp)
                        human_response = humanize_response(formatted, "non-linear programming")
                        st.session_state.session_problem = modified_nlp
                        st.session_state.history.append(("user", user_input))
                        st.session_state.history.append(("assistant", human_response))

# --- Chat Display ---
for sender, message in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(message)