import streamlit as st
import openai
from scipy.optimize import linprog, minimize, linear_sum_assignment
from itertools import permutations
import re
import numpy as np
import pandas as pd
import ast
import logging
import uuid
import networkx as nx
from openai import OpenAI

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)

OPENAI_API_KEY = "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"
OPENAI_PROJECT_ID = "proj_bb1YRYbP5wd4P8quGgIrXOsd"
OPENAI_ORG_ID = "org-adVrGVntY5ftjNinDvOf53Ku"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID,
)

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
    "stochastic_programming": False,
    "network_optimization": False
}

count = []

def safe_eval(response_text):
    """Safely evaluate a string response by replacing 'true'/'false' with Python booleans."""
    response_text = response_text.replace('true', 'True').replace('false', 'False')
    return eval(response_text)

def detect_optimization_type(problem_statement):
    prompt = f"""
    You are an expert in optimization theory. Read the following problem statement and determine which types of optimization problems it involves.

    ---
    {problem_statement}
    ---

    In addition to standard categories, pay special attention to:
    - Set Covering Problems: Involving a universe of elements and sets, aiming to cover all elements with minimal sets. Toggle 'set_covering' to True if detected.
    - Stochastic Programming: Problems with uncertainty (e.g., random demand, uncertain costs). Set "stochastic_programming": True if detected.
    - Network Optimization: Problems involving flows, shortest paths, or minimum cost flows in a network. Set "network_optimization": True if detected.

    Return a Python dictionary in the format:
    {OPT_TYPES}

    Use Python boolean values `True` and `False` (uppercase), not `true` or `false`.
    Respond ONLY with the dictionary.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            detected_types = safe_eval(match.group(0))
            if re.search(r'[a-zA-Z]\^\d+|[a-zA-Z]+\*[a-zA-Z]+|sin\(|cos\(|exp\(|log\(', problem_statement):
                detected_types["nonlinear_programming"] = True
            if "set cover" in problem_statement.lower() or "set covering" in problem_statement.lower():
                detected_types["set_covering"] = True
            if "network" in problem_statement.lower() or "flow" in problem_statement.lower() or "shortest path" in problem_statement.lower():
                detected_types["network_optimization"] = True
            count.append(response.usage.total_tokens)
            return detected_types
    except Exception as e:
        logging.error(f"Error in detect_optimization_type: {e}")
        return OPT_TYPES

# --- Humanize Response ---
def humanize_response(technical_output, problem_type="optimization"):
    prompt = f"""
    You are a formal mathematical assistant. The following is a technical explanation of a {problem_type} solution:

    ---
    {technical_output}
    ---

    Rewrite this in natural language to help a user understand the solution in simple terms.
    Highlight the optimal value, key variables, and key takeaways. Be brief, helpful, and conversational.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        count.append(response.usage.total_tokens)
        if len(count) > 100:
            count[:] = count[-50:]
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(⚠️ Could not generate humanized response: {e})\n\n{technical_output}"

# --- Simplify Math Expressions ---
def simplify_math_expressions(text):
    pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)\b')
    return pattern.sub(lambda m: str(float(m.group(1)) * float(m.group(2))), text)

# --- Generic Problem Extraction ---
def extract_problem(text, problem_type):
    if problem_type == "linear_programming":
        return extract_lpp_from_text(text)
    elif problem_type == "combinatorial_optimization":
        return extract_combinatorial_assignment_from_text(text)
    elif problem_type == "network_optimization":
        return extract_network_flow_from_text(text)
    else:
        prompt = f"""
        You are a mathematical assistant designed to extract {problem_type.replace('_', ' ')} problems from natural language.

        ---
        Input:
        \"\"\"{text}\"\"

        ---
        Output Format:
        A dictionary containing the problem components (e.g., objective, constraints, variables).
        If the problem type is not fully defined, return a dictionary with at least a "description" field.

        Use Python boolean values `True` and `False` (uppercase), not `true` or `false`.
        Only return the dictionary.
        """
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        count.append(response.usage.total_tokens)
        try:
            match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
            if match:
                parsed_dict = safe_eval(match.group(0))
                if not parsed_dict.get("description"):
                    parsed_dict["description"] = text
                return parsed_dict, None
            return None, f"Invalid {problem_type} format returned."
        except Exception as e:
            logging.error(f"Error parsing {problem_type}: {e}")
            return None, None

# --- Extract LPP from Text ---
def extract_lpp_from_text(text):
    prompt = f"""
    You are a mathematical assistant designed to extract Linear Programming Problems (LPPs) from natural language.

    ---
    Your task involves three stages:

    Stage 1: Unit Standardization
    - Convert all quantities into SI units.
    - Rewrite the problem text using SI units for internal processing.
    - Before final output, revert to original units.

    Stage 2: LPP Extraction
    From the text below, extract:
    - Objective function coefficients for both `maximize` and `minimize` objectives.
    - Inequality constraints: matrix `A_ub`, vector `b_ub`
    - Equality constraints: matrix `A_eq`, vector `b_eq`
    - Variable bounds: list of `(lower, upper)` tuples. Default: `(0, None)`
    - Variable names: e.g., `["x1", "x2", ...]`
    - Constraint names: e.g., `["Constraint 1", "Constraint 2", ...]`
    - Objective type: `"maximize"`, `"minimize"`, or `"weighted"`

    Stage 3: Matrix Verification
    - Verify matrices for consistency five times.

    ---
    Input:
    ```{text}
    ```

    ---
    Output Format:
    {{
        "c_max": [float, ...],
        "c_min": [float, ...],
        "A_ub": [[float, ...], ...],
        "b_ub": [float, ...],
        "A_eq": [[float, ...], ...] or None,
        "b_eq": [float, ...] or [],
        "bounds": [(float, float or None), ...],
        "objective": "maximize" or "minimize" or "weighted",
        "variable_names": ["x1", "x2", ...],
        "constraint_names": ["Constraint 1", "Constraint 2", ...]
    }}

    Use Python boolean values `True` and `False` (uppercase).
    Only return the dictionary.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            raw_dict = simplify_math_expressions(match.group(0))
            parsed_dict = safe_eval(raw_dict)
            return parsed_dict, None
        return None, "Invalid LPP format returned."
    except Exception as e:
        logging.error(f"Error parsing LPP: {e}")
        return None, None

# --- Extract Combinatorial Assignment from Text ---
def extract_combinatorial_assignment_from_text(text):
    prompt = f"""
    You are a mathematical assistant designed to extract Assignment Problems (a type of combinatorial optimization) from natural language.

    An Assignment Problem typically involves:
    - Assigning `n` agents to `n` tasks.
    - A cost matrix where `cost[i][j]` is the cost of assigning agent `i` to task `j`.
    - The goal is to minimize the total cost of assignments.

    From the text below, extract:
    - Cost matrix: A 2D list `[[cost_11, cost_12, ...], [cost_21, cost_22, ...]`
    - Agent names: e.g., `["Agent1", "Agent2", ...]`
    - Task names: e.g., `["Task1", "Task2", ...]`

    ---
    Input:
    ```{text}
    ```

    ---
    ```
    Output Format:
    {{
        "cost_matrix": [[float, ...], ...],
        "agents": ["Agent1", "Agent2", ...],
        "tasks": ["Task1", "Task2", ...]
    }}

    Use Python boolean values `True` and `False` (uppercase).
    Only return the dictionary.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            parsed_dict = safe_eval(match.group(0))
            return parsed_dict, None
        return None, "Invalid assignment problem format returned."
    except Exception as e:
        logging.error(f"Error parsing assignment problem: {e}")
        return None, None

# --- Extract Network Flow from Text ---
def extract_network_flow_from_text(text):
    prompt = f"""
    You are a mathematical assistant designed to extract Minimum Cost Flow Problems (a type of network optimization) from natural language.

    A Minimum Cost Flow Problem typically involves:
    - A directed graph with nodes and edges.
    - Each edge has a capacity and a cost per unit of flow.
    - Supply/demand at nodes (positive for supply, negative for demand).
    - The goal is to minimize the total cost of sending flow from supply nodes to demand nodes.

    From the text below, extract:
    - Nodes: List of node names, e.g., `["A", "B", "C"]`
    - Edges: List of tuples `(source, target, capacity, cost)`, e.g., `[("A", "B", 10, 2), ...]`
    - Supply/Demand: Dictionary mapping nodes to their supply/demand, e.g., `{"A": 20, "B": -10, "C": -10}`

    ---
    Input:
    ```{text}
    ```

    ---
    Output Format:
    {{
        "nodes": ["A", "B", ...],
        "edges": [["source", "target", capacity, cost], ...],
        "supply_demand": {{"node1": supply1, "node2": demand2, ...}}
    }}

    Use Python boolean values `True` and `False` (uppercase).
    Only return the dictionary.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            parsed_dict = safe_eval(match.group(0))
            return parsed_dict, None
        return None, "Invalid network flow problem format returned."
    except Exception as e:
        logging.error(f"Error parsing network flow problem: {e}")
        return None, None

# --- Handle Missing Values ---
def ask_user_to_fill_missing_values(problem_dict, problem_type):
    updated = False
    missing_fields = []
    for key, value in problem_dict.items():
        if value is None or (isinstance(value, list) and not value) or (isinstance(value, dict) and not value):
            missing_fields.append(key)
            logging.debug(f"Missing field detected: {key}")

    if missing_fields:
        st.subheader("🛠 Help Complete the Optimization Problem")
        with st.form(key=f"missing_values_form_{uuid.uuid4()}"):
            for field in missing_fields:
                if problem_type == "linear_programming" and field == "A_eq" and (problem_dict.get("c_max") or problem_dict.get("c_min")):
                    num_vars = len(problem_dict.get("c_max") or problem_dict.get("c_min"))
                    num_eq = st.number_input(f"How many equality constraints for {field}?", min_value=1, value=1, key=f"num_eq_{field}")
                    problem_dict["A_eq"] = []
                    for i in range(num_eq):
                        row = []
                        for j in range(num_vars):
                            var_name = problem_dict.get("variable_names", [f"x{j+1}" for j in range(num_vars)])[j]
                            val = st.number_input(
                                f"Coefficient for {var_name} in equality constraint {i+1}",
                                value=0.0,
                                key=f"aeq_{i}_{j}_{field}"
                            )
                            row.append(float(val))
                        problem_dict["A_eq"].append(row)
                    updated = True
                elif problem_type == "linear_programming" and field == "b_eq" and problem_dict.get("A_eq"):
                    problem_dict["b_eq"] = []
                    for i in range(len(problem_dict["A_eq"])):
                        con_name = problem_dict.get("constraint_names", [f"Constraint {i+1}" for i in range(len(problem_dict["A_eq"]))])[i]
                        val = st.number_input(
                            f"RHS value for {con_name}",
                            value=0.0,
                            key=f"beq_{i}_{field}"
                        )
                        problem_dict["b_eq"].append(float(val))
                    updated = True
                elif problem_type == "combinatorial_optimization" and field == "cost_matrix":
                    num_agents = len(problem_dict.get("agents", [])) or st.number_input("Number of agents", min_value=1, value=3, key="num_agents")
                    num_tasks = len(problem_dict.get("tasks", [])) or st.number_input("Number of tasks", min_value=1, value=3, key="num_tasks")
                    problem_dict["cost_matrix"] = []
                    for i in range(num_agents):
                        row = []
                        for j in range(num_tasks):
                            val = st.number_input(
                                f"Cost of assigning Agent {i+1} to Task {j+1}",
                                value=0.0,
                                key=f"cost_{i}_{j}"
                            )
                            row.append(float(val))
                        problem_dict["cost_matrix"].append(row)
                    updated = True
                elif problem_type == "network_optimization" and field == "supply_demand":
                    nodes = problem_dict.get("nodes", [])
                    problem_dict["supply_demand"] = {}
                    for node in nodes:
                        val = st.number_input(
                            f"Supply (positive) or Demand (negative) for node {node}",
                            value=0,
                            key=f"sd_{node}"
                        )
                        problem_dict["supply_demand"][node] = int(val)
                    updated = True
                elif field in ["c_max", "c_min", "b_ub", "b_eq"]:
                    num_vars = len(problem_dict.get("variable_names", [])) or st.number_input(f"Number of variables for {field}", min_value=1, value=2, key=f"num_vars_{field}")
                    problem_dict[field] = []
                    for i in range(num_vars):
                        val = st.number_input(
                            f"Value for {field} at position {i+1}",
                            value=0.0,
                            key=f"{field}_{i}"
                        )
                        problem_dict[field].append(float(val))
                    updated = True
                elif field == "edges" and problem_type == "network_optimization":
                    num_edges = st.number_input("Number of edges", min_value=1, value=1, key="num_edges")
                    problem_dict["edges"] = []
                    nodes = problem_dict.get("nodes", [])
                    for i in range(num_edges):
                        source = st.selectbox(f"Source node for edge {i+1}", nodes, key=f"source_{i}")
                        target = st.selectbox(f"Target node for edge {i+1}", nodes, key=f"target_{i}")
                        capacity = st.number_input(f"Capacity for edge {i+1}", min_value=0.0, value=1.0, key=f"cap_{i}")
                        cost = st.number_input(f"Cost for edge {i+1}", value=1.0, key=f"cost_{i}")
                        problem_dict["edges"].append([source, target, capacity, cost])
                    updated = True
                else:
                    val = st.text_input(f"Value for {field} (enter as a Python expression)", key=f"missing_{field}")
                    if val:
                        try:
                            problem_dict[field] = ast.literal_eval(val)
                            updated = True
                        except:
                            st.error(f"Invalid input for {field}. Please provide a valid Python expression (e.g., [1, 2, 3] for a list).")
            submit = st.form_submit_button("Submit Missing Values")
            if submit and updated:
                st.success("✅ Missing values updated!")
                logging.info("Missing values updated by user.")
                return problem_dict
    return problem_dict if updated else None

# --- Generic Problem Solving ---
def solve_problem(problem_dict, problem_type, alpha=0.5):
    if problem_type == "linear_programming":
        return solve_lpp(problem_dict, alpha)
    elif problem_type == "combinatorial_optimization":
        return solve_assignment_problem(problem_dict)
    elif problem_type == "network_optimization":
        return solve_minimum_cost_flow(problem_dict)
    else:
        return f"Solver for {problem_type.replace('_', ' ')} not implemented yet.", None, None

# --- Solve LPP ---
def solve_lpp(lpp_dict, alpha=0.5):
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
    elif objective == 'weighted' and c_max and c_min:
        c = [(alpha * -x) + ((1 - alpha) * y) for x, y in zip(c_max, c_min)]
    else:
        return "Unknown or invalid objective type.", None, None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            if objective == 'maximize':
                res.fun = -res.fun
            return None, res.fun, res.x
        return f"LPP solving failed: {res.message}", None, None
    except ValueError as e:
        return f"Error solving LPP: {e}", None, None

# --- Solve Combinatorial Problem (Assignment Problem) ---
def solve_assignment_problem(assignment_dict):
    cost_matrix = np.array(assignment_dict.get("cost_matrix"))
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        assignments = list(zip(row_ind, col_ind))
        return None, total_cost, assignments
    except Exception as e:
        return f"Error solving assignment problem: {e}", None, None

# --- Solve Network Problem (Minimum Cost Flow) ---
def solve_minimum_cost_flow(network_dict):
    try:
        supply_demand = network_dict.get("supply_demand", {})
        total_supply = sum(val for val in supply_demand.values() if val > 0)
        total_demand = sum(abs(val) for val in supply_demand.values() if val < 0)
        if total_supply != total_demand:
            return f"Error: Total supply ({total_supply}) does not equal total demand ({total_demand}).", None, None

        G = nx.DiGraph()
        G.add_nodes_from(network_dict.get("nodes", []))
        for edge in network_dict.get("edges", []):
            source, target, capacity, cost = edge
            G.add_edge(source, target, capacity=capacity, weight=cost)
        flow_dict = nx.min_cost_flow(G, demand=supply_demand)
        total_cost = nx.cost_of_flow(G, flow_dict)
        return None, total_cost, flow_dict
    except Exception as e:
        return f"Error solving minimum cost flow: {e}", None, None

# --- Generic Solution Formatting ---
def format_solution(opt_val, opt_vars, problem_dict, problem_type):
    if problem_type == "linear_programming":
        return format_lpp_solution(opt_val, opt_vars, problem_dict.get("objective"), problem_dict)
    elif problem_type == "combinatorial_optimization":
        return format_assignment_solution(opt_val, opt_vars, problem_dict)
    elif problem_type == "network_optimization":
        return format_network_flow_solution(opt_val, opt_vars, problem_dict)
    else:
        if opt_val is None and opt_vars is None:
            return f"Solver for {problem_type.replace('_', ' ')} not implemented yet.\nProblem Description: {problem_dict.get('description', 'No description available.')}"
        return f"Optimal Value: **{opt_val}**\nSolution: {opt_vars}"

# --- Format LPP Solution ---
def format_lpp_solution(opt_val, opt_vars, objective, lpp_dict):
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
    elif objective == 'weighted' and lpp_dict.get('c_max') and lpp_dict.get('c_min'):
        terms = [f"(α*-{x} + (1-α)*{y})x{i+1}" for i, (x, y) in enumerate(zip(lpp_dict['c_max'], lpp_dict['c_min']))]
        summary += "**Objective Function:** Weighted = " + " + ".join(terms) + "\n"

    summary += "\n**Constraints:**\n" + display_constraints(lpp_dict) + "\n\n"
    result_text = f"Optimal Value: **{opt_val:.2f}**\n\nVariable Values:\n{var_details}"
    return summary + result_text

# --- Format Assignment Solution ---
def format_assignment_solution(total_cost, assignments, assignment_dict):
    if total_cost is None or assignments is None:
        return "No feasible solution found."

    agents = assignment_dict.get("agents") or [f"Agent{i+1}" for i in range(len(assignment_dict.get("cost_matrix", [])))]
    tasks = assignment_dict.get("tasks") or [f"Task{i+1}" for i in range(len(assignment_dict.get("cost_matrix", [[]])[0]))]
    assignment_details = "\n".join([f"  - {agents[agent]} assigned to {tasks[task]}" for agent, task in assignments])
    return f"Total Cost: **{total_cost:.2f}**\n\nAssignments:\n{assignment_details}"

# --- Format Network Flow Solution ---
def format_network_flow_solution(total_cost, flow_dict, network_dict):
    if total_cost is None or flow_dict is None:
        return "No feasible solution found."

    flow_details = []
    for source in flow_dict:
        for target, flow in flow_dict[source].items():
            if flow > 0:
                flow_details.append(f"  - Flow from {source} to {target}: {flow}")
    return f"Total Cost: **{total_cost:.2f}**\n\nFlow Details:\n" + "\n".join(flow_details)

# --- Display Constraints ---
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

# --- Generic Problem Modification ---
def modify_problem(session_problem, user_input, problem_type):
    logging.info(f"Modifying problem of type {problem_type} with input: {user_input}")
    if problem_type == "linear_programming":
        return modify_lpp(session_problem, user_input)
    elif problem_type == "combinatorial_optimization":
        return modify_assignment(session_problem, user_input)
    elif problem_type == "network_optimization":
        return modify_network_flow(session_problem, user_input)
    else:
        prompt = f"""
        You are assisting in modifying a {problem_type.replace('_', ' ')} problem. Here is the existing problem dictionary:

        ```python
        {session_problem}
        ```

        Based on this user instruction:
        ```{user_input}
        ```

        Update the dictionary to reflect the requested changes (e.g., modifying variables, constraints, or parameters).
        Examples of modifications:
        - "Change x1 to 5" → Update variable bounds or constraints involving x1.
        - "Set constraint 1 to x1 + x2 <= 10" → Update A_ub and b_ub.
        - "Now cost for agent 1 to task 2 is 15" → Update cost_matrix[0][1].

        Return an updated dictionary with only the necessary changes made.
        Preserve all existing fields unless explicitly modified.
        If the instruction is unclear, return the original dictionary with an error message.

        Use Python boolean values `True` and `False` (uppercase).
        Return a tuple: (updated_dictionary, error_message or None)
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            count.append(response.usage.total_tokens)
            match = re.search(r'\((.*),\s*(None|"[^"]*")\)', response.choices[0].message.content, re.DOTALL)
            if match:
                dict_str = match.group(1)
                error = None if match.group(2) == 'None' else match.group(2).strip('"')
                updated_dict = safe_eval(dict_str)
                logging.info(f"Modification successful: {updated_dict}")
                return updated_dict, error
            return session_problem, "Failed to parse modified dictionary."
        except Exception as e:
            logging.error(f"Error modifying {problem_type}: {e}")
            return session_problem, f"Error modifying {problem_type}: {e}"

# --- Modify LPP ---
def modify_lpp(session_problem, user_input):
    prompt = f"""
    You are assisting in modifying a Linear Programming Problem (LPP). Here is the existing LPP dictionary:

    ```python
    {session_problem}
    ```

    Based on this user instruction:
    ```{user_input}
    ```

    Update the dictionary to reflect changes such as:
    - Modifying objective coefficients (c_max, c_min).
    - Changing constraints (A_ub, b_ub, A_eq, b_eq).
    - Updating variable bounds or names.
    - Examples:
      - "Change x1 to 5" → Update bounds for x1 to (5, 5).
      - "Set constraint 1 to x1 + x2 <= 10" → Update A_ub[0] and b_ub[0].
      - "Now x3 coefficient is 4 in maximize" → Update c_max[2].

    Return an updated dictionary with only the necessary changes made.
    Preserve all existing fields unless explicitly modified.
    If the instruction is unclear, return the original dictionary with an error message.

    Use Python boolean values `True` and `False` (uppercase).
    Return a tuple: (updated_dictionary, error_message or None)
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        count.append(response.usage.total_tokens)
        match = re.search(r'\((.*),\s*(None|"[^"]*")\)', response.choices[0].message.content, re.DOTALL)
        if match:
            dict_str = match.group(1)
            error = None if match.group(2) == 'None' else match.group(2).strip('"')
            updated_dict = safe_eval(dict_str)
            logging.info(f"LPP modification successful: {updated_dict}")
            return updated_dict, error
        return session_problem, "Failed to parse modified LPP."
    except Exception as e:
        logging.error(f"Error modifying LPP: {e}")
        return session_problem, f"Error modifying LPP: {e}"

# --- Modify Assignment Problem ---
def modify_assignment(session_problem, user_input):
    prompt = f"""
    You are assisting in modifying an Assignment Problem (combinatorial optimization). Here is the existing problem dictionary:

    ```python
    {session_problem}
    ```

    Based on this user instruction:
    ```{user_input}
    ```

    Update the dictionary to reflect changes such as:
    - Modifying the cost matrix (cost_matrix).
    - Changing agent or task names.
    - Examples:
      - "Now cost for agent 1 to task 2 is 15" → Update cost_matrix[0][1] to 15.
      - "Rename agent 1 to Alice" → Update agents[0] to "Alice".

    Return an updated dictionary with only the necessary changes made.
    Preserve all existing fields unless explicitly modified.
    If the instruction is unclear, return the original dictionary with an error message.

    Use Python boolean values `True` and `False` (uppercase).
    Return a tuple: (updated_dictionary, error_message or None)
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        count.append(response.usage.total_tokens)
        match = re.search(r'\((.*),\s*(None|"[^"]*")\)', response.choices[0].message.content, re.DOTALL)
        if match:
            dict_str = match.group(1)
            error = None if match.group(2) == 'None' else match.group(2).strip('"')
            updated_dict = safe_eval(dict_str)
            logging.info(f"Assignment modification successful: {updated_dict}")
            return updated_dict, error
        return session_problem, "Failed to parse modified assignment problem."
    except Exception as e:
        logging.error(f"Error modifying assignment problem: {e}")
        return session_problem, f"Error modifying assignment problem: {e}"

# --- Modify Network Flow Problem ---
def modify_network_flow(session_problem, user_input):
    prompt = f"""
    You are assisting in modifying a Minimum Cost Flow Problem (network optimization). Here is the existing problem dictionary:

    ```python
    {session_problem}
    ```

    Based on this user instruction:
    ```{user_input}
    ```

    Update the dictionary to reflect changes such as:
    - Modifying nodes, edges (capacities, costs), or supply/demand.
    - Examples:
      - "Change capacity from A to B to 15" → Update edges entry for (A, B).
      - "Set demand at node C to -20" → Update supply_demand["C"] to -20.

    Return an updated dictionary with only the necessary changes made.
    Preserve all existing fields unless explicitly modified.
    If the instruction is unclear, return the original dictionary with an error message.

    Use Python boolean values `True` and `False` (uppercase).
    Return a tuple: (updated_dictionary, error_message or None)
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        count.append(response.usage.total_tokens)
        match = re.search(r'\((.*),\s*(None|"[^"]*")\)', response.choices[0].message.content, re.DOTALL)
        if match:
            dict_str = match.group(1)
            error = None if match.group(2) == 'None' else match.group(2).strip('"')
            updated_dict = safe_eval(dict_str)
            logging.info(f"Network flow modification successful: {updated_dict}")
            return updated_dict, error
        return session_problem, "Failed to parse modified network flow problem."
    except Exception as e:
        logging.error(f"Error modifying network flow problem: {e}")
        return session_problem, f"Error modifying network flow problem: {e}"

# --- Classify User Input ---
def classify_user_input(user_input, session_problem=None):
    prompt = f"""
    You are an intelligent assistant that classifies user instructions for optimization problems.

    Given the current problem (if any):
    ```python
    {session_problem or 'None'}
    ```

    And the user's latest input:
    ```{user_input}
    ```

    Decide whether the input is:
    - A **new optimization problem** (describes a new problem with objectives/constraints).
    - A **modification or follow-up** (modifies variables, constraints, or parameters of the existing problem, e.g., "now x1 is 5", "change constraint to x1 + x2 <= 10").

    Return only one word:
    "new" → if it's a new problem
    "followup" → if it's modifying the existing one
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        count.append(response.usage.total_tokens)
        answer = response.choices[0].message.content.strip().lower()
        logging.debug(f"Input classified as: {answer}")
        return "new" if "new" in answer else "followup" if "followup" in answer else "unknown"
    except Exception as e:
        logging.error(f"Error classifying input: {e}")
        return None

# --- Check Problem Completeness ---
def check_problem_completeness(problem_type, session_problem_dict, user_input):
    problem_type_description = problem_type.replace('_', ' ') if problem_type else "unknown optimization"
    prompt = f"""
    You are an expert in {problem_type_description} optimization.

    Current problem dictionary:
    ```python
    {session_problem_dict}
    ```

    User input:
    ```{user_input}
    ```

    Determine if the problem dictionary contains sufficient information to solve the optimization problem.
    If not, list missing fields required for solving.

    Typical requirements for each optimization type:
    - Linear Programming: Objective coefficients (`c_max` or `c_min`), constraints (`A_ub`, `b_ub`, `A_eq`, `b_eq`), bounds, variable names.
    - Combinatorial Optimization (Assignment): Cost matrix (`cost_matrix`), agent names, task names.
    - Network Optimization: Nodes (`nodes`), edges (`edges` with capacities and costs), supply/demand (`supply_demand`).

    Return:
    {{
        "is_complete": True or False,
        "missing_fields": ["field1", "field2", ...]
    }}

    Use Python boolean values `True` and `False` (uppercase).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        count.append(response.usage.total_tokens)
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            completeness_check = safe_eval(match.group(0))
            if not completeness_check.get("is_complete", False):
                missing_fields = completeness_check.get("missing_fields", [])
                error_message = f"Missing information: {', '.join(missing_fields)}"
                return completeness_check, error_message
            return completeness_check, None
        return None, "Failed to parse completeness check response."
    except Exception as e:
        logging.error(f"Error during completeness check: {e}")
        return None, f"Error during completeness check: {str(e)}"

# --- Streamlit App ---
st.set_page_config(page_title="Optimization Solver Chat", layout="wide")
st.title("🔢 Optimization Problem Solver")
st.markdown("Describe your problem in natural language, and let AI extract, solve, and refine it interactively.")

if st.button("Reset Session"):
    st.session_state.history = []
    st.session_state.session_problem = None
    st.session_state.problem_type = ''
    st.session_state.user_input_history = ''
    st.success("Session reset successfully!")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'session_problem' not in st.session_state:
    st.session_state.session_problem = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = ''
if 'user_input_history' not in st.session_state:
    st.session_state.user_input_history = ''

# --- Main Interaction Loop ---
user_input = st.chat_input("Enter a new problem description or a follow-up modification...")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.spinner("Processing..."):
        classification = classify_user_input(user_input, st.session_state.session_problem)
        logging.info(f"Input classified as: {classification}")

        if classification == "new":
            st.session_state.user_input_history = user_input
            opt_types = detect_optimization_type(user_input)
            st.session_state.problem_type = next((key for key, val in opt_types.items() if val), "unknown")

            if not any(opt_types.values()):
                st.session_state.history.append(("assistant", "⚠️ Could not identify optimization type. Please provide more details."))
            else:
                st.session_state.session_problem, error = extract_problem(user_input, st.session_state.problem_type)
                if error or st.session_state.session_problem is None:
                    st.session_state.history.append(("assistant", f"❌ Error extracting problem: {error or 'Invalid problem format.'}"))
                else:
                    completeness_check, error_message = check_problem_completeness(
                        st.session_state.problem_type,
                        st.session_state.session_problem,
                        user_input
                    )
                    if error_message:
                        st.session_state.history.append(("assistant", f"⚠️ {error_message}"))
                        updated_dict = ask_user_to_fill_missing_values(
                            st.session_state.session_problem,
                            st.session_state.problem_type
                        )
                        if updated_dict:
                            st.session_state.session_problem = updated_dict
                    err, opt_val, opt_vars = solve_problem(st.session_state.session_problem, st.session_state.problem_type)
                    if err:
                        st.session_state.history.append(("assistant", f"❌ {err}"))
                    else:
                        formatted = format_solution(opt_val, opt_vars, st.session_state.session_problem, st.session_state.problem_type)
                        human_response = humanize_response(formatted, st.session_state.problem_type.replace('_', ' '))
                        st.session_state.history.append(("assistant", human_response))
                        st.session_state.history.append(("assistant", f"📚 **Detected Optimization Types:** {', '.join([key.replace('_', ' ').title() for key, val in opt_types.items() if val])}"))

        elif classification == "followup" and st.session_state.session_problem:
            modified_problem, error = modify_problem(st.session_state.session_problem, user_input, st.session_state.problem_type)
            if error or modified_problem is None:
                st.session_state.history.append(("assistant", f"❌ Failed to modify problem: {error or 'Invalid modification request.'}"))
            else:
                completeness_check, error_message = check_problem_completeness(
                    st.session_state.session_state.problem_type,
                    modified_problem,
                    user_input
                )
                if error_message:
                    st.session_state.history.append(("assistant", f"⚠️ {error_message}"))
                    updated_problem = ask_user_to_fill_missing_values(modified_problem, st.session_state.problem_type)
                    if updated_problem:
                        modified_problem = updated_problem
                err, opt_val, opt_vars = solve_problem(modified_problem, st.session_state.session_state)
                if err:
                    st.session_state.history.append(("assistant", f"❌ {err}"))
                else:
                    st.session_state.session_problem = modified_problem
                    formatted = format_solution(opt_val, opt_vars, st.session_state.session_problem, st.session_state.session_state)
                    human_response = humanize_response(formatted, st.session_state.problem_type.replace('_', ' '))
                    st.session_state.history.append(("assistant", human_response))

        else:
            st.session_state.history.append(("assistant", "⚠️ Unable to classify input or no existing problem to modify. Please clarify if this is a new problem or a modification."))

# --- Display Chat History ---
for sender, message in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(message)

# --- Display Token Count ---
st.markdown(f"Total tokens used: {sum(count)}")