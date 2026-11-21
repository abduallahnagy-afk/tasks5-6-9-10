# ---------------- 1. STATES ----------------
WIDTH, HEIGHT = 3, 3
WALL = (1, 1)
STATES = [(x, y) for x in range(WIDTH) for y in range(HEIGHT) if (x, y) != WALL]

# ---------------- 2. ACTIONS ----------------
ACTIONS = {"Up": (0, -1), "Down": (0, 1), "Left": (-1, 0), "Right": (1, 0)}

# ---------------- 3. REWARD FUNCTION ----------------
START, GOAL, PIT = (0, 0), (2, 2), (2, 1)
TERMINALS = {GOAL: 1.0, PIT: -1.0}
STEP_REWARD = -0.04                         # small cost per move, so the robot hurries
GAMMA = 0.9                                 # discount: future rewards count a bit less


def reward(state):
    return TERMINALS.get(state, STEP_REWARD)


# ---------------- 4. TRANSITION MODEL ----------------
# Movement is uncertain: 80% the robot goes where it wants,
# 10% it slips to the left side, 10% it slips to the right side.
SIDEWAYS = {"Up": ["Left", "Right"], "Down": ["Left", "Right"],
            "Left": ["Up", "Down"], "Right": ["Up", "Down"]}


def move(state, action):
    """Where the robot lands after one move. Hitting a wall or edge = stay in place."""
    dx, dy = ACTIONS[action]
    new_state = (state[0] + dx, state[1] + dy)
    inside_grid = 0 <= new_state[0] < WIDTH and 0 <= new_state[1] < HEIGHT
    if new_state == WALL or not inside_grid:
        return state
    return new_state


def transitions(state, action):
    """List of (probability, next state) for taking an action in a state."""
    side1, side2 = SIDEWAYS[action]
    return [(0.8, move(state, action)),
            (0.1, move(state, side1)),
            (0.1, move(state, side2))]


# ---------------- 5. SOLVING WITH VALUE ITERATION ----------------
def q_value(state, action, V):
    """Bellman equation: reward now + discounted expected value of where we land."""
    expected_future = sum(prob * V[next_s] for prob, next_s in transitions(state, action))
    return reward(state) + GAMMA * expected_future


def value_iteration(theta=0.000001):
    V = {s: 0.0 for s in STATES}          # start with every value = 0
    history = []                           # (iteration, value of start, biggest change)
    iteration = 0
    while True:
        iteration += 1
        new_V = {}
        for s in STATES:
            if s in TERMINALS:
                new_V[s] = reward(s)       # game ends here, value is just the reward
            else:
                new_V[s] = max(q_value(s, a, V) for a in ACTIONS)   # best action wins
        biggest_change = max(abs(new_V[s] - V[s]) for s in STATES)
        V = new_V
        history.append((iteration, V[START], biggest_change))

        if biggest_change < theta:         # values stopped changing -> solved
            return V, history


def best_policy(V):
    """For every non-terminal state, pick the action with the highest Q-value."""
    return {s: max(ACTIONS, key=lambda a: q_value(s, a, V))
            for s in STATES if s not in TERMINALS}


def follow_policy(policy):
    """The path the robot takes from START if every move goes as intended."""
    path, s = [START], START
    while s not in TERMINALS and len(path) < 20:
        s = move(s, policy[s])
        path.append(s)
    return path


# ---------------- 6. PRINTING THE RESULTS ----------------
CELL = 9                                    # width of one grid cell
ARROWS = {"Up": "^", "Down": "v", "Left": "<", "Right": ">"}


def title(text):
    print("\n" + "=" * 52)
    print(" " + text)
    print("=" * 52)


def draw_grid(cell_text):
    """Draw the 3x3 grid with borders. cell_text(state) gives the text for each cell."""
    line = "      +" + ("-" * CELL + "+") * WIDTH
    print("      " + "".join(f"x={x}".center(CELL + 1) for x in range(WIDTH)))
    print(line)
    for y in range(HEIGHT):                    # y=0 is the top row
        cells = [cell_text((x, y)).center(CELL) for x in range(WIDTH)]
        print(f" y={y}  |" + "|".join(cells) + "|")
        print(line)


def draw_table(headers, rows):
    """Draw a simple table with borders."""
    widths = [max(len(str(v)) for v in col) + 2 for col in zip(headers, *rows)]
    line = "  +" + "+".join("-" * w for w in widths) + "+"
    show = lambda vals: "  |" + "|".join(str(v).center(w) for v, w in zip(vals, widths)) + "|"
    print(line)
    print(show(headers))
    print(line)
    for r in rows:
        print(show(r))
    print(line)


def special(s):
    """Label for the special cells, or None for normal cells."""
    if s == WALL:
        return "WALL"
    if s == GOAL:
        return "GOAL"
    if s == PIT:
        return "PIT"
    return None
# ---------------- 7. RUN THE PROGRAM ----------------

print("\nThe grid world:\n")
draw_grid(lambda s: special(s) or ("START" if s == START else ""))
print("  START = where the robot begins   GOAL = +1 (ends)")
print("  PIT   = -1 (ends)                WALL = robot cannot enter")

print("\nMDP definition:\n")
draw_table(["Part", "Value"], [
    ["States", f"{len(STATES)} cells (all except the wall)"],
    ["Actions", "Up, Down, Left, Right"],
    ["Transitions", "80% intended, 10% slip each side"],
    ["Rewards", f"GOAL +1, PIT -1, each step {STEP_REWARD}"],
    ["Discount (gamma)", f"{GAMMA}"],
])

title("STEP 1 : VALUE ITERATION")
print("Repeat the Bellman update until the values stop changing.\n")
V, history = value_iteration()
shown = [h for h in history if h[0] in (1, 2, 3, 5, 10, 15)] + [history[-1]]
draw_table(["Iteration", "Value of START", "Biggest change"],
           [[i, f"{v:+.4f}", f"{c:.6f}"] for i, v, c in shown])
print(f"\n  -> Converged after {len(history)} iterations (change < 0.000001)")

title("STEP 2 : VALUE OF EACH CELL")
print("How good it is to be in each cell (higher = better).\n")
draw_grid(lambda s: "WALL" if s == WALL else f"{V[s]:+.3f}")

policy = best_policy(V)
title("STEP 3 : OPTIMAL POLICY")
print("The best move in each cell.\n")
draw_grid(lambda s: special(s) or ARROWS[policy[s]])
print("  ^ = Up    v = Down    < = Left    > = Right")

path = follow_policy(policy)
title("STEP 4 : ROUTE FROM START TO GOAL")
print("Numbers show the order of the moves (if no slips happen).\n")


def route_cell(s):
    if s == START:
        return "0=START"
    if s == GOAL:
        return f"{path.index(s)}=GOAL"
    if s in path:
        return str(path.index(s))          # step number on the route
    return special(s) or "."               # "." = not on the route

draw_grid(route_cell)
print("  Route: " + " -> ".join(str(s) for s in path))
print(f"  Moves needed: {len(path) - 1}")

print(f"\nNote: at (2,0) the best move is '{policy[(2, 0)]}' ({ARROWS[policy[(2, 0)]]}), not Down.")
print("      Going Down next to the PIT is too risky, so the robot takes the safe way round.")