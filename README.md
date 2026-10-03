import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# 1. System Parameters
REGIMES = {
    'periodic': 6.5,
    'single_scroll': 9.0,
    'double_scroll': 10.5
}
BETA = 14.286
M0 = -1/7
M1 = 2/7
DT = 0.01  # Time step size for your LLM sequence generation

# Piecewise-linear function for Chua's Diode
def chua_diode(x):
    return M1 * x + 0.5 * (M0 - M1) * (np.abs(x + 1) - np.abs(x - 1))

# 3D Vector field equations
def chua_system(t, state, alpha):
    x, y, z = state
    dxdt = alpha * (y - x - chua_diode(x))
    dydt = x - y + z
    dzdt = -BETA * y
    return [dxdt, dydt, dzdt]

# 2. Benettin's Two-Trajectory LLE Estimation Method
def compute_lle_and_trajectory(alpha, t_max=400, dt=0.01, d0=1e-8):
    # Set specialized initial positions to bypass initial transients
    init_state = [0.1, 0.1, 0.0] if alpha < 10 else [0.1, 0.1, 0.1]

    # Pre-integrate for 100 time units to shed transient behavior
    burn_in = solve_ivp(chua_system, (0, 100), init_state, args=(alpha,), rtol=1e-9, atol=1e-9)
    state1 = burn_in.y[:, -1]

    # Establish a tracking shadow trajectory displaced by distance d0
    state2 = state1 + np.array([d0, 0.0, 0.0])

    n_steps = int(t_max / dt)
    lyap_sum = 0
    curr_t = 0.0

    # Save arrays for the baseline trajectory
    trajectory_points = []

    for step in range(n_steps):
        # Record exactly 100 steps after burn-in to form your prompt context
        if step < 100:
            trajectory_points.append(state1.copy())

        # Step both vectors forward via identical integration steps
        sol1 = solve_ivp(chua_system, (curr_t, curr_t + dt), state1, args=(alpha,), rtol=1e-8, atol=1e-8)
        sol2 = solve_ivp(chua_system, (curr_t, curr_t + dt), state2, args=(alpha,), rtol=1e-8, atol=1e-8)

        state1 = sol1.y[:, -1]
        state2 = sol2.y[:, -1]

        # Determine the Euclidean distance after divergence
        d1 = np.linalg.norm(state1 - state2)

        # Accumulate local exponential growth rate
        if d1 > 0:
            lyap_sum += np.log(d1 / d0)

        # Renormalize the shadow vector back along the exact axis of expansion
        state2 = state1 + (d0 / d1) * (state2 - state1)
        curr_t += dt

    # LLE is the average exponential growth over the time horizon
    lle = lyap_sum / t_max
    return lle, np.array(trajectory_points)

# 3. Process Regimes and Export Metrics Independently
for name, alpha in REGIMES.items():
    print(f"Processing and exporting files for {name} regime...")
    lle, prompt_data = compute_lle_and_trajectory(alpha)

    # Handle mathematical boundaries for stable periodic trajectories (LLE <= 0)
    if lle <= 0:
        lle = 0.0
        lyap_time = float('inf')
        steps_limit = float('inf')
    else:
        lyap_time = 1.0 / lle
        steps_limit = lyap_time / DT

    # File A: Save isolated Lyapunov metrics for just this specific regime
    metrics_data = [{
        'Regime': name,
        'Alpha': alpha,
        'LLE': round(lle, 4),
        'Lyapunov_Time_SystemUnits': round(lyap_time, 2) if lyap_time != float('inf') else 'Infinite',
        'Max_Predictable_Steps_Horizon': round(steps_limit, 1) if steps_limit != float('inf') else 'Infinite'
    }]
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(f'chua_lyapunov_metrics_{name}.csv', index=False)

    # Convert trajectory to a clean DataFrame
    df_prompt = pd.DataFrame(prompt_data, columns=['X', 'Y', 'Z'])

    # File B: Export Raw Decimal Format (Rounded cleanly to 4 decimal places)
    df_prompt_decimal = df_prompt.round(4)
    df_prompt_decimal.to_csv(f'chua_prompt_snapshot_decimal_{name}.csv', index=False)

    # File C: Export Scaled Integer Format (Multiplied by 1000 to preserve precision as clean individual text tokens)
    df_prompt_integer = np.round(df_prompt * 1000).astype(int)
    df_prompt_integer.to_csv(f'chua_prompt_snapshot_integer_{name}.csv', index=False)

print("\nExecution complete. All isolated data sheets and dual prompt files written successfully.")

