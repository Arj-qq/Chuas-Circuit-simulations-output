import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# 1. System Parameters
REGIMES = {
    'periodic': 7.5,
    'single_scroll': 8.5,
    'double_scroll': 10.5
}
BETA = 100/7
M0 = -8/7
M1 = -5/7
DT = 0.01  # Time step size for your LLM sequence generation and for bennettin renormalization
N_CONTEXT = 150      # Number of recorded trajectory points 100 * DT = 1 time unit of context, may change less or more based of more readings i do not sure, 100 seems safe.
T_BURN = 100.0       # warm up period get it runnin
T_MAX = 400.0        # LLE averaging horizon
D0 = 1e-6            # Benettin separation (well above integrator noise; hopefully)
RTOL, ATOL = 1e-10, 1e-12
INIT_STATE = [0.1, 0.1, 0.1]
LLE_CHAOS_THRESHOLD = 0.01  # LLE below this is treated as non-chaotic 

# function for Chua's Diode
def chua_diode(x):
    return M1 * x + 0.5 * (M0 - M1) * (np.abs(x + 1) - np.abs(x - 1))

def chua_system(t, state, alpha):
    x, y, z = state
    dxdt = alpha * (y - x - chua_diode(x))
    dydt = x - y + z
    dzdt = -BETA * y
    return [dxdt, dydt, dzdt]
 
 
def integrate(state, t0, t1, alpha):
    sol = solve_ivp(chua_system, (t0, t1), state, args=(alpha,),
                    method='DOP853', rtol=RTOL, atol=ATOL)
    if sol.status != 0:
        raise RuntimeError(f"Integration failed at t={t0} (alpha={alpha}): {sol.message}")
    return sol.y[:, -1]
 
 
# 2.  Two-Trajectory LLE Estimation Method using benettin method
def compute_lle_and_trajectory(alpha, t_max=T_MAX, dt=DT, d0=D0):
    # Pre-integrate to shed transient behavior
    state1 = integrate(INIT_STATE, 0.0, T_BURN, alpha)
 
    # Shadow trajectory displaced by d0
    state2 = state1 + np.array([d0, 0.0, 0.0])
 
    n_steps = int(round(t_max / dt))
    lyap_sum = 0.0
    curr_t = 0.0
    trajectory_points = []
 
    for step in range(n_steps):
        # Record the first N_CONTEXT points after burn-in as the prompt context
        if step < N_CONTEXT:
            trajectory_points.append(state1.copy())
 
        state1 = integrate(state1, curr_t, curr_t + dt, alpha)
        state2 = integrate(state2, curr_t, curr_t + dt, alpha)
 
        d1 = np.linalg.norm(state1 - state2)
        if d1 == 0:
            raise RuntimeError("Trajectories coincided exactly; increase d0.")
 
        lyap_sum += np.log(d1 / d0)
 
        # Renormalize the shadow trajectory back to distance d0 along the separation direction
        state2 = state1 + (d0 / d1) * (state2 - state1)
        curr_t += dt
 
    # LLE = total log-growth / total elapsed time (n_steps * dt == t_max)
    lle = lyap_sum / (n_steps * dt)
    return lle, np.array(trajectory_points)
 
 
# 3. Process Regimes and import the data.
for name, alpha in REGIMES.items():
    print(f"Processing and exporting files for {name} regime (alpha={alpha})...")
    lle, prompt_data = compute_lle_and_trajectory(alpha)
 
    # Keep the signed LLE. Only the Lyapunov time is undefined for non-chaotic regimes.
    if lle < LLE_CHAOS_THRESHOLD:
        lyap_time = float('inf')
        steps_limit = float('inf')
    else:
        lyap_time = 1.0 / lle
        steps_limit = lyap_time / DT
 
    print(f"  LLE = {lle:+.4f}  |  Lyapunov time = {lyap_time:.2f}  |  horizon steps = {steps_limit:.1f}")
 
    # File A: Save isolated Lyapunov metrics for just this specific regime
    DECIMAL_PLACES = 4   # <<< NEW: named constants so the metrics file can record them
    INT_SCALE = 1000.     # <<< NEW
 
    metrics_data = [{
        'Regime': name,
        'Alpha': alpha,
        # <<< CHANGED: renamed from 'LLE' so the column says what it is and its units
        'Largest_Lyapunov_Exponent_per_time_unit': round(lle, 4),
        'Lyapunov_Time_SystemUnits': round(lyap_time, 2) if lyap_time != float('inf') else 'Infinite',
        'Max_Predictable_Steps_Horizon': round(steps_limit, 1) if steps_limit != float('inf') else 'Infinite',
        # <<< NEW: everything below documents how the number was produced and i added it for more clarifications.
        'Chaotic_Threshold_LLE': LLE_CHAOS_THRESHOLD,  
        'Beta': round(BETA, 6),
        'm0_inner_slope': round(M0, 6),
        'm1_outer_slope': round(M1, 6),
        'Averaging_Horizon_T_max': T_MAX,
        'Burn_In_Time': T_BURN,
        'Renorm_Step_DT': DT,
        'Initial_Separation_d0': D0,
        'Integrator_rtol': RTOL,
        'Integrator_atol': ATOL,
        'Method': 'Benettin two-trajectory (largest exponent only)',
        'Snapshot_Points': N_CONTEXT,
        'Snapshot_Spacing_DT': DT,
        'Decimal_Places': DECIMAL_PLACES,
        'Integer_Scale_Factor': INT_SCALE,
    }]
    pd.DataFrame(metrics_data).to_csv(f'chua_lyapunov_metrics_{name}.csv', index=False)
 
    df_prompt = pd.DataFrame(prompt_data, columns=['X', 'Y', 'Z'])
 
    # File B: Raw decimal format (DECIMAL_PLACES decimals). Kept as bare X,Y,Z so it is prompt-ready n easy for me.
    df_prompt.round(DECIMAL_PLACES).to_csv(f'chua_prompt_snapshot_decimal_{name}.csv', index=False)
 
    # File C: Scaled integer format (value * INT_SCALE, rounded). Kept as bare X,Y,Z.
    np.round(df_prompt * INT_SCALE).astype(int).to_csv(f'chua_prompt_snapshot_integer_{name}.csv', index=False)
 
print("\nExecution complete. All isolated data sheets and dual prompt files written successfully.")
