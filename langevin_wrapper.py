import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt 
from matplotlib import cm
from scipy.signal import find_peaks
from heapq import nlargest
import os 


''' --- 1. Define Bare Molecular Forces & Potentials --- '''

def force_harmonic(xm_values, mol_params):
    wm = mol_params['wm']
    # Broadcasting works if wm is an array of shape (num_mol,) and xm_values is (num_mol,)
    return -xm_values * (wm**2)

def potential_harmonic(xm_values, mol_params):
    wm = mol_params['wm']
    # Reshape frequencies so they broadcast correctly across the time points
    w2 = (wm**2)[:, np.newaxis] if xm_values.ndim > 1 else (wm**2)
    return 0.5 * w2 * (xm_values**2)

def force_morse(xm_values, mol_params):
    D_e = mol_params['D_e']
    a = mol_params['a']
    return -2 * D_e * a * (1 - np.exp(-a * xm_values)) * np.exp(-a * xm_values)

def potential_morse(xm_values, mol_params):
    D_e = mol_params['D_e']
    a = mol_params['a']
    return D_e * (1 - np.exp(-a * xm_values))**2

def force_cubic(xm_values, mol_params):
    """
    Force for a harmonic oscillator with a cubic anharmonicity.
    Force = -w^2 * x - 3 * alpha * x^2
    """
    wm = mol_params['wm']
    alpha = mol_params['alpha']
    
    return -xm_values * (wm**2) - 3 * alpha * (xm_values**2)

def potential_cubic(xm_values, mol_params):
    """
    Potential energy for a harmonic oscillator with a cubic anharmonicity.
    V(x) = 0.5 * w^2 * x^2 + alpha * x^3
    """
    wm = mol_params['wm']
    alpha = mol_params['alpha']
    
    # Reshape frequencies so they broadcast correctly across time points if needed
    w2 = (wm**2)[:, np.newaxis] if xm_values.ndim > 1 else (wm**2)
    
    return 0.5 * w2 * (xm_values**2) + alpha * (xm_values**3)


''' --- 2. Factory Function for Coupled EOMs --- '''

def build_pauli_fierz(molecular_force_func, driven=False, static_global=False, static_local=False):
    """
    Factory function that builds the coupled equations of motion.
    Returns an acceleration function compatible with the Verlet integrator.
    """
    def acceleration(xc, vc, xm_values, vm_values, num_mol, param_cav, mol_params, t):
        wc, E0 = param_cav
        lam = mol_params['lam']
        
        sum_xm = np.sum(xm_values)
        
        # --- Cavity Acceleration ---
        if driven:
            a_xc = E0 * np.cos(wc * t) - (wc**2) * xc + lam * wc * sum_xm
        else:
            a_xc = -(wc**2) * xc + lam * wc * sum_xm
            
        # --- Molecular Acceleration ---
        bare_mol_force = molecular_force_func(xm_values, mol_params)
        
        # Base coupled acceleration (including Dipole Self-Energy)
        a_xm_values = bare_mol_force + lam * xc * wc - (lam**2) * sum_xm
        
        # Apply static fields if requested
        if static_global:
            a_xm_values += E0
        if static_local:
            a_xm_values[0] += E0
            
        return a_xc, a_xm_values
        
    return acceleration


''' --- 3. Velocity Verlet Algorithm --- '''


def velocity_verlet(acceleration, init_cond, time_points, num_mol, param_cav, mol_params):
    n_points = len(time_points)
    dt = time_points[1] - time_points[0]
    
    dt_2 = dt**2
    dt_12 = np.sqrt(dt)
    dt_32 = dt**(3 / 2)
    sqrt3_inv = 1 / (2 * np.sqrt(3))
    
    xc_values = np.zeros(n_points)
    vc_values = np.zeros(n_points)
    xm_values = np.zeros((num_mol, n_points))
    vm_values = np.zeros((num_mol, n_points))
    
    xc_values[0], vc_values[0] = init_cond[:2]
    xm_values[:, 0] = init_cond[2:num_mol+2]
    vm_values[:, 0] = init_cond[num_mol+2:(num_mol+2)+num_mol]
    
    a_xc, a_xm = acceleration(xc_values[0], vc_values[0], xm_values[:, 0], vm_values[:, 0], num_mol, param_cav, mol_params, time_points[0])
    
    for i in range(n_points - 1):
        t = time_points[i]
        
        xi_c, theta_c = np.random.normal(0.0, 1.0, 2)
        xi_m = np.random.normal(0.0, 1.0, num_mol)
        theta_m = np.random.normal(0.0, 1.0, num_mol)
        
        R_c = sigma_c * dt_32 * (0.5 * xi_c + sqrt3_inv * theta_c)
        R_m = sigma_m * dt_32 * (0.5 * xi_m + sqrt3_inv * theta_m)
        
        xc_values[i + 1] = xc_values[i] + vc_values[i] * dt + 0.5 * (a_xc - k * vc_values[i]) * dt_2 + R_c
        xm_values[:, i + 1] = xm_values[:, i] + vm_values[:, i] * dt + 0.5 * (a_xm - lamb * vm_values[:, i]) * dt_2 + R_m
        
        a_xc_new, a_xm_new = acceleration(xc_values[i + 1], vc_values[i], xm_values[:, i + 1], vm_values[:, i], num_mol, param_cav, mol_params, t + dt)
        
        vc_values[i + 1] = (vc_values[i] + 0.5 * (a_xc + a_xc_new) * dt - k * vc_values[i] * dt 
                            + sigma_c * dt_12 * xi_c - k * (0.5 * dt_2 * (a_xc - k * vc_values[i]) + R_c))
        
        vm_values[:, i + 1] = (vm_values[:, i] + 0.5 * (a_xm + a_xm_new) * dt - lamb * vm_values[:, i] * dt 
                               + sigma_m * dt_12 * xi_m - lamb * (0.5 * dt_2 * (a_xm - lamb * vm_values[:, i]) + R_m))
        
        a_xc, a_xm = a_xc_new, a_xm_new
    
    return xc_values, vc_values, xm_values, vm_values



def check_temperature_consistency(vc_values, vm_values, kT, num_mol):
    v2_photon = np.mean(vc_values[-1000:] ** 2)
    v2_molecules = np.zeros(num_mol)

    for j in range(num_mol):
        v2_molecules[j] = np.mean(vm_values[j, -400:] ** 2)
    
    avg_v2_molecules = np.mean(v2_molecules)
    total_avg_v2 = (v2_photon + avg_v2_molecules * num_mol) / (num_mol + 1)
    
    system_consistent = np.isclose(total_avg_v2, kT, rtol=0.8)

    if system_consistent:
        message =("The time-averaged velocities of the photon and molecules are consistent with the initial temperature.")
    else:
        message =("The time-averaged velocities of the photon and molecules are NOT consistent with the initial temperature.")

    return message


''' --- 4. Define Analysis Tool Functions --- '''

def autocorr(values):
    results = np.correlate(values, values, mode='full')
    results = results[results.size // 2:]
    autocorr = results / results[0]
    return autocorr

def crosscorr(values_1, values_2):
    results = np.correlate(values_1, values_2, mode='full')
    results = results[results.size // 2:]
    norm_1 = np.sum(values_1**2)
    norm_2 = np.sum(values_2**2)
    C_12 = results / np.sqrt(norm_1 * norm_2)
    return C_12

def fft_autocorr(autocorr):
    pad_length = len(autocorr) * 9
    padded_autocorr = np.pad(autocorr, (0, pad_length), mode='constant')
    
    fft = np.fft.fft(padded_autocorr)
    fftfreq = 2*np.pi*np.fft.fftfreq(fft.shape[-1], d=t_step)
    
    return fft, fftfreq

def bright_autocorr(xm_values):
    P_t = np.sum(xm_values, axis=0)
    results = np.correlate(P_t, P_t, mode='full')
    results = results[results.size // 2:]
    C_PP = results / results[0]
    return C_PP

def calc_energies(xm_values, vm_values, potential_func, mol_params, mass=1.0):
    """
    Generic energy calculator using the assigned potential energy function.
    """
    kinetic = 0.5 * mass * vm_values**2
    potential = potential_func(xm_values, mol_params)
    return kinetic + potential

def calc_ipr(energies):
    sum_sq = np.sum(energies**2, axis=0)
    sq_sum = (np.sum(energies, axis=0))**2
    ipr_t = np.divide(sum_sq, sq_sum, out=np.zeros_like(sum_sq), where=sq_sum!=0)
    return ipr_t


#-------------------------------------------------------------------------------------#

if __name__ == "__main__":

    aut_to_fs = 0.0241888
    au_to_ev  = 27.2114

    ##########################################
    ''' DEFINE PARAMETERS (Thermalization) '''
    ##########################################

    num_mol = 2
    wc = 0.005512
    wm = 0.005512

    E0 = 0.0001
    param_cav = [wc, E0]
    freqs = np.random.normal(wm, 0.0, num_mol)
    gc = 0.0*wm 
    lam = gc/np.sqrt(num_mol)
    
    # Use dictionary for molecular parameters
    mol_params = {
        'wm': freqs,
        'lam': lam
    }

    k    = 1.0e-5
    lamb = 1.0e-5
    kT   = 1.0*9.44e-4
    beta = 1/kT
    mu   = 1
    sigma_c = np.sqrt(2*kT*k/mu)
    sigma_m = np.sqrt(2*kT*lamb/mu)

    std_x = 1/np.sqrt(beta)
    std_v = 1/np.sqrt(beta * wm**2)*0.05

    init_xm = np.random.normal(0, std_x, num_mol)
    init_vm = np.random.normal(0, std_v, num_mol)

    I = 3*wc/2
    theta = np.random.uniform(0, 2*np.pi,1).item()

    init_xc = np.sqrt(2 * I / (wc)) * np.sin(theta)
    init_vc = np.sqrt(2 * I * wc) * np.cos(theta) * 0.05

    init_cond = [init_xc, init_vc] + init_xm.tolist() + init_vm.tolist()

    t_eq = 350000
    t_step  = 10
    time_points = np.arange(0, t_eq, t_step)
  
    # -------------------------------------------------------------
    # Build standard Hamiltonian for equilibration
    PF_standard = build_pauli_fierz(force_harmonic, driven=False)
    # -------------------------------------------------------------

    for i in range(10):
        xc_values_eq, vc_values_eq, xm_values_eq, vm_values_eq = velocity_verlet(
            PF_standard, init_cond, time_points, num_mol, param_cav, mol_params
        )
        message = check_temperature_consistency(vc_values_eq, vm_values_eq, kT, num_mol)

        if message == "The time-averaged velocities of the photon and molecules are consistent with the initial temperature.":
            print(message)
            break
    else:
        print('\n EQUIBILBRATION FAILED! \n modify system parameters, script terminated')
            
    
    ''' Take initial condition from equilibration run and propagate '''

    init_xm = xm_values_eq[:,-1]
    init_vm = vm_values_eq[:,-1]
    init_xc = xc_values_eq[-1]
    init_vc = vc_values_eq[-1]

    init_cond = [init_xc, init_vc] + init_xm.tolist() + init_vm.tolist()

    # Redefine parameters
    param_cav = [wc, 0.0] # Setting E0 to 0.0 if you aren't driving the cavity
    gc = 0.08 * wc
    lam_prop = gc / np.sqrt(num_mol) 
    
    # Explicitly redefine the dictionary for absolute clarity
    mol_params = {
        'wm': freqs,
        'lam': lam_prop
    }

    t_final = 1000000
    time_points = np.arange(0, t_final, t_step)

    # -------------------------------------------------------------
    # Build the specific Hamiltonian for propagation (Local Static Field)
    PF_static_local = build_pauli_fierz(force_harmonic, static_global=False)
    # -------------------------------------------------------------

    xc_values, vc_values, xm_values, vm_values = velocity_verlet(
        PF_static_local, init_cond, time_points, num_mol, param_cav, mol_params
    )

    ''' Calculate things '''
    
    C_xcxc = autocorr(xm_values[0])
    C_vcvc = autocorr(vm_values[0])
    C_xx_bright = bright_autocorr(xm_values)
    C_vv_bright = bright_autocorr(vm_values)

    fft_xcxc, fftfreq_xcxc = fft_autocorr(C_xcxc)
    fft_vcvc, fftfreq_vcvc = fft_autocorr(C_vcvc)
    fft_xx_bright, fftfreq_xx_bright = fft_autocorr(C_xx_bright)
    fft_vv_bright, fftfreq_vv_bright = fft_autocorr(C_vv_bright)
    
    # Calculate energies using the corresponding potential function
    energies = calc_energies(xm_values, vm_values, potential_harmonic, mol_params)
    ipr = calc_ipr(energies)

    av_pos = np.mean(xm_values)
    print('Average position of the molecules:', av_pos)


    ''' PLOT '''

    fig, axs = plt.subplots(2, 2, figsize=(8,4), constrained_layout=True, sharex='col')

    axis_thickness = 1.5
    label_fontsize = 12

    for ax in axs.flat:
        ax.tick_params(width=axis_thickness, labelsize=label_fontsize)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(1,-1))
        for spine in ax.spines.values():
            spine.set_linewidth(axis_thickness)

    ax00 = axs[0,0]
    ax00.plot(time_points*aut_to_fs, C_xcxc, label='C_xcxc(t)', color='darkorange')
    ax00.legend(loc="upper right")
    ax00.set_xlabel('Time (fs)')
    ax00.set_ylabel('Photon pos. autocorr.')

    ax01 = axs[0,1]
    ax01.plot(time_points*aut_to_fs, C_vcvc, label='C_vcvc(t)', color='darkorange')
    ax01.legend(loc="upper right")
    ax01.set_xlabel('Time (fs)')
    ax01.set_ylabel('Photon vel. autocorr.')

    ax10 = axs[1,0]
    ax10.plot(time_points*aut_to_fs, C_xx_bright, label='C_xx_bright(t)', color='tab:blue')
    ax10.legend(loc="upper right")
    ax10.set_xlabel('Time (fs)')
    ax10.set_ylabel('Bright pos. autocorr.')

    ax11 = axs[1,1]
    ax11.plot(time_points*aut_to_fs, C_vv_bright, label='C_vv_bright(t)', color='tab:blue')
    ax11.legend(loc="upper right")
    ax11.set_xlabel('Time (fs)')
    ax11.set_ylabel('Bright vel. autocorr.')

    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(8,4), constrained_layout=True, sharex='col')

    for ax in axs.flat:
        ax.set_xlim((wc-1.0*gc)*au_to_ev , (wc+1.0*gc)*au_to_ev)
        ax.set_ylim(0.0,1.1)
        ax.tick_params(width=axis_thickness, labelsize=label_fontsize)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(1,-1))
        for spine in ax.spines.values():
            spine.set_linewidth(axis_thickness)

    ax00 = axs[0,0]
    ax00.plot(fftfreq_xcxc*au_to_ev, np.abs(fft_xcxc.real)/np.max(np.abs(fft_xcxc.real)), label='FT(C_xcxc(t))', color='darkorange')
    ax00.legend(loc="upper right")
    ax00.set_xlabel('Frequency (eV)')
    ax00.set_ylabel('Photon pos. FT')

    ax01 = axs[0,1]
    ax01.plot(fftfreq_vcvc*au_to_ev, np.abs(fft_vcvc.real)/np.max(np.abs(fft_vcvc.real)), label='FT(C_vcvc(t))', color='darkorange')
    ax01.legend(loc="upper right")
    ax01.set_xlabel('Frequency (eV)')
    ax01.set_ylabel('Photon vel. FT')

    ax10 = axs[1,0]
    ax10.plot(fftfreq_xx_bright*au_to_ev, np.abs(fft_xx_bright.real)/np.max(np.abs(fft_xx_bright.real)), label='FT(C_xx_bright(t))', color='tab:blue')
    ax10.legend(loc="upper right")
    ax10.set_xlabel('Frequency (eV)')
    ax10.set_ylabel('Bright pos. FT')

    ax11 = axs[1,1]
    ax11.plot(fftfreq_vv_bright*au_to_ev, np.abs(fft_vv_bright.real)/np.max(np.abs(fft_vv_bright.real)), label='FT(C_vv_bright(t))', color='tab:blue')
    ax11.legend(loc="upper right")
    ax11.set_xlabel('Frequency (eV)')
    ax11.set_ylabel('Bright vel. FT')

    plt.show()

    plt.figure(figsize=[12,6])
    plt.title('Coupled Generalized Langevin')
    plt.plot(time_points[-500:]*aut_to_fs, xc_values[-500:], label='xc(t)', color='black')
    plt.plot(time_points[-500:]*aut_to_fs, xm_values[i,-500:], label=f'xm0(t)', color='tab:red')
    for i in range(1, num_mol):
        plt.plot(time_points[-500:]*aut_to_fs, xm_values[i,-500:], label=f'xm{i+1}(t)', color='tab:cyan', alpha=0.4)
    plt.xlabel('Time (fs)')
    plt.ylabel('Values')
    plt.show()

    plt.figure(figsize=[12,6])
    plt.title('IPR')
    plt.plot(time_points*aut_to_fs, ipr, label='ipr', color='tab:blue')
    plt.xlabel('Time (fs)')
    plt.ylabel('Values')
    plt.show()

    # Save initial positions to a text file
    with open("initial_positions.txt", "w") as f_pos:
        f_pos.write(f"{init_xc}\n")
        f_pos.write(' '.join(map(str, init_xm)))

    # Save initial velocities to a text file
    with open("initial_velocities.txt", "w") as f_vel:
        f_vel.write(f"{init_vc}\n")
        f_vel.write(' '.join(map(str, init_vm)))

    E_ampl = 0
    dir_path = f"{gc}_{E_ampl}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    thermalization_path = os.path.join(dir_path, 'thermalization.txt')
    with open(thermalization_path, 'w') as readme_file:
        readme_file.write(message)