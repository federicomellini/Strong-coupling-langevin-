import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
from heapq import nlargest
import os 





'''Define accelerations based on different hamiltonians'''

def Pauli_Fierz(xc, vc, xm_values, vm_values, num_mol, param_cav, param_mol, t):
    wc, E = param_cav
    wm = param_mol[0:num_mol]
    gamma = param_mol[-1]
    
    a_xc = - xc * wc**2  - gamma *(np.sum(xm_values)) #- k * vc
    
    a_xm_values = np.zeros(num_mol)
    for i in range(num_mol):
        a_xm_values[i] =  -xm_values[i] * wm[i]** 2  - gamma * xc - gamma**2/(wc**2) *(np.sum(xm_values))
    
    return a_xc, a_xm_values

def Pauli_Fierz_driven(xc, vc, xm_values, vm_values, num_mol, param_cav, param_mol, t):
    wc, E = param_cav
    wm = param_mol[0:num_mol]
    gamma = param_mol[-1]
    
    a_xc = E * np.cos(wc * t) - xc * wc**2  - gamma *(np.sum(xm_values)) #- k * vc
    
    a_xm_values = np.zeros(num_mol)
    for i in range(num_mol):
        a_xm_values[i] =  -xm_values[i] * wm[i]** 2  - gamma * xc - gamma**2/(wc**2) *(np.sum(xm_values))
    
    return a_xc, a_xm_values


#------------------------------------------------------------------------------------------#

''' Define Velocity Verlet algorithm'''
def velocity_verlet(acceleration, init_cond, time_points, num_mol, param_cav, param_mol):
    n_points = len(time_points)
    dt = time_points[1] - time_points[0]
    
    # Initialize arrays to store positions and velocities
    xc_values = np.zeros(n_points)
    vc_values = np.zeros(n_points)
    xm_values = np.zeros((num_mol, n_points))
    vm_values = np.zeros((num_mol, n_points))
    
    # Set initial conditions
    xc_values[0], vc_values[0] = init_cond[:2]
    xm_values[:, 0], vm_values[:, 0] = init_cond[2:num_mol+2], init_cond[num_mol+2:(num_mol+2)+num_mol]
    
    for i in range(n_points - 1):
        t = time_points[i]
        
        # Calculate current accelerations
        a_xc, a_xm_values = acceleration(xc_values[i], vc_values[i], xm_values[:, i], vm_values[:, i], num_mol, param_cav, param_mol, t)
        
        # Update positions
        R_c = sigma_c * dt**(3 / 2) * (0.5*np.random.normal(0.0, 1.0, 1) + 1/(2*np.sqrt(3))*np.random.normal(0.0, 1.0, 1)) # xi and theta
        #print('current xm_values[:,i]:', xm_values[:,i])
        xc_values[i + 1] = xc_values[i] + vc_values[i] * dt + 0.5 * (a_xc - k * vc_values[i]) * dt**2  + R_c.item()

        for j in range(num_mol):
            R_m = sigma_m * dt ** (3 / 2) * (0.5*np.random.normal(0.0, 1.0, 1) + 1/(2*np.sqrt(3))*np.random.normal(0.0, 1.0, 1))  # xi and theta
            #print('current R_m:', R_m)
            xm_values[j, i + 1] = xm_values[j, i] + vm_values[j, i] * dt + 0.5 * (a_xm_values[j] - lamb * vm_values[j, i]) * dt ** 2 + R_m.item()
        
        # Calculate new accelerations at the new positions
        a_xc_new, a_xm_values_new = acceleration(xc_values[i + 1], vc_values[i], xm_values[:, i + 1], vm_values[:, i], num_mol, param_cav, param_mol, t + dt)
        
        # Update velocities
        vc_values[i + 1] = vc_values[i] + 0.5 * (a_xc + a_xc_new) * dt - k * vc_values[i]*dt + sigma_c * np.sqrt(dt) * np.random.normal(0.0, 1.0, 1).item() - k *(0.5 * dt ** 2 * (a_xc - k * vc_values[i]) + R_c.item())
        for j in range(num_mol):
            vm_values[j, i + 1] = vm_values[j, i] + 0.5 * (a_xm_values[j] + a_xm_values_new[j]) * dt - lamb * vm_values[j, i]*dt + sigma_m * np.sqrt(dt) * np.random.normal(0.0, 1.0, 1).item() - lamb * (0.5 * dt ** 2 * (a_xm_values[j] - lamb * vm_values[j, i]) + R_m.item())
    
    return xc_values, vc_values, xm_values, vm_values


def check_temperature_consistency(vc_values, vm_values, kT, num_mol):
    """
    This function checks if the time-averaged velocities (photon + molecules) are consistent with the initial temperature.
    
    Parameters:
    - vc_values: Array of photon velocities (1D array).
    - vm_values: Array of molecular velocities (2D array, each row corresponds to a molecule).
    - kT: The value of k_B * T (in atomic units).
    - num_mol: Number of molecules.
    
    Returns:
    - Consistency check for the system (photon + molecules).
    """
    
    # Calculate the time-averaged <v^2> for the photon
    v2_photon = np.mean(vc_values[-40000:] ** 2)

    # Calculate the time-averaged <v^2> for each molecule
    v2_molecules = np.zeros(num_mol)

    for j in range(num_mol):
        v2_molecules[j] = np.mean(vm_values[j, -40000:] ** 2)  # Average over time for each molecule
    
    # Average <v^2> across all molecules
    avg_v2_molecules = np.mean(v2_molecules)



    # Combine the photon and molecule contributions
    total_avg_v2 = (v2_photon + avg_v2_molecules * num_mol) / (num_mol + 1)
    
    print("Time-averaged <v^2> for photon:", v2_photon)
    print("Average <v^2> for molecules:", avg_v2_molecules)
    print("Combined average <v^2> for system (photon + molecules):", total_avg_v2)
    print("Expected <v^2> from temperature (kT):", kT)
    
    # Compare with the expected value from temperature (kT)
    system_consistent = np.isclose(total_avg_v2, kT, rtol=0.2)

    if system_consistent:
        message = print("The time-averaged velocities of the photon and molecules are consistent with the initial temperature.")
    else:
        message =("The time-averaged velocities of the photon and molecules are NOT consistent with the initial temperature.")
    
    return message




if __name__ == "__main__":


    ##########################################
    ''' DEFINE PARAMETERS (Thermalization) '''
    ##########################################


    # Set up number of molecules:
    num_mol = 10

    # Set up parameters
    wc = 0.0161 
    wm = 0.0161 # freqs au of the CH sym stretch in CHCl3
    E0 = 0.0 #0.00009 #0.0005 # Amplitude of driving laser
    param_cav = [wc, E0]  # wc, E
    #freqs = np.random.normal(wm, 0.0002, num_mol)  # wm
    freqs = [0.0158667, 0.01628286, 0.01618053, 0.0160544,  0.01621614, 0.01600337, 0.01590205, 0.01618737, 0.01633669, 0.01622438]
    gc =0.0
    gamma = gc/np.sqrt(num_mol) # light-matter coupling
    param_mol = freqs + [gamma] # wm, gamma

    # Set up friction coefficients k, lamb and random kick sigma: 
    k    = 5.46e-6
    lamb = 5.46e-6
    kT   = 0.000944*0.01 # 9.44x10⁻4 au is value of room temperature energy 25,7 meV
    beta = 1/kT
    mu   = 1
    sigma_c = np.sqrt(2*kT*k/mu)
    sigma_m = np.sqrt(2*kT*lamb/mu)

    # Not really that important how the initial conditions are chosen, because of Markovianity
    # Molecules
    std_x = 1/np.sqrt(beta)
    std_v = 1/np.sqrt(beta * wm**2)*0.1

    init_xm = np.random.normal(0, std_x, num_mol)
    init_vm = np.random.normal(0, std_v, num_mol)

    # Photon
    I = 3*wc/2 # energy when cavity excited with one photon
    theta = np.random.uniform(0, 2*np.pi,1).item()

    init_xc = np.sqrt(2 * I / (wc)) * np.sin(theta)
    init_vc = np.sqrt(2 * I * wc) * np.cos(theta) * 0.1

    # Combine initial conditions into a single list
    init_cond = [init_xc, init_vc] + init_xm.tolist() + init_vm.tolist()


    # Define time pointsp
    time_points = np.arange(0, 2000000, 100)  # Time points from 0 to 10

    # Solve the system using Velocity-Verlet algorithm
    xc_values_eq, vc_values_eq, xm_values_eq, vm_values_eq = velocity_verlet(Pauli_Fierz_driven, init_cond, time_points, num_mol, param_cav, param_mol)

    # Check temperature consistency with equipartition theorem
    message = check_temperature_consistency(vc_values_eq, vm_values_eq, kT, num_mol)
    print(message)



    # Molecules
    init_xm = xm_values_eq[:,-1]
    init_vm = vm_values_eq[:,-1]

    # Photon 
    init_xc = xc_values_eq[-1]
    init_vc = vc_values_eq[-1]


    # Save initial positions to a text file
    with open("initial_positions.txt", "w") as f_pos:
        # Save init_xc and init_xm in separate lines
        f_pos.write(f"{init_xc}\n")
        f_pos.write(' '.join(map(str, init_xm)))

    # Save initial velocities to a text file
    with open("initial_velocities.txt", "w") as f_vel:
        # Save init_vc and init_vm in separate lines
        f_vel.write(f"{init_vc}\n")
        f_vel.write(' '.join(map(str, init_vm)))


    ''' Define path of directory for saving stuff'''
    E_ampl =0
    dir = f"{gc}_{E_ampl}"


    # Create the directory if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    thermalization_path = os.path.join(dir, 'thermalization.txt')

    with open(thermalization_path, 'w') as readme_file:
        readme_file.write(message)





    
   
