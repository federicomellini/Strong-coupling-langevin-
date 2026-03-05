import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
from heapq import nlargest
import os 


''' Define Velocity Verlet algorithm'''


# Define the acceleration function for the system
def acceleration(xc, vc, xm_values, vm_values, num_mol, param_cav, param_mol, t):
    wc, E = param_cav
    wm = param_mol[0:num_mol]
    gamma = param_mol[-1]
    
    a_xc = E * np.cos(wc * t) - xc * wc**2  - gamma *(np.sum(xm_values)) #- k * vc
    
    a_xm_values = np.zeros(num_mol)
    for i in range(num_mol):
        a_xm_values[i] =  -xm_values[i] * wm[i-1]** 2  - gamma * xc - gamma**2/(2*wc**2) *(np.sum(xm_values))
    
    return a_xc, a_xm_values

# Define the Velocity-Verlet algorithm
def velocity_verlet(init_cond, time_points, num_mol, param_cav, param_mol):
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
        
        # Calculate gaussian random variables for the photon
        R_c = sigma_c * dt**(3 / 2) * (0.5*np.random.normal(0.0, 1.0, 1) + 1/(2*np.sqrt(3))*np.random.normal(0.0, 1.0, 1)) # xi and theta
        
        # Update positions
        xc_values[i + 1] = xc_values[i] + vc_values[i] * dt + 0.5 * (a_xc - k * vc_values[i]) * dt**2  + R_c.item()

        for j in range(num_mol):
            # Calculate gaussian random variables for each molecule
            R_m = sigma_m * dt ** (3 / 2) * (0.5*np.random.normal(0.0, 1.0, 1) + 1/(2*np.sqrt(3))*np.random.normal(0.0, 1.0, 1))  # xi and theta
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
    system_consistent = np.isclose(total_avg_v2, kT, rtol=0.1)

    if system_consistent:
        print("The time-averaged velocities of the photon and molecules are consistent with the initial temperature.")
    else:
        print("The time-averaged velocities of the photon and molecules are NOT consistent with the initial temperature.")
    
    return system_consistent




if __name__ == "__main__":

    

    #######################################################
    ''' REDEFINE INITIAL CONDITIONS AFTER THERMALIZATION'''
    #######################################################

    ''' PARAMETERS '''
    # Set up number of molecules:
    num_mol = 10

    # Set up parameters
    wc = 0.0161#0.01376 
    wm = 0.0161 # freqs au of the CH sym stretch in CHCl3
    # Set up friction coefficients k, lamb and random kick sigma: 
    k    = 5.46e-6
    lamb = 5.46e-6
    E_ampl =0
    E0 = E_ampl*k*wc**2 #0.0005 # Amplitude of driving laser
    param_cav = [wc, E0]  # wc, E
    #freqs = np.random.normal(wm, 0.0002, num_mol)  # wm
    freqs = [0.0158667, 0.01628286, 0.01618053, 0.0160544,  0.01621614, 0.01600337, 0.01590205, 0.01618737, 0.01633669, 0.01622438]
    gc =0.0
    gamma = gc/np.sqrt(num_mol) # light-matter coupling
    param_mol = freqs + [gamma] # wm, gamma
    mu   = 1
    kT   = 0.000944*0.0 # 9.44x10⁻4 au is value of room temperature energy 25,7 meV
    sigma_c = np.sqrt(2*kT*k/mu)
    sigma_m = np.sqrt(2*kT*lamb/mu)

    ''' New initial conditions'''

    # Read initial positions from text file
    with open("initial_positions.txt", "r") as f_pos:
        # Read init_xc and init_xm
        init_xc = float(f_pos.readline().strip())
        init_xm = list(map(float, f_pos.readline().split()))

    # Read initial velocities from text file
    with open("initial_velocities.txt", "r") as f_vel:
        # Read init_vc and init_vm
        init_vc = float(f_vel.readline().strip())
        init_vm = list(map(float, f_vel.readline().split()))

    # Combine initial conditions into a single list
    init_cond = [init_xc, init_vc] + init_xm + init_vm


    # Define time pointsp
    t_final = 100000
    t_step   = 10
    time_points = np.arange(0, t_final, t_step)  # Time points from 0 to 10

    # Solve the system using Velocity-Verlet algorithm
    xc_values, vc_values, xm_values, vm_values = velocity_verlet(init_cond, time_points, num_mol, param_cav, param_mol)

    # Solve the uncoupled system for comparison
    param_mol = freqs + [0]
    xc_values_u, vc_values_u, xm_values_u, vm_values_u = velocity_verlet(init_cond, time_points, num_mol, param_cav, param_mol)



    # Define bright state: 
    xb = np.empty(len(time_points))

    for i in range(0,len(time_points)):
        xb[i] = 1/np.sqrt(num_mol) * np.sum(xm_values[:,i])


    ################################################################
    ''' COMPUTE POSITION AUTOCORRELATION FUNCTION '''
    ################################################################

    # Molecular position autocorrelation function
    C_xmxm = np.zeros((len(xm_values), len(time_points)))

    for mol in range(len(xm_values)):
        C_xmxm[mol] = xm_values[mol,0]*xm_values[mol,:]/xm_values[mol,0]**2

    #C_xmxm_fac = xm_values / xm_values[:, 0:1] # Alternatively one can use this super compact version but I think it's harder to read

    # Photon position autocorrelation function
    C_xcxc = xc_values[0]*xc_values / xc_values[0]**2 
    # Bright state position autocorrelation function
    C_xbxb = xb[0]*xb / xb[0]**2

    '''Calculate correlation functions for uncoupled system'''

    # Molecular position autocorrelation function
    C_xmxm_u = np.zeros((len(xm_values_u), len(time_points)))

    for mol in range(len(xm_values_u)):
        C_xmxm_u[mol] = xm_values_u[mol,0]*xm_values_u[mol,:]/xm_values_u[mol,0]**2

    # Photon position autocorrelation function
    C_xcxc_u = xc_values_u[0]*xc_values_u / xc_values_u[0]**2 

     
    ################################################################
    ''' COMPUTE THE FFT OF THE TIME EVOLUTION OF EACH OSCILLATOR '''
    ################################################################


    # Compute FFTs and frequencies for xm_values and xc_values
    fft_xm = [np.fft.fft(xm) for xm in C_xmxm]
    fftfreq_xm = [2*np.pi*np.fft.fftfreq(fft.shape[-1], d=t_step) for fft in fft_xm]

    fft_xm_late = [np.fft.fft(xm[-100000:]) for xm in C_xmxm]
    fftfreq_xm_late = [2*np.pi*np.fft.fftfreq(fft.shape[-1], d=t_step) for fft in fft_xm_late]
    av_fft_late = np.mean(fft_xm_late, axis=0)

    fft_xm_early = [np.fft.fft(xm[:20000]) for xm in C_xmxm]
    fftfreq_xm_early = [2*np.pi*np.fft.fftfreq(fft.shape[-1], d=t_step) for fft in fft_xm_early]

    fft_xc = np.fft.fft(C_xcxc)
    fftfreq_xc = 2*np.pi*np.fft.fftfreq(fft_xc.shape[-1], d=t_step)

    fft_xc_late = np.fft.fft(C_xcxc[-10000:])
    fftfreq_xc_late = 2*np.pi*np.fft.fftfreq(fft_xc_late.shape[-1], d=t_step)

    fft_xc_early = np.fft.fft(C_xcxc[:20000])
    fftfreq_xc_early = 2*np.pi*np.fft.fftfreq(fft_xc_early.shape[-1], d=t_step)

    fft_xb = np.fft.fft(C_xbxb)
    fftfreq_xb = 2*np.pi*np.fft.fftfreq(fft_xb.shape[-1], d=t_step)

    fft_xb_late = np.fft.fft(C_xbxb[-100000:])
    fftfreq_xb_late = 2*np.pi*np.fft.fftfreq(fft_xb_late.shape[-1], d=t_step)

    fft_xb_early = np.fft.fft(C_xbxb[:20000])
    fftfreq_xb_early = 2*np.pi*np.fft.fftfreq(fft_xb_early.shape[-1], d=t_step)

    # Compute frequencies of the uncoupled system

    fft_xm_u = [np.fft.fft(xm) for xm in C_xmxm_u]
    fftfreq_xm_u = [2*np.pi*np.fft.fftfreq(fft.shape[-1], d=t_step) for fft in fft_xm_u]
    av_fft_u = np.mean(fft_xm_u, axis=0)

    fft_xc_u = np.fft.fft(C_xcxc_u)
    fftfreq_xc_u = 2*np.pi*np.fft.fftfreq(fft_xc.shape[-1], d=t_step)


    # Find main frequencies for xm_values late
    main_frequencies_late = []
    for i, fft_late in enumerate(fft_xm_late):
        ampl = np.abs(fft_late)
        peaks, _ = find_peaks(ampl)
        top_peak = nlargest(1, peaks, key=lambda i: ampl[i])
        main_freq_late = fftfreq_xm_late[i][top_peak]
        main_frequencies_late.append(main_freq_late)
        #print(f'Main frequency for xm_values[{i}] (late):', main_freq_late)
    #print('main_frequencies:', main_frequencies_late)


    ''' Define path of directory for saving stuff'''

    dir = f"{gc}_{E_ampl}"

    # Create the directory if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Define the full file path (join the directory with the desired filename)
    file_path = os.path.join(dir, 'figure_name.png')




    ###################################################
    ''' SAVE IMPORTANT QUANTITIES IN EXTERNAL FILES '''
    ###################################################
    # Save the averaged spectra in a txt file 
    np.savetxt('av_fft_u', np.abs(av_fft_u.real))
    np.savetxt('fftfreq_u', fftfreq_xm_u[0])

    np.savetxt('av_fft_late', np.abs(av_fft_late.real))
    np.savetxt('fftfreq_late', fftfreq_xm_late[0])


    with open('freqs.txt', 'a') as f:
        for item in main_frequencies_late:
            item = np.abs(item[0])
            f.write(f"{item}\n")

    ######################
    ''' PLOT DYNAMICS '''
    ######################


    # Plot the results
    plt.figure(figsize=[18,12])
    plt.title('Coupled Generalized Langevin eq with friction and stochastic term')

    plt.subplot(3,1,1)
    plt.title('Photon and Molecules')
    plt.plot(time_points/41000, xc_values, label='xc(t)', color='black')
    #for i in range(num_mol):
    #    plt.plot(time_points/41000, xm_values[i], label=f'xm{i+1}(t)')
    #plt.xlim(4000,5000)
    #plt.xticks(np.arange(0, t_final+1, 100))
    plt.xlabel('Time (ps)')
    plt.ylabel('Values')

    plt.subplot(3,1,2)
    plt.title('Molecules + photon')
    plt.plot(time_points/41000, xc_values, label='xc(t)', color='black')
    for i in range(num_mol):
        plt.plot(time_points/41000, xm_values[i], label=f'xm{i+1}(t)')
    #plt.ylim(-10,10)
    plt.xlim(0,0.4)
    plt.xlabel('Time')
    plt.ylabel('Values')

    plt.subplot(3,1,3)
    plt.title('Molecules + photon')
    plt.plot(time_points/41000, xc_values, label='xc(t)', color='black')
    for i in range(num_mol):
        plt.plot(time_points/41000, xm_values[i], label=f'xm{i+1}(t)')
    plt.ylim(-0.1,0.1)
    plt.xlim(19.8,20)
    plt.xlabel('Time')
    plt.ylabel('Values')

    # Define the full file path (join the directory with the desired filename)
    file_path1 = os.path.join(dir, 'dynamics.png')

    plt.savefig(file_path1, format='png', dpi=300) 


    ################################
    ''' PLOT FREQUENCIES SPECTRA '''
    ################################


    # Plot results
    plt.figure(figsize=[15, 12])
    # Plot for all frequencies
    ax = plt.subplot(2, 2, 1)
    for i in range(len(fft_xm)):
        ax.plot(fftfreq_xm_early[i], np.abs(fft_xm_early[i].real)/np.max(np.abs(fft_xm_early[i].real)), label=f'xm_values[{i}]')
    ax.plot(fftfreq_xc_early, np.abs(fft_xc_early.real)/np.max(np.abs(fft_xc_early.real)), color='black', label='xc_values')
    ax.plot(fftfreq_xb_early, np.abs(fft_xb_early.real)/np.max(np.abs(fft_xb_early.real)), color='red', linestyle=(0, (5, 10)), label='xb_values')
    ax.set_xlim(0.012, 0.022)
    ax.set_ylim(0.0, 2.0)
    ax.set_title("Early Signal FFT")
    #ax.legend(loc=4)

    ax2 = ax.twinx()
    for i in range(len(fft_xm_u)):
        ax2.plot(fftfreq_xm_u[i], np.abs(fft_xm_u[i].real)/np.max(np.abs(fft_xm_u[i].real)), label=f'xm_values[{i}] uncoupled', linestyle='--')
    ax2.plot(fftfreq_xc_u, np.abs(fft_xc_u.real)/np.max(np.abs(fft_xc_u.real)), color='black', linestyle='--', label='xc uncoupled')
    ax2.set_ylim(0.0, 2.0)
    ax2.invert_yaxis()  # Flip the secondary y-axis
    #ax2.legend(loc=1)

    # Plot for late signal frequencies
    ax = plt.subplot(2, 2, 2)
    for i in range(len(fft_xm_late)):
        ax.plot(fftfreq_xm_late[i], np.abs(fft_xm_late[i].real)/np.max(np.abs(fft_xm_late[i].real)), label=f'xm_values[{i}] Late')
    #ax.plot(fftfreq_xm_late[0], np.abs(av_fft_late.real)/np.max(np.abs(av_fft_late.real)), label=f'average xm_values Late')
    #for i in range(len(main_frequencies_late)):
     #   ax.axvline(x=np.abs(main_frequencies_late[i]), linestyle='dashed')
    #ax.plot(fftfreq_xc_late, np.abs(fft_xc_late.real)/np.max(np.abs(fft_xc_late.real)), color='black', label='xc_values Late')
    #plt.plot(fftfreq_xb_late, np.abs(fft_xb_late.real)/np.max(np.abs(fft_xb_late.real)), color='gray', label='xb_values Late')
    ax.set_xlim(0.012, 0.022)
    ax.set_ylim(0.0, 2.0)
    ax.set_title("Late Signal FFT")

    ax2 = ax.twinx()
    for i in range(len(fft_xm_u)):
        ax2.plot(fftfreq_xm_u[i], np.abs(fft_xm_u[i].real)/np.max(np.abs(fft_xm_u[i].real)), label=f'xm_values[{i}] uncoupled', linestyle='--')
    #ax2.plot(fftfreq_xm_u[0], np.abs(av_fft_u.real)/np.max(np.abs(av_fft_u.real)), label=f'average xm_values uncoupled', linestyle='--')
    ax2.plot(fftfreq_xc_u, np.abs(fft_xc_u.real)/np.max(np.abs(fft_xc_u.real)), color='black', linestyle='--', label='xc_values uncoupled (flipped)')
    ax2.set_ylim(0.0, 2.0)
    ax2.invert_yaxis()  # Flip the secondary y-axis
    #plt.legend()

    file_path2 = os.path.join(dir, 'spectra.png')

    plt.savefig(file_path2, format='png', dpi=300) 
    
    
