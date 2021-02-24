from cherenkov_yield import CherenkovYield as cy

class CHASM(cy):
    """A class for generating extensive air shower profiles and their Cherenkov
    outputs. The shower can either be a Gaisser Hillas shower or a Griessen
    Shower. The Cartesian origin is at the point where the shower axis intersects
    with the Earth's surface.

    Parameters:
    X_max: depth at shower max (g/cm^2)
    N_max: number of charged particles at X_max
    h0: height of first interaction above the ground level (meters)
    X0: Start depth
    theta: Polar angle of the shower axis with respect to vertical. Vertical
    is defined as normal to the Earth's surface at the point where the axis
    intersects with the surface.
    direction: Shower direction, either 'up' for upward going showers, or 'down'
    for downward going showers.
    tel_vectors: user defined array of cartesian telescope locations
    min_l: minimum accepted Cherenkov wavelength
    max_l: maximum accepted Cherenkov wavelength
    """
    def __init__(self,X_max,N_max,h0,theta,direction,tel_vectors,min_l,max_l):
        super().__init__(X_max,N_max,h0,theta,direction,tel_vectors,min_l,max_l)

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    tel_vectors = np.empty([100,3])

    theta = np.radians(85)
    phi = 0
    r = 2141673.2772862054

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    tel_vectors[:,0] = np.full(100,x)
    tel_vectors[:,1] = np.linspace(y-100.e3,y+100.e3,100)
    tel_vectors[:,2] = np.full(100,z)

    start_time = time.time()
    CHASM = CHASM(765,8000.e4,0,np.radians(85),'up',tel_vectors,300,600)
    end_time = time.time()
    print("Calculations take: %.3f s"%(
        end_time-start_time))

    x = CHASM.axis_r * np.sin(CHASM.theta)
    z = CHASM.axis_r * np.cos(CHASM.theta)

    arc_angle = 5
    arc = np.linspace(-np.radians(arc_angle),3*np.radians(arc_angle),100)
    x_surf = CHASM.earth_radius * np.sin(arc)
    z_surf = CHASM.earth_radius * np.cos(arc) - CHASM.earth_radius

    x_shower = CHASM.shower_r * np.sin(CHASM.theta)
    z_shower = CHASM.shower_r * np.cos(CHASM.theta)

    x_width = -CHASM.shower_rms_w * np.cos(CHASM.theta)
    z_width = CHASM.shower_rms_w * np.sin(CHASM.theta)

    plt.ion()

    plt.figure()
    ax = plt.gca()
    plt.plot(x,z,label='shower axis' )
    plt.plot(x_surf,z_surf,label="Earth's surface")
    plt.scatter(CHASM.tel_vectors[:,0],CHASM.tel_vectors[:,2], label='telescopes')
    plt.quiver(x_shower,z_shower,x_width,z_width, angles='xy', scale_units='xy', scale=1,label='shower width')
    plt.quiver(x_shower,z_shower,-x_width,-z_width, angles='xy', scale_units='xy', scale=1)
    plt.plot(x_shower,z_shower,'r',label='Cherenkov region')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.title('Downward Shower 5 degree EE')
    ax.set_aspect('equal')
    plt.grid()
    # Plot Cherenkov distributions of showers that start at different heights
    plt.ion()
    plt.figure()
    plt.plot(CHASM.tel_vectors[:,1]/1000,CHASM.ng_sum)
    plt.semilogy()
    plt.xlabel('Counter Position [km from axis]')
    plt.ylabel('Photon Flux [$m^{-2}$]')
    plt.suptitle('Cherenkov Lateral Distribution at Altitude 525 Km')
    plt.title('(5 degree Earth emergence angle)')
    # plt.legend(title = 'Starting Altitude')
    plt.grid()

    plt.figure()
    hb = plt.hist(CHASM.counter_time[99],
                      100,
                      weights=CHASM.ng[99],
                      histtype='step',label='no correction')
    hc = plt.hist(CHASM.counter_time_prime[99],
                      100,
                      weights=CHASM.ng[99],
                      histtype='step',label='correction')
    plt.title('Preliminary Arrival Time Distribution (100 Km from axis)')
    plt.suptitle('(5 degree EE, start height = 0 m, counter height = 525 Km, Xmax = 500 g/cm^2, Nmax = 1.e8)')
    plt.xlabel('Arrival Time [nS]')
    plt.ylabel('Number of Cherenkov Photons')
    plt.legend()
    plt.show()
