import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from cherenkov_yield import CherenkovYield
from arrival_times_new import ArrivalTimes

# Time upward shower calculations and instantiate UpwardShower
start_time = time.time()
# sh = Shower(765,8000.e4,0,np.radians(85),'up')

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

# gc = CounterArray(765,8000.e4,0,np.radians(85),'up',tel_vectors)
# cy = CherenkovYield(sh.shower_t,sh.shower_delta,gc.tel_q,sh.shower_nch,sh.shower_dr,gc.tel_omega,2.0664032897701547)
cy = CherenkovYield(765,8000.e4,0,np.radians(85),'up',tel_vectors)
at = ArrivalTimes(sh,gc)

end_time = time.time()
print("Calculations take: %.3f s"%(
    end_time-start_time))

x = sh.axis_r * np.sin(sh.theta)
z = sh.axis_r * np.cos(sh.theta)

arc_angle = 5
arc = np.linspace(-np.radians(arc_angle),3*np.radians(arc_angle),100)
x_surf = sh.earth_radius * np.sin(arc)
z_surf = sh.earth_radius * np.cos(arc) - sh.earth_radius

x_shower = sh.shower_r * np.sin(sh.theta)
z_shower = sh.shower_r * np.cos(sh.theta)

x_width = -sh.shower_rms_w * np.cos(sh.theta)
z_width = sh.shower_rms_w * np.sin(sh.theta)

plt.ion()
plt.figure()
ax = plt.gca()
plt.plot(x,z,label='shower axis' )
plt.plot(x_surf,z_surf,label="Earth's surface")
plt.scatter(gc.tel_vectors[:,0],gc.tel_vectors[:,2], label='telescopes')
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
plt.plot(gc.tel_vectors[:,1]/1000,cy.ng_sum)
plt.semilogy()
plt.xlabel('Counter Position [km from axis]')
plt.ylabel('Photon Flux [$m^{-2}$]')
plt.suptitle('Cherenkov Lateral Distribution at Altitude 525 Km')
plt.title('(5 degree Earth emergence angle)')
# plt.legend(title = 'Starting Altitude')
plt.grid()

plt.figure()
hb = plt.hist(at.counter_time[99],
                  100,
                  weights=cy.ng[99],
                  histtype='step',label='no correction')
hc = plt.hist(at.counter_time_prime[99],
                  100,
                  weights=cy.ng[99],
                  histtype='step',label='correction')
plt.title('Preliminary Arrival Time Distribution (100 Km from axis)')
plt.suptitle('(5 degree EE, start height = 0 m, counter height = 525 Km, Xmax = 500 g/cm^2, Nmax = 1.e8)')
plt.xlabel('Arrival Time [nS]')
plt.ylabel('Number of Cherenkov Photons')
plt.legend()
plt.show()
