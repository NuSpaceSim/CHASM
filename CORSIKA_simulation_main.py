import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from shower_axis import Shower
from counter_array import CounterArray
from cherenkov_yield import CherenkovYield
from iact_read import IACTread
from arrival_times import ArrivalTimes

start_time = time.time()
sh = IACTread('/home/isaac/Cherenkov/corsika_dat/iact_s_000102.dat')
# gc = GroundCounters(sh.tel_vectors,sh.r, sh.theta, sh.phi,sh.tel_area)
gc = CounterArray(sh.tel_vectors,sh.r, sh.theta, sh.phi,sh.direction, sh.tel_area)
cy = CherenkovYield(sh.t,sh.delta,gc.tel_q,sh.nch,sh.dr,gc.tel_omega,sh.tel_dE)
at = ArrivalTimes(sh.r,sh.dh,sh.delta,gc.travel_length,gc.cQ,sh.shower_rms_w,gc.tel_q,cy.ng)
end_time = time.time()
print("Calculations take: %.3f s"%(
    end_time-start_time))

plt.ion()
plt.figure(0)
plt.plot(gc.tel_vectors[:,0],cy.ng_sum,'.',label = 'universality')
plt.plot(gc.tel_vectors[:,0],sh.ng,'.',label = 'CORSIKA')
plt.loglog()
plt.legend()
plt.xlabel('Counter Position [m from axis]')
plt.ylabel('Number of photons')
# plt.suptitle('Cherenkov Lateral Distribution at Altitude 525 Km')
# plt.title('(5 degree Earth emergence angle)')
plt.legend()
plt.grid()

# c2plot = np.array([0,1,2,5,8,10,12,15,20,25,30,35,40,45])
c2plot = np.array([0,1,2,5,8])

for i,ct in enumerate(c2plot):
    plt.figure(i+1)
    ha = plt.hist(sh.ev.photon_bunches[ct]['time'],
                  sh.iact_ghnb[ct],(sh.iact_ghmn[ct],sh.iact_ghmx[ct]),
                  weights=sh.ev.photon_bunches[ct]['photons'],
                  histtype='step',label='CORSIKA-IACT')
    hb = plt.hist(at.counter_time[ct],
                  sh.iact_ghnb[ct],(sh.iact_ghmn[ct],sh.iact_ghmx[ct]),
                  weights=cy.ng[ct],
                  histtype='step',label='Universality')
    # plt.yscale('log')
    if (sh.iact_gmxt[ct]-sh.iact_g99t[ct])/sh.iact_ghdt[ct] > 10.:
        plt.xlim(sh.iact_gmnt[ct],sh.iact_g99t[ct])
    else:
        plt.xlim(sh.iact_gmnt[ct],sh.iact_gmxt[ct])
    plt.title('Counter %d'%ct)
    plt.xlabel('Arrival Time [nS]')
    plt.ylabel('Number of Cherenkov Photons')
    plt.grid()
    plt.legend()

# for i,ct in enumerate(c2plot):
#     plt.figure(i+1)
#     ha = plt.hist(sh.ev.photon_bunches[ct]['time'],
#                   sh.iact_ghnb[ct],(sh.iact_ghmn[ct],sh.iact_ghmx[ct]),
#                   weights=sh.ev.photon_bunches[ct]['photons'],
#                   histtype='step',label='CORSIKA-IACT')
#     hb = plt.hist(at.counter_time[ct],
#                   sh.iact_ghnb[ct],(sh.iact_ghmn[ct],sh.iact_ghmx[ct]),
#                   weights=at.ng_prime[ct],
#                   histtype='step',label='Universality')
#     # plt.yscale('log')
#     if (sh.iact_gmxt[ct]-sh.iact_g99t[ct])/sh.iact_ghdt[ct] > 10.:
#         plt.xlim(sh.iact_gmnt[ct],sh.iact_g99t[ct])
#     else:
#         plt.xlim(sh.iact_gmnt[ct],sh.iact_gmxt[ct])
#     plt.title('Counter %d'%ct)
#     plt.xlabel('Arrival Time [nS]')
#     plt.ylabel('Number of Cherenkov Photons')
#     plt.grid()
#     plt.legend()

plt.show()
