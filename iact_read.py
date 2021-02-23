import sys
import numpy as np
import eventio as ei
from shower_axis import Shower as sh
import atmosphere as at
from scipy.constants import value,nano

class IACTread():
    def __init__(self,infile):
        f = ei.IACTFile(infile)
        ev = self.get_event(infile,0)
        atm = at.Atmosphere()

        Moliere_data = np.load('lateral.npz')
        t_Moliere = Moliere_data['t']
        AVG_Moliere = Moliere_data['avg']

        c = value('speed of light in vacuum')
        hc = value('Planck constant in eV s') * c

        # Get CORSIKA parameters
        self.theta = ev.header[10] # this is already in radians
        if self.theta > np.pi/2:
            self.direction = 'up'
        else:
            self.direction = 'down'
        iact_cq = np.cos(self.theta)
        self.phi = ev.header[11] + np.pi# not sure of this index
        self.tel_area = (np.pi*(f.telescope_positions['r']/100.)**2)[0]
        nl = ev.longitudinal['nthick']
        iact_obs = f.header[5][0]/100.
        self.X = (np.arange(nl,dtype=float)*ev.longitudinal['thickstep'])
        self.nch = ev.longitudinal['data'][6]
        self.Xmax = self.X[self.nch.argmax()]

        # Calculate parameters needed for universality calculation
        atm_h = (np.arange(np.ceil(atm.maximum_height/100))*100)[::-1]
        atm_h2 = np.roll(atm_h,1)
        atm_h2[0] = atm_h[0]+100
        atm_deltaX = atm.depth(atm_h,atm_h2)
        atm_X = np.cumsum(atm_deltaX)

        iact_E1 = ev.header[57]
        iact_E2 = ev.header[58]
        self.tel_dE = hc/(ev.header[57]*nano) - hc/(ev.header[58]*nano) # Cherenkov wavelengths
        self.nu_dE = hc/(300*nano) - hc/(600*nano) # Cherenkov wavelengths

        self.X *= iact_cq
        iact_h = np.interp(self.X,atm_X,atm_h)
        self.X /= iact_cq
        self.delta = atm.delta(iact_h)
        self.r = (iact_h-iact_obs)/iact_cq
        self.iact_h = iact_h

        iact_midh = np.sqrt(iact_h[1:]*iact_h[:-1])
        iact_dh = np.empty_like(iact_h)
        iact_dh[1:-1] = iact_midh[:-1]-iact_midh[1:]
        iact_dh[-1] = iact_dh[-2]
        iact_dh[0] = iact_dh[1]
        self.dh = iact_dh
        self.dr = iact_dh/iact_cq

        # self.axis_vectors = self.set_axis_vectors(self.r,self.theta,self.phi)
        self.tel_vectors = self.set_iact_pos(f)
        self.t = sh.stage(self.X,self.Xmax)

        self.shower_avg_M = np.interp(self.t,t_Moliere,AVG_Moliere)
        self.shower_Moliere = 96. / atm.density(iact_h)
        self.shower_rms_w = self.shower_avg_M * self.shower_Moliere
        self.shower_rms_w[self.shower_rms_w > 1.e3] = 1

        # Get CORSIKA photon info
        self.nc = self.tel_vectors.shape[0]
        self.ng = np.array([ev.n_photons[i] for i in range(self.nc)])

        iact_gmnt = np.array([ev.photon_bunches[i]['time'].min() for i in range(self.nc)])
        iact_gmxt = np.array([ev.photon_bunches[i]['time'].max() for i in range(self.nc)])
        iact_g05t = np.array([np.percentile(ev.photon_bunches[i]['time'], 5.) for i in range(self.nc)])
        iact_g95t = np.array([np.percentile(ev.photon_bunches[i]['time'],95.) for i in range(self.nc)])
        iact_gd90 = iact_g95t-iact_g05t
        iact_g01t = np.array([np.percentile(ev.photon_bunches[i]['time'], 1.) for i in range(self.nc)])
        iact_g99t = np.array([np.percentile(ev.photon_bunches[i]['time'],99.) for i in range(self.nc)])
        iact_ghdt = np.ones_like(iact_gmnt)
        iact_ghdt[iact_gd90<30.]   = 0.2
        iact_ghdt[iact_gd90>100.] = 5.
        iact_ghmn = np.floor(iact_gmnt)
        iact_ghmn[iact_ghdt==5.] = 5*np.floor(iact_ghmn[iact_ghdt==5.]/5.)
        iact_ghmx = np.ceil(iact_gmxt)
        iact_ghmx[iact_ghdt==5.] = 5*np.ceil(iact_ghmx[iact_ghdt==5.]/5.)
        self.iact_ghnb = ((iact_ghmx-iact_ghmn)/iact_ghdt).astype(int)
        self.iact_ghmn = iact_ghmn
        self.iact_ghmx = iact_ghmx
        self.iact_gmxt = iact_gmxt
        self.iact_g99t = iact_g99t
        self.iact_ghdt = iact_ghdt
        self.iact_gmnt = iact_gmnt
        self.ev = ev

    def get_event(self,iact_file,event_no):
        with ei.IACTFile(iact_file) as f:
            for i,ev in enumerate(f):
                if i==event_no:
                    return ev
            else:
                return None

    def set_iact_pos(self,f):
        counter_x = f.telescope_positions['x']/100. # cm -> m
        counter_y = f.telescope_positions['y']/100. # cm -> m
        counter_z = f.telescope_positions['z']/100. # cm -> m
        tel_vectors = np.empty((counter_x.shape[0],3),dtype=float)
        tel_vectors[:,0] = counter_x
        tel_vectors[:,1] = counter_y
        tel_vectors[:,2] = counter_z
        return tel_vectors

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    iact = IACTread('/home/isaac/Cherenkov/corsika_dat/iact_s_000102.dat')
    plt.figure()
    plt.plot(iact.r,iact.delta)
