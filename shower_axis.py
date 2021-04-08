import numpy as np
from scipy.constants import value,nano
import atmosphere as at

class Shower():
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
    phi: azimuthal angle of axis intercept (radians) measured from the x
    axis. Standard physics spherical coordinate convention. Positive x axis is
    North, positive y axis is west.
    ground_level: Height above sea level of center of shower footprint (meters)
    type: Shower type, either 'GN' for Greisen, or GH for Gaisser-Hillas.
    """
    earth_radius = 6.371e6
    c = value('speed of light in vacuum')
    Lambda = 70
    atm = at.Atmosphere()
    Moliere_data = np.load('lateral.npz')
    t_Moliere = Moliere_data['t']
    AVG_Moliere = Moliere_data['avg']
    theta_upper_limit = np.pi/2
    theta_lower_limit = 0

    def __init__(self,X_max,N_max,h0,theta,direction,phi=0,ground_level=0,type='GH'):
        if theta < self.theta_lower_limit or theta > self.theta_upper_limit:
            raise Exception("Theta value out of bounds")
        self.reset_shower(X_max,N_max,h0,theta,direction,phi,ground_level,type)

    def reset_shower(self,X_max,N_max,h0,theta,direction,phi,ground_level,type):
        '''Set necessary attributes and perform calculations
        '''
        self.type = type
        self.input_X_max = X_max
        self.N_max = N_max
        self.h0 = h0
        self.direction = direction
        self.theta = theta
        self.phi = phi
        self.axis_h = np.linspace(ground_level+1,self.atm.maximum_height,10000)
        self.axis_rho = self.atm.density(self.axis_h)
        self.axis_delta = self.atm.delta(self.axis_h)
        self.axis_h -= ground_level
        self.axis_Moliere = 96. / self.axis_rho
        axis_midh = np.sqrt(self.axis_h[1:]*self.axis_h[:-1])
        self.axis_dh = np.empty_like(self.axis_h)
        self.axis_dh[1:-1] = np.abs(axis_midh[:-1]-axis_midh[1:])
        self.axis_dh[-1] = self.axis_dh[-2]
        self.axis_dh[0] = self.axis_dh[1]
        self.ground_level = ground_level
        self.earth_radius += ground_level # adjust earth radius
        self.axis_r = self.h_to_axis_R_LOC(self.axis_h, theta)
        self.theta_difference = theta - self.theta_normal(self.axis_h, self.axis_r)
        self.axis_start_r = self.h_to_axis_R_LOC(h0, theta)
        self.axis_X, self.axis_dr, self.X0 = self.set_depth(self.axis_r,
                self.axis_start_r)
        self.X_max = X_max + self.X0
        self.axis_nch = self.size(self.axis_X)
        self.axis_nch[self.axis_nch<1.e3] = 0
        self.i_ch = np.nonzero(self.axis_nch)[0]
        self.shower_X = self.axis_X[self.i_ch]
        self.shower_r = self.axis_r[self.i_ch]
        self.shower_dr = self.axis_dr[self.i_ch]
        self.shower_nch = self.axis_nch[self.i_ch]
        self.shower_Moliere = self.axis_Moliere[self.i_ch]
        self.shower_delta = self.axis_delta[self.i_ch]
        self.shower_h = self.axis_h[self.i_ch]
        self.shower_dh = self.axis_dh[self.i_ch]
        self.shower_t = self.stage(self.shower_X,self.X_max)
        self.shower_avg_M = np.interp(self.shower_t,self.t_Moliere,self.AVG_Moliere)
        self.shower_rms_w = self.shower_avg_M * self.shower_Moliere

    @classmethod
    def h_to_axis_R_LOC(cls,h,theta):
        '''Return the length along the shower axis from the point of Earth
        emergence to the height above the surface specified

        Parameters:
        h: array of heights (m above sea level)
        theta: polar angle of shower axis (radians)

        returns: r (m) (same size as h), an array of distances along the shower
        axis_sp.
        '''
        cos_EM = np.cos(np.pi-theta)
        R = cls.earth_radius
        r_CoE= h + R # distance from the center of the earth to the specified height
        return R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)

    @classmethod
    def theta_normal(cls,h,r):
        ''' Convert a polar angle (at a given height) with respect to the z axis
        to a polar angle with respect to vertical in the atmosphere (at that
        height)

        Parameters:
        h: array of heights (m above sea level)
        theta: array of polar angle of shower axis (radians)

        Returns:
        The corrected angles(s)
        '''
        cq = (r**2 + h**2 + 2*cls.earth_radius*h)/(2*r*(cls.earth_radius+h))
        # cq = ((cls.earth_radius+h)**2+r**2-cls.earth_radius**2)/(2*r*(cls.earth_radius+h))
        return np.arccos(cq)

    def set_depth(self,axis_r,axis_start_r):
        '''Integrate atmospheric density over selected direction to create
        a table of depth values.

        Parameters:
        axis_r: distances along the shower axis
        axis_start_r: distance along the axis where the shower starts

        returns:
        axis_X: depths at each axis distances (g/cm^2)
        axis_dr: corresponding spatial distance associated with each depth (m)
        X0: start depth (g/cm^2)
        '''
        axis_dr = axis_r[1:] - axis_r[:-1]
        axis_deltaX = np.sqrt(self.axis_rho[1:]*self.axis_rho[:-1])*axis_dr / 10# converting to g/cm^2
        if self.direction == 'up':
            axis_X = np.concatenate((np.array([0]),np.cumsum(axis_deltaX)))
        elif self.direction == 'down':
            axis_X = np.concatenate((np.cumsum(axis_deltaX[::-1])[::-1],
                    np.array([0])))
        axis_dr = np.concatenate((np.array([0]),axis_dr))
        X0 = np.interp(axis_start_r,axis_r,axis_X)
        return axis_X, axis_dr, X0

    def size(self,X):
        """Return the size of the shower at a slant-depth X

        Parameters:
            X: the slant depth at which to calculate the shower size [g/cm2]

        Returns:
            N: the shower size (# of charged particles)
        """
        if self.type == 'GH':
            value = self.GaisserHillas(X)
        elif self.type == 'GN':
            value = self.Greisen(X)
        return value

    def GaisserHillas(self,X):
        '''Return the size of a GH shower at a given depth.
        Parameters:
        X: depth

        Returns:
        # of charged particles
        '''
        x =         (self.axis_X-self.X0)/self.Lambda
        g0 = x>0.
        m = (self.X_max-self.X0)/self.Lambda
        n = np.zeros_like(x)
        n[g0] = np.exp( m*(np.log(x[g0])-np.log(m)) - (x[g0]-m) )
        return self.N_max * n

    def Greisen(self,X_in,p=36.62):
        '''Return the size of a Greisen shower at a given depth.
        Parameters:
        X_in: depth

        Returns:
        # of charged particles
        '''
        X = [x if x > self.X0 else self.X0 for x in X_in]
        Delta = X - self.X_max
        W = self.X_max-self.X0
        eps = Delta / W
        s = (1+eps)/(1+eps/3)
        i = np.nonzero(s)
        n = np.zeros_like(X)
        n[i] = np.exp((eps[i]*(1-1.5*np.log(s[i]))-1.5*np.log(s[i]))*(W/p))
        return self.N_max * n

    @classmethod
    def stage(cls,X,X_max,X0=36.62):
        """Return the shower stage at a given slant-depth X. This
        is after Lafebre et al.

        Parameters:
            X: atmosphering slant-depth [g/cm2]
            X0: radiation length of air [g/cm2]

        Returns:
            t: shower stage
        """
        return (X-X_max)/X0

if __name__ == '__main__':
    import time
    import matplotlib
    import matplotlib.pyplot as plt
    start_time = time.time()
    sh = Shower(700,1.e7,10000,np.radians(85),'up')
    end_time = time.time()
    print("Calculations take: %.3f s"%(
        end_time-start_time))

    x = sh.axis_r * np.sin(sh.theta)
    z = sh.axis_r * np.cos(sh.theta)

    arc_angle = 5
    arc = np.linspace(-np.radians(arc_angle),np.radians(arc_angle),100)
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
    plt.quiver(x_shower,z_shower,x_width,z_width, angles='xy', scale_units='xy', scale=1,label='shower width')
    plt.quiver(x_shower,z_shower,-x_width,-z_width, angles='xy', scale_units='xy', scale=1)
    plt.plot(x_shower,z_shower,'r',label='Cherenkov region')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.title('Downward Shower 5 degree EE')
    ax.set_aspect('equal')

    plt.figure()
    plt.plot(sh.axis_r,sh.axis_X)
    plt.plot(sh.shower_r,sh.shower_X)
    plt.scatter(sh.axis_start_r,sh.X0)

    plt.figure()
    plt.plot(sh.axis_X,sh.axis_nch)
    plt.show()
