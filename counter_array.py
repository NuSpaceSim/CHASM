import numpy as np
from iact_read import IACTread
from scipy.constants import value,nano
from shower_axis import Shower as sh

class CounterArray(sh):
    '''Class for calculating Cherenkov yield of upward going showers at a
    hypothetical orbital telescope array_z

    Parameters:
    shower_r: array of distances along shower axis (m)
    n_tel: number of telescopes
    tel_distance: how far they are along the axis from first interaction point (m)
    tel_area: surface area of telescopes (m^2)

    '''

    def __init__(self,X_max,N_max,h0,theta,direction,tel_vectors):
        super().__init__(X_max,N_max,h0,theta,direction)
        if tel_vectors.shape[1] != 3 or len(tel_vectors.shape) != 2:
            raise Exception("tel_vectors is not an array of vectors.")
        self.reset_array(tel_vectors)

    def reset_array(self,tel_vectors):
        self.axis_vectors = self.set_axis_vectors(self.shower_r, self.theta, self.phi)
        self.tel_vectors = tel_vectors
        self.tel_area = 1
        self.tel_q, self.tel_omega, self.travel_length, self.cQ = self.set_travel_params(self.axis_vectors,self.tel_vectors,self.tel_area)

    def set_axis_vectors(self,shower_r,theta, phi):
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        axis_vectors = np.empty([np.shape(shower_r)[0],3])
        axis_vectors[:,0] = shower_r * st * cp
        axis_vectors[:,1] = shower_r * st * sp
        axis_vectors[:,2] = shower_r * ct
        return axis_vectors

    def set_travel_params(self,axis_vectors,tel_vectors,tel_area):
        travel_vectors = tel_vectors.reshape(-1,1,3) - axis_vectors
        travel_length =  np.sqrt( (travel_vectors**2).sum(axis=2) )
        axis_r = np.sqrt( (axis_vectors**2).sum(axis=1) )
        tel_r = np.sqrt( (tel_vectors**2).sum(axis=1) )

        travel_n = travel_vectors/travel_length[:,:,np.newaxis]
        travel_cQ = np.abs(travel_n[:,:,-1])
        axis_length = np.broadcast_to(axis_r,travel_length.shape)
        tel_length = np.broadcast_to(tel_r,travel_length.T.shape).T

        cq = (tel_length**2-axis_length**2-travel_length**2)/(-2*axis_length*travel_length) #cosine of angle between axis and vector
        tel_q = np.arccos(cq)
        if self.direction == 'up':
            tel_q = np.pi - tel_q
        tel_omega = tel_area / travel_length **2
        return tel_q, tel_omega, travel_length, travel_cQ
