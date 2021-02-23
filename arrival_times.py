import numpy as np
from scipy.constants import value,nano

class ArrivalTimes():
    c = value('speed of light in vacuum')
    def __init__(self,shower_R,shower_dh,shower_delta,travel_length,cQ,shower_rms_w,tel_q,ng):
        self.axis_time = -shower_R/self.c/nano
        self.vertical_delay = np.cumsum((shower_delta*shower_dh)[::-1])[::-1]/self.c/nano
        self.counter_time = self.axis_time + travel_length/self.c/nano + self.vertical_delay/cQ
        self.diff = self.calculate_diff(shower_rms_w,travel_length,tel_q)
        self.t_d = self.calculate_delay(self.diff,shower_delta)
        # self.ng_prime = self.smear_bins(self.counter_time,self.t_d,ng)

    def calculate_diff(self,shower_rms_w,travel_length,tel_q):
        shower_rms_w = np.array([shower_rms_w,] * travel_length.shape[0])
        r_prime = np.sqrt(shower_rms_w**2 + travel_length**2 -2*shower_rms_w*travel_length*np.cos(np.pi/2-tel_q))
        return np.abs(travel_length - r_prime)

    def calculate_delay(self,diff,shower_delta):
        c_n = self.c / (shower_delta + 1)
        c_n = np.array([c_n,] * diff.shape[0])
        t_d = diff / c_n
        return t_d/nano

    def smear_bins(self,counter_time,t_d,ng):
        ng_prime = np.empty_like(ng)
        for i in range(ng.shape[0]):
            for j in range(ng.shape[1]):
                t = counter_time[i,:]
                t_0 = counter_time[i,j]
                dist = self.gaussian(t,t_0,t_d[i,j])
                dist *= ng[i,j]/np.sum(dist)
                ng_prime[i,:] += dist
        return ng_prime


    def gaussian(self,t,t_0,w):
        return (np.exp(-.5*((t-t_0)/w)**2))
