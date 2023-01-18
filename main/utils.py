from math import pi, isnan, inf
from scipy.integrate import odeint
from scipy.optimize import bisect
from constants import *
from time import time

import matplotlib.pyplot as plt
import numpy as np


class Star:
    def __init__(self):
        self.r0          = 1e-15                                    #starting r. Ideal would be 0, but singularity is a thing
        #self.r           = np.linspace(self.r0, 30*R_sun, 1e7)     #radius range we wish to integrate over.  #currently obsolete
        self.hmax        = 1e8  #~0.1% sun resol
        self.hmin        = 5e2 #1km min resol
               
        self.dtau_list      = []
        self.rho_c_list     = []
        self.optimal_target = []

        
    def MS_star(self,central_T,save,file_name):
        '''
        Returns MS attributes for a star with given central Temperature
        '''
        start_time      = time()
        self.save       = save
        self.file_name  = file_name
        self.central_T  = central_T
        
        bisect(self.generate_trial_star, 0.3e3,500e3,xtol=1e-30) #Finds root of f(rho_c) function (finds rho_c for boundary conditions)
                                                                            #find rho_c to within 1 kg/m3
        
        correct_central_rho = self.rho_c_list[np.argmin(np.array(self.optimal_target))]   #############

        
        L,M,T,R = self.generate_final_star(correct_central_rho)
    
        print('Luminosity:', L)
        print('Mass:', M)
        print('Temperature:',T)
        print('Elapsed Time:', (time()-start_time)/60.,'minutes')
 
        output = [L,M,T,R]
        
        return output
        
    def generate_trial_star(self,central_rho):
        '''
        Generates trial star. Does not meet boundary conditions!
        ''' 

        self.index = 1
        self.htol  = 1e-8         #allowed error in RK45

        rho  = central_rho                                      #define central conditions for star we are about to run
        T    = self.central_T
        M    = 4*pi*self.r0**3.*rho/3.
        L    = 4*pi*self.r0**3.*rho*self.epsilon(rho,T)/3.

        
        RK_init = np.array([rho, #Density                                               # Inputs for odeint
                            T,   #Temperature
                            M,   #Mass
                            L,   #Luminosity
                            0])  #Tau
        
        
        print('rho_c:', central_rho)
        
        RK_output = np.array(self.odeint(RK_init))
        
        rho  = RK_output[:,0]
        T    = RK_output[:,1]
        M    = RK_output[:,2]
        L    = RK_output[:,3]
        tau  = RK_output[:,4]

        
        index = self.determine_R_s(tau)                                            # WARNING: this needs to be reviewed. 
                
        target_optimization = self.f(self.r[index], T[index], L[index])

    
        self.optimal_target.append(abs(target_optimization))
        self.dtau_list.append(self.dtau)
        self.rho_c_list.append(central_rho)

        print('-------')

        return target_optimization
        
    def generate_final_star(self,central_rho):
        '''
        Generates star with central rho & central T that match boundary conditions
        '''
        print('final_star Initializing')
        
        self.index = 1
        self.htol  = 1e-8           #allowed error in RK45
        
        rho  = central_rho
        T    = self.central_T
        M    = 4*pi*self.r0**3.*rho/3.
        L    = 4*pi*self.r0**3.*rho*self.epsilon(rho,T)/3.
            
            
        RK_init = np.array([rho, #Density                                               # Inputs for odeint
                            T,   #Temperature
                            M,   #Mass
                            L,   #Luminosity
                            0])  #Tau
        
        RK_output = np.array(self.odeint(RK_init))
        
        
        rho  = RK_output[:,0]
        T    = RK_output[:,1]
        M    = RK_output[:,2]
        L    = RK_output[:,3]
        tau  = RK_output[:,4]
        
        
        P    = self.pressure(rho,T)
            
        P_IGL = rho*k*T/(mu*m_p)                                                                            #Ideal Gas Law
        P_DEG = (3*pi**2.)**(2/3.)*hbar**2.*(rho)**(5/3.)/(5*m_e*m_p**(5/3.))                               #Degenerate
        P_RAD = 1/3*a*T**4.                                                                                 #Radiative
        
        dL     = self.dL_dr(np.array(self.r),rho,T)
        dL_pp  = self.dL_pp(np.array(self.r),rho,T)
        dL_cno = self.dL_cno(np.array(self.r),rho,T)
        
        
        kaps = self.kappa(rho,T)
        kaps_es = np.zeros(len(self.r))+kappa_es
        kaps_H  = self.kappa_H(rho,T)
        kaps_ff = self.kappa_ff(rho,T)
        
        index = self.determine_R_s(tau) 
        
        dlogT = np.diff(np.log10(T))
        dlogP = np.diff(np.log10(P))
        
        std = np.where(dlogP/dlogT - 2.5 < 0.01)
        
        lst = []
        for i in range(len(std[0]) - 1):
            
            if std[0][i + 1] - std[0][i] > 1:
                lst.append(std[0][i])
                lst.append(std[0][i+1])
        if len(lst) == 0:
            lst.append(std[0][0])
        else:
            lst.insert(0,std[0][0])
        lst.append(std[0][-1])
        
        if self.save:
            np.savetxt(self.file_name,np.concatenate((np.reshape(self.r,(len(self.r),1)),RK_output),axis=1), delimiter=',', header='radius, rho, T, M, L, tau, index = {}'.format(index))
        
        '''
        csfont = {'fontname':'Times New Roman'}
        plt.rcParams["font.family"] = "Times New Roman"
        
        
        
        fig1,ax1 = plt.subplots()
        ax1.plot(self.r/self.r[index],T/T[0], label = 'Temperature')
        ax1.plot(self.r/self.r[index],rho/rho[0], label = 'Density')
        ax1.plot(self.r/self.r[index],M/M[index], label = 'Mass')
        ax1.plot(self.r/self.r[index],L/L[index], label = 'Luminosity')
        ax1.axvspan(self.r[lst[0]]/self.r[index], self.r[lst[1]]/self.r[index], color='grey', alpha=0.25)
        
        if len(lst) > 2:
            ax1.axvspan(self.r[lst[2]]/self.r[index], self.r[lst[3]]/self.r[index], color='grey', alpha=0.25)
        
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.1)
        ax1.set_xlabel(r'Normalized Radius', size=20,**csfont)
        ax1.set_ylabel(r'Normalized T, $\rho$, M, L',size=20,**csfont)
        ax1.legend(fontsize=14,loc='center right')
        ax1.tick_params('both', labelsize=16)
        ax1.grid()
        fig1.tight_layout()
        fig1.savefig('./plots/star_fantastic4.png',dpi=200)
        
        
        
        
        
        fig2,ax2 = plt.subplots()
        ax2.plot(self.r/self.r[index],tau)
        
        ax2.tick_params('both', labelsize=16)
        ax2.set_xlim(0.0, 1.0)
        ax2.set_xlabel(r'Normalized Radius', size=20,**csfont)
        ax2.set_ylabel(r'Optical Depth', size=20,**csfont)
        
        
        
        ax2.grid()
        ax2.axvspan(self.r[lst[0]]/self.r[index], self.r[lst[1]]/self.r[index], color='grey', alpha=0.25)
        if len(lst) > 2:
            ax2.axvspan(self.r[lst[2]]/self.r[index], self.r[lst[3]]/self.r[index], color='grey', alpha=0.25)
        fig2.tight_layout()
        fig2.savefig('./plots/optical_depth.png',dpi=200)
        
        
        
        
        
        fig3,ax3 = plt.subplots()
        ax3.plot(self.r/self.r[index],P, color='black',label = 'Total Pressure')
        ax3.plot(self.r/self.r[index],P_IGL,':' ,label='Ideal Gas')
        ax3.plot(self.r/self.r[index],P_DEG,'-.' ,label='Degeneracy')
        ax3.plot(self.r/self.r[index],P_RAD,'--' ,label='Radiative')
        
        ax3.legend(fontsize=14,loc='center right')
        ax3.tick_params('both', labelsize=16)
        ax3.set_xlim(0.0, 1.0)
        ax3.set_xlabel(r'Normalized Radius', size=20,**csfont)
        ax3.set_ylabel(r'Pressure [Pa]', size=20,**csfont)
        
        
        
        ax3.grid()
        ax3.axvspan(self.r[lst[0]]/self.r[index], self.r[lst[1]]/self.r[index], color='grey', alpha=0.25)
        if len(lst) > 2:
            ax3.axvspan(self.r[lst[2]]/self.r[index], self.r[lst[3]]/self.r[index], color='grey', alpha=0.25)
        fig3.tight_layout()
        fig3.savefig('./plots/pressure.png',dpi=200)
        
        
        
        fig4,ax4 = plt.subplots()
        ax4.semilogy(self.r/self.r[index],kaps,color='black',label=r'$\kappa$')
        ax4.semilogy(self.r/self.r[index],kaps_es,':',label=r'$\kappa_{es}$')
        ax4.semilogy(self.r/self.r[index],kaps_ff,'-.',label=r'$\kappa_{ff}$')
        ax4.semilogy(self.r/self.r[index],kaps_H, '--',label=r'$\kappa_{H^-}$')
        
        
        ax4.set_ylim(10**(-2),10**(10))
        ax4.set_xlim(0.0, 1.0)
        ax4.legend(fontsize=14,loc='center right')
        ax4.tick_params('both', labelsize=16)
        ax4.set_xlabel(r'Normalized Radius', size=20,**csfont)
        ax4.set_ylabel(r'Opacity [m2 kg-1]', size=20,**csfont)
        
        
        ax4.grid()
        ax4.axvspan(self.r[lst[0]]/self.r[index], self.r[lst[1]]/self.r[index], color='grey', alpha=0.25)
        if len(lst) > 2:
            ax4.axvspan(self.r[lst[2]]/self.r[index], self.r[lst[3]]/self.r[index], color='grey', alpha=0.25)
        fig4.tight_layout()
        fig4.savefig('./plots/kappa.png',dpi=200)
        
        
        
        fig5,ax5 = plt.subplots()
        ax5.plot(self.r/self.r[index],dL,color='black',label=r'$\frac{dL}{dr}$')
        ax5.plot(self.r/self.r[index],dL_pp,':',label=r'$\frac{dL_{pp}}{dr}$')
        ax5.plot(self.r/self.r[index],dL_cno,'--',label=r'$\frac{dL_{cno}}{dr}$')
        
        ax5.legend(fontsize=14,loc='center right')
        ax5.tick_params('both', labelsize=16)
        ax5.set_xlim(0.0, 1.0)
        ax5.set_xlabel(r'Normalized Radius', size=20,**csfont)
        ax5.set_ylabel(r'dL/dr [W m-2]', size=20,**csfont)
        
        ax5.grid()
        ax5.axvspan(self.r[lst[0]]/self.r[index], self.r[lst[1]]/self.r[index], color='grey', alpha=0.25)
        if len(lst) > 2:
            ax5.axvspan(self.r[lst[2]]/self.r[index], self.r[lst[3]]/self.r[index], color='grey', alpha=0.25)
        fig5.tight_layout()
        fig5.savefig('./plots/dL_dr.png',dpi=200)
        
        
        
        
        fig6,ax6 = plt.subplots()
        ax6.plot(self.r[:index-1]/self.r[index], dlogP[:index]/dlogT[:index])
        
        ax6.tick_params('both', labelsize=16)
        ax6.set_xlim(0.0, 1.0)
        
        ax6.grid()
        ax6.axvspan(self.r[lst[0]]/self.r[index], self.r[lst[1]]/self.r[index], color='grey', alpha=0.25)        
        if len(lst) > 2:
            ax6.axvspan(self.r[lst[2]]/self.r[index], self.r[lst[3]]/self.r[index], color='grey', alpha=0.25)   
            
        ax6.set_xlabel(r'Normalized Radius', size=20,**csfont)
        ax6.set_ylabel(r'dlogP/dlogT', size=20,**csfont)
        
        fig6.tight_layout()
        fig6.savefig('./plots/dlop_dlogT.png',dpi=200)
        
        
        plt.show()
        '''

        
        return L[index],M[index],T[index],self.r[index]
        
    def odeint(self,vals):
        current_r = self.r0
        self.step = self.hmin           #meters inital step
        self.condition = True        # Always run at least once
        force = False
        
        RK_output = [vals]      # List of values to output
        self.r = [self.r0]      # Initial r

        M   = vals[2]           # Initial mass condition
        
        while self.condition and M <= M_sun*3e2 and self.r[-1] <= 20*R_sun:
            
            vals_2 = self.RK45(vals, current_r, force=False)
            
            rho = vals_2[0]
            T   = vals_2[1]
            M   = vals_2[2]
            L   = vals_2[3]

            while isnan(rho) or isnan(T) or isnan(M) or isnan(L):  # if there is a nan value in our iteration, the thing is dead... should go back and do smaller step?
                print('NaN value')
                vals_2 = self.RK45(vals, current_r,force=True)
                rho = vals_2[0]
                T   = vals_2[1]
                M   = vals_2[2]
                L   = vals_2[3]
            
            vals = vals_2
                
            self.condition = self.opacity_limit(rho,T) 
                
            if not self.condition:
                print('Opacity limited')
                
            current_r += self.step
            self.r.append(current_r)
            RK_output.append(vals)
            
        #print(current_r)
        return RK_output
        
    def opacity_limit(self,rho,T):
        '''
        order of magnitude estimate for remaining opacity # MAYBE WRONG
        '''
        drho_dr = self.drho # hack. drho_dr VALUE is made global variable so it can be used here
        kappa = self.kappa(rho,T)
        delta_tau = kappa*rho**2./abs(drho_dr)
        if delta_tau <= 1e-2: # This is the victory condition that should stop the integrator
            return False
        else:
            return True
        
        
    def determine_R_s(self,tau): 
        '''
        WARNING: sometimes initial dtau is more than 25. It should be ~0!!  # MAYBE WRONG
        May need to be rethought after implementation of RK45
        '''
        i = -1
        first = True
        self.dtau = 0
        while self.dtau <= 2/3.:
            if len(tau) == abs(i):
                i = 0
                break
            elif not isnan(tau[i]) and first:
                tau_inf = tau[i]
                first = False
            elif not first:
                self.dtau = tau_inf - tau[i]   
            i -= 1
        print('delta_tau:',self.dtau)
        return i
    
    def DE(self,init,r):
        '''
        Coupled DE of stelar equations
        '''
        rho = init[0]
        T   = init[1]
        M   = init[2]
        L   = init[3]
        
        P     = self.pressure(rho,T)
        kappa = self.kappa(rho,T)
        
        self.drho   = self.drho_dr(r,rho,T,L,M,P,kappa)  #WARNING: made it global because we need it to determine opacity condition
        dT          = self.dT_dr(r,rho,T,L,M,P,kappa)
        dM          = self.dM_dr(r,rho)
        dL          = self.dL_dr(r,rho,T)
        dtau        = self.dtau_dr(rho,T)
        
        return np.array([self.drho,dT,dM,dL,dtau])
    
    
    def RK45(self,inits,r,force):
        f = self.DE
        i = 0                       # assume that step size is not optimal 
        while i <= 1:
            k1 = self.step*f(inits,r)
            k2 = self.step*f(inits + 1/4.*k1                                                    , r+1/4.*self.step  )
            k3 = self.step*f(inits + 3/32.*k1 + 9/32.*k2                                        , r+3/8.*self.step  )
            k4 = self.step*f(inits + 1932/2197.*k1 - 7200/2197.*k2 + 7296/2197.*k3              , r+12/13.*self.step)
            k5 = self.step*f(inits + 439/216.*k1 - 8*k2 + 3680/513.*k3 - 845/4104.*k4           , r+self.step       )
            k6 = self.step*f(inits - 8/27.*k1 + 2*k2 - 3544/2565.*k3 + 1859/4104.*k4 - 11/40.*k5, r+1/2.*self.step  )
    
            y1 = inits + 25/216.*k1 + 1408/2565.*k3 + 2197/4104.*k4 - 1/5.*k5
            z1 = inits + 16/135.*k1 + 6656/12825.*k3 + 28561/56430.*k4 - 9/50.*k5 + 2/55.*k6

            if i == 0:
                s = (self.htol/(2*abs(z1[self.index]-y1[self.index])))**(1/4.)              
                
                if abs(s) == inf:
                    s = 1
                elif isnan(s):
                    s = 0.5
                
                self.step *= s              # optimal step is step*s
                
                #### Keep adaptive step in check. Not too small, not too big. force first step to be hmin
                if r == self.r0:
                    self.step = self.hmin
                elif self.step > self.hmax:
                    self.step = self.hmax
                elif self.step < self.hmin and not force:
                    self.step = self.hmin
            
            i += 1
            
        return y1

    
    def f(self,R_s,T_s,L_s):
        '''
        function that we want to find the root of so we can determine rho_c for given T
        '''
        print('Error value:', (L_s-4*pi*sigma*R_s**2.*T_s**4.)/np.sqrt(4*pi*sigma*R_s**2.*T_s**4.*L_s))#*np.ceil(self.dtau-2/3.)**5. )
        return (L_s-4*pi*sigma*R_s**2.*T_s**4.)/np.sqrt(4*pi*sigma*R_s**2.*T_s**4.*L_s)#*np.ceil(self.dtau-2/3.)**5.  
    
    
    #### Pressure
    def pressure(self,rho,T):                                                                               #Pressure
        P_IGL = rho*k*T/(mu*m_p)                                                                            #Ideal Gas Law
        P_DEG = (3*pi**2.)**(2/3.)*hbar**2.*(rho)**(5/3.)/(5*m_e*m_p**(5/3.))                               #Degenerate
        P_RAD = 1/3*a*T**4.                                                                                 #Radiative

        return P_IGL + P_DEG + P_RAD                                            


    #### Opacities
    def kappa_ff(self,rho,T):                                                                               # Free-free opacity
        return 1.0e24*(Z+0.0001)*(rho*1e-3)**0.7*T**(-3.5)

    def kappa_H(self,rho,T):                                                                                # H- opacity
        return 2.5e-32*(Z/0.02)*(rho*1e-3)**0.5*(T**9.)   
    
    def kappa(self,rho,T):# Final roselend mean opacity
        return ( 1./self.kappa_H(rho,T) + 1./np.maximum(kappa_es,self.kappa_ff(rho,T)) )**(-1) 
    
    
    #### dT_dr
    def dT_dr_rad(self,r,rho,T,L,kappa):                                                                    # radiative dT
        return 3.*self.kappa(rho,T)*rho*L / ( 16. * pi * a * c * (T**3.) * (r**2.) )         
    
    def dT_dr_conv(self,r,rho,T,M,P):                                                                       # convective dT
        return (1-1./gamma)*T*G*M*rho/(self.pressure(rho,T)*r**2.)                                              

    def dT_dr(self,r,rho,T,L,M,P,kappa):                                                                    # dT
        return -min(self.dT_dr_rad(r,rho,T,L,kappa), self.dT_dr_conv(r,rho,T,M,P))
    
    
    #### Reaction Rates 
    def epsilon_pp(self,rho,T):                                                                             # reaction rate PP chain
        return 1.07e-7*(rho*1e-5)*X**2.*(T*1e-6)**4                                       
    
    def epsilon_cno(self,rho,T):                                                                            # reaction rate CNO cycle
        return 8.24e-26*(rho*1e-5)*X**2.*0.03*(T*1e-6)**19.9                                
    
    def epsilon(self,rho,T):                                                                                # total reaction rate
        return self.epsilon_pp(rho,T) + self.epsilon_cno(rho,T)                             


    #### drho_dr
    def dP_drho(self,rho,T):                                                                                # dP_drho needed for drho_dr
        return (3*pi**2)**(2/3.)*hbar**2.*rho**(2/3.)/(3*m_e*m_p*m_p**(2/3.)) + k*T/(mu*m_p)                

    def dP_dT(self,rho,T):                                                                                  # dP_dT needed for drho_dr
        return rho*k/(mu*m_p) + 4/3.*a*(T**3.)
    
    def drho_dr(self,r,rho,T,L,M,P,kappa):                                                                  # drho_dr
        return -(G*M*rho/(r**2.) + self.dP_dT(rho,T)*self.dT_dr(r,rho,T,L,M,P,kappa))/self.dP_drho(rho,T)
    
    
    #### Luminosity
    def dL_dr(self,r,rho,T):                                                                                # dL
        return 4*pi*r**2.*rho*self.epsilon(rho,T)                                                           
    
    def dL_pp(self,r,rho,T):                                                                                # dL_pp
        return 4*pi*r**2.*rho*self.epsilon_pp(rho,T)    
    
    def dL_cno(self,r,rho,T):                                                                               # dL_cno
        return 4*pi*r**2.*rho*self.epsilon_cno(rho,T)    
    
    
    #### Mass
    def dM_dr(self,r,rho):                                                                                  # mass continuity
        return 4*pi*r**2.*rho                                                                                
    
    
    
    #### Optical depth
    def dtau_dr(self,rho,T):
        return self.kappa(rho,T)*rho    
        