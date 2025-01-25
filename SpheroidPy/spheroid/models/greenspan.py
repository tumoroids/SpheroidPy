# Parameters in our "reduced" parameterization
Q = .7
R1 = 200.0
gamma = 3
s = 0.4
R0 = 80.0

color_proliferation = "#00A600"
color_inhibited = "#B9FC00"
color_necrotic = "#EB2427"
color_treatment_effect = ""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from scipy.optimize import fsolve
#from scipy.integrate import solve

def circle(r):
    theta = np.linspace(0, 2 * np.pi, 100)
    return r * np.sin(theta), r * np.cos(theta)

class greenspan_spheroid:
    param_dic: dict # parameters needed for the reduced model (Q:balance between nutrient and inhibitor concentration < 1 [-], γ: balance between cell growth and the loss due to necrosis (lambda/s) [-], s: cell proliferation (at the per-volume rate) [1/d], R0: initial Spheroid size [µm], R1: spheroid size at which necrotic core emerges [µm])
    deriv_param_dic: dict
    sol_dic: dict # solution for ODE [...x...]

    ''' normal Greenspan Model'''
    def calc_phi_eta(self, R:float):
        """
            Calculate the inner variables Phi and Eta.

            Parameters:
            R : float
                Radius
            theta : array_like
                Parameters [Q, R1, gamma, s]

            Returns:
            tuple
                Phi and Eta.
        """
        try:
            R = R[0]
        except:
            R = R

        # Phase1:
        if R <= min(Q, 1.0) * R1:
            Phi, Eta = 0.0, 0.0

        # Phase2:
        elif R <= R1:
            Phi = np.sqrt(1 - (self.param_dic['Q'] * self.param_dic['R1']) ** 2 / R ** 2)
            Eta = 0.0

        # Phase3:
        else:
            # Berechnung von Eta
            coefs = [2 * R ** 2, -3 * R ** 2, 0.0, R ** 2 - self.param_dic['R1'] ** 2]  # [R ** 2 - R1 ** 2, 0.0, -3 * R ** 2, 2 * R ** 2]
            roots = np.roots(coefs)
            roots = [root for root in roots if np.isreal(root) and 0.0 <= root <= 1.0]
            if roots:
                Eta = np.real(roots[0])
            else:
                raise ValueError("Keine gültigen Wurzeln für eta gefunden")

            # Berechnung von Phi
            coefs = [R ** 2, 0.0, (self.param_dic['Q'] ** 2) * (self.param_dic['R1'] ** 2) - (R ** 2) * (1 + 2 * (Eta ** 3)), (
                        2 * Eta ** 3 * R ** 2)]  # [2*(Eta ** 3)*(R**2), (Q**2)*(R1**2)-(R**2)*(1+2*(Eta**3)), 0.0, (R**2)]
            roots = np.roots(coefs)
            roots = [root for root in roots if np.isreal(root) and Eta <= root <= 1.0]
            print(coefs, roots)
            if roots:
                Phi = np.real(roots[0])
            else:
                Phi = np.sqrt(1 - (self.param_dic['Q'] * self.param_dic['R1']) ** 2 / R ** 2)
                print(R,"µm --- Keine gültigen Wurzeln für phi gefunden\n\n")
                # raise ValueError("Keine gültigen Wurzeln für phi gefunden")

        #self.phi, self.eta = Phi, Eta #macht keinen sinn weil von t bzw R bhängig!!!!!!!
        return Phi, Eta
    def greenspan_ode(self, t:float, R:float):
        """
        Calculate the right-hand side of the Greenspan ODE model.

        Parameters:
        t : float
            Time
        R : float
            Radius
        theta : array_like
            Parameters [Q, R1, gamma, s]

        Returns:
        float
            Derivative dR/dt.
        """
        phi, eta = self.calc_phi_eta(R)
        dR = self.param_dic['s'] / 3 * R * (1 - phi**3 - 3 * self.param_dic['gamma'] * eta**3)
        return dR
    def solve_model_ode(self, tmax:float):
        """
        Solve the Greenspan model from t=0 to t=tmax and return all variables.

        Parameters:
        theta : array_like
            Parameters [Q, R1, gamma, s, R0]
        tmax : float
            Maximum time

        Returns:
        function
            Function to get R(t), Phi(t), Eta(t).
        """

        sol = solve_ivp(self.greenspan_ode, [0, tmax], [self.param_dic['R0']], dense_output=True)
        print('Solution =', sol)

        def all_vars(t):
            R = sol.sol(t)[0]
            print('### R = ', R)
            phi, eta = self.calc_phi_eta(R)
            #print('Phi =', phi)
            return R, R * phi, R * eta

        # print(all_vars(tmax))
        return all_vars

    def omega_fun(self, r, Ro, Ri, Rn): # oxygen and nutrients (as a function of r)
        if r > Ro:
            return self.deriv_param_dic['omega2']
        elif Rn <= r <= Ro:
            return self.deriv_param_dic['omega2'] - self.deriv_param_dic['A'] / (6 * self.deriv_param_dic['k']) * (Ro ** 2 - r ** 2) + self.deriv_param_dic['A'] * Rn**3 / (3 * self.deriv_param_dic['k']) * (1/r - 1/Ro)
        else:
            return self.deriv_param_dic['omega1']
    def beta_fun(self, r, Ro, Ri, Rn):# waste products (as a function of r)
        if Rn <= r <= Ro:
            return self.deriv_param_dic['P'] / (6 * self.deriv_param_dic['kappa']) * ( Ro ** 2 - r ** 2 - 2 * Rn ** 3 * (1 / r - 1 / Ro))
        elif r < Rn:
            return self.beta_fun(Rn, Ro, Ri, Rn)
        else:
            return 0.0

    def plot_spheroid_timeseries(self, t_max:float, r_max:float): # t_max: time until which growth is displayed; r_max: max radius the spheroid reaches during that time
        fig_all = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1,1],
                               hspace=0.2, wspace=0.8) #define grid layout

        ax_cycling_spheroid = fig_all.add_subplot(gs[0, 0:2])
        ax_nutrient_spheroid = fig_all.add_subplot(gs[0, 2:4])
        ax_waste_spheroid = fig_all.add_subplot(gs[0, 4:6])
        ax_nutrient_radial = fig_all.add_subplot(gs[1, 0:3])
        ax_waste_radial = fig_all.add_subplot(gs[1, 3:6])

        def update_plots(tp: float):
            for ax in [ax_waste_radial, ax_nutrient_radial, ax_waste_spheroid, ax_nutrient_spheroid, ax_cycling_spheroid]:
                ax.clear()

            Ro, Ri, Rn = self.solution(tp)

            # Cell Cycle Status
            ax_cycling_spheroid.clear()
            ax_cycling_spheroid.fill(*circle(Ro), c=color_proliferation, label="Cycling")
            ax_cycling_spheroid.fill(*circle(Ri), c=color_inhibited, label="Inhibited")
            ax_cycling_spheroid.fill(*circle(Rn), c=color_necrotic, label="Necrotic")
            ax_cycling_spheroid.set_title(f"{tp:.1f} d", fontsize=18)
            ax_cycling_spheroid.legend(loc=3, prop={'size': 6})

            r = np.linspace(-1.25*r_max, 1.25*r_max, 100)
            x_grid, y_grid = np.meshgrid(r, r)
            R = np.sqrt(x_grid ** 2 + y_grid ** 2)
            # # #   Nutrients   # # #
            omega_grid = np.vectorize(self.omega_fun)(R, Ro, Ri, Rn)
            ax_nutrient_spheroid.imshow(omega_grid, extent=(-1.25*r_max, 1.25*r_max, -1.25*r_max, 1.25*r_max), origin='lower', vmin=0, vmax=100, cmap='summer')
            # # #   Waste Products   # # #
            beta_grid = np.vectorize(self.beta_fun)(R, Ro, Ri, Rn)
            ax_waste_spheroid.imshow(beta_grid, extent=(-1.25*r_max, 1.25*r_max, -1.25*r_max, 1.25*r_max), origin='lower', vmin=0, vmax=1000, cmap='summer')

            ax_nutrient_spheroid.set_title('Nutrient', fontsize=18)
            ax_nutrient_spheroid.set_aspect('equal')
            ax_waste_spheroid.set_title('Waste', fontsize=18)
            ax_waste_spheroid.set_aspect('equal')

            r = np.linspace(0.01, 1.25 * r_max, 100)
            ax_nutrient_radial.axhline(self.deriv_param_dic['omega1'], linestyle='--', color='grey')
            ax_nutrient_radial.plot(r, [self.omega_fun(ri, Ro, Ri, Rn) for ri in r], c='black')
            ax_nutrient_radial.set_ylim(-1, 101)
            ax_nutrient_radial.set_ylabel("Nutrient (a.u.)", fontsize=14)
            #ax_nutrient_radial.annotate("ω_crit", xy=(300, self.deriv_param_dic['omega1']+5), xytext=(300, 35), arrowprops=dict(facecolor='black', shrink=0.05))

            ax_waste_radial.axhline(self.deriv_param_dic['beta1'], linestyle='--', color='grey')
            ax_waste_radial.plot(r, [self.beta_fun(ri, Ro, Ri, Rn) for ri in r], c='black')
            ax_waste_radial.set_ylim(-10, 1010)
            ax_waste_radial.set_ylabel("Waste (a.u.)", fontsize=14)
            #ax_waste_radial.annotate("β_crit", xy=(300, self.deriv_param_dic['beta1']+60), xytext=(300, 380), arrowprops=dict(facecolor='black', shrink=0.05))

            # Formating Plots
            for ax in [ax_waste_spheroid, ax_nutrient_spheroid, ax_cycling_spheroid]:
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                ax.set_xlim([-1.25*r_max, 1.25*r_max])
                ax.set_ylim([-1.25*r_max, 1.25*r_max])
                ax.set_aspect('equal')
            for ax in [ax_nutrient_spheroid, ax_waste_spheroid]:
                circle_ro = Circle((0, 0), Ro, color="black", fill=False)
                ax.add_patch(circle_ro)
                if Ri > 0:
                    circle_ri = Circle((0, 0), Ri, color="black", fill=False, linestyle='--')
                    ax.add_patch(circle_ri)
                if Rn > 0:
                    circle_rn = Circle((0, 0), Rn, color="black", fill=False, linestyle=':')
                    ax.add_patch(circle_rn)
            for ax in [ax_waste_radial, ax_nutrient_radial]:
                ax.set_xlim([0, 1.25*r_max])
                ax.set_xlabel('Radius [µm]', fontsize=14)
                ax.fill_between([0, Rn], -10, 5000, color=color_necrotic, alpha=0.5)
                ax.fill_between([Rn, Ri], -10, 5000, color=color_inhibited, alpha=0.5)
                ax.fill_between([Ri, Ro], -10, 5000, color=color_proliferation, alpha=0.5)


        ani = animation.FuncAnimation(fig_all, update_plots, frames=np.linspace(0.0, t_max, 200), repeat=False)
        ani.save("animation.mov", writer='ffmpeg', fps=7)

    def __init__(self, Q:float, gamma:float, s:float, R1:float, R0:float, t_interval:(float, float)=None):
        """
        Initialize the Greenspan model.

        :param Q:
        :param gamma:
        :param s:
        :param R1:
        :param R0:
        :param t_interval:
        """
        # initialize all dictionaries
        self.param_dic = {}
        self.deriv_param_dic = {}
        self.sol_dic = {}

        # Setting up all Parameters
        # for reduced model (Q: [], γ: [], s: [], R0: [], R1: [])
        self.param_dic['Q'], self.param_dic['gamma'], self.param_dic['s'], self.param_dic['R0'], self.param_dic['R1'] = Q, gamma, s, R0, R1
        self.param_dic['delta'] = 2
        try:
            t_init, t_fin = t_interval
        except:
            print('No values for time intervall specified! Using standard with $t_{init}=0h$, $t_{final}=5h$')
            t_init, t_fin = 0, 20 #specify later!!!!!!!!!!

        # derived parameters (Q: [], γ: [], s: [], R0: [], R1: [])
        _lambda = gamma * s * 3 #braucht man das????????
        self.deriv_param_dic['omega1'], self.deriv_param_dic['omega2'] = 20., 100.
        self.deriv_param_dic['A'], self.deriv_param_dic['k'] = 12., 1000.
        self.deriv_param_dic['beta1'], self.deriv_param_dic['kappa'] = 300., 500.
        self.deriv_param_dic['P'] = self.deriv_param_dic['A'] / (self.deriv_param_dic['k'] * (self.deriv_param_dic['omega2'] - self.deriv_param_dic['omega1'])) * self.deriv_param_dic['beta1'] * self.deriv_param_dic['kappa'] / (self.param_dic['Q'] ** 2)


        # Setting up all Parameters for reduced Model
        # solve greenspan ODE
        self.solution = self.solve_model_ode(t_fin)
        t_array = np.linspace(0, t_fin, 100)
        self.sol_dic['Ro_array'], self.sol_dic['Ri_array'], self.sol_dic['Rn_array'] = np.vectorize(self.solution)(t_array)
        Ro_array, Ri_array, Rn_array= np.vectorize(self.solution)(t_array)
        #print('a,', Ro_array, '\nb', Ri_array, Rn_array)
        plt.plot(t_array, Ro_array, color=color_proliferation, label='Cycling')
        plt.plot(t_array, Ri_array, color=color_inhibited, label='Inhibited')
        plt.plot(t_array, Rn_array, color=color_necrotic, label='Necrotic')
        plt.title(self.param_dic)
        plt.legend()
        plt.xlabel('time (d)')
        plt.show()

        #print(self.sol_dic)
        t_max, r_max = np.max(t_array), self.sol_dic['Ro_array'].max()
        self.plot_spheroid_timeseries(t_max, r_max)




''' Visualisation '''
# Function to plot circle
def circle(r):
    theta = np.linspace(0, 2 * np.pi, 100)
    return r * np.sin(theta), r * np.cos(theta)

def ellipsoid(r, theta=0, phi=0): #https://photonics101.com/multipole-moments-electric/quadrupole-multipole-moments-homogeneously-charged-ellipsoid.html#hints
    x_ratio, y_ratio, z_ratio = 1, 1, 1
    x = r * x_ratio * np.cos(theta) * np.sin(phi)
    y = r * y_ratio * np.cos(theta) * np.cos(phi)
    z = r * z_ratio * np.sin(theta)
    return y, z

def plot_ellipsoid(ax=None, Ro=2, Ri=1, Rn=.5):
    # Setup grid
    x = np.linspace(-350.0, 350.0, 100)
    y = np.linspace(-350.0, 350.0, 100)
    X, Y = np.meshgrid(x, y)

    if ax is None:
        fig, ax = plt.subplots()

    ax.clear()
    ax.fill(*ellipsoid(Ro, np.linspace(0, 2 * np.pi, 100), 0), c="#A6D272", label="Cycling")
    ax.fill(*ellipsoid(Ri, np.linspace(0, 2 * np.pi, 100), 0), c="#AC79B5", label="Inhibited")
    ax.fill(*ellipsoid(Rn, np.linspace(0, 2 * np.pi, 100), 0), c="#5C6AB2", label="Necrotic")
    ax.axis('equal')
    ax.legend()
    #ax.xlabel('x [µm]')
    #ax.ylabel('y [µm]')

    return ax


if __name__ == '__main__':
    '''
    solution = solve_model_all_vars([Q, R1, gamma, s, R0], 80.0)
    print(solution(2))
    t_array = np.linspace(0,80,100)
    Ro_array, Ri_array, Rn_array = np.vectorize(solution)(t_array)
    print('a,',Ro_array,'\nb', Ri_array, Rn_array)
    plt.plot(t_array,Ro_array)
    plt.plot(t_array, Ri_array)
    plt.plot(t_array, Rn_array)
    plt.show()
    '''



    '''
    for t in np.linspace(0, 10, 30):
        Ro, Ri, Rn = solution(t)
        ax=plot_ellipsoid(None,2+.1*t,1+.1*t,.5+.1*t)
        ax.set_title('t={}'.format(t))
        plt.show()
    '''

    print('\n\n\nHello :) ############################‘‘')

    spheroid_test = greenspan_spheroid(Q, gamma, s, R0, R1)