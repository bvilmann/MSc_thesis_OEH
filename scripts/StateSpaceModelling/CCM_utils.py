import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
from pscad_data_reader import PowerSourceDataLoader
from CCM_class import ComponentConnectionMethod, StateSpaceSystem

from scipy.signal import TransferFunction, freqresp
import control
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from  plot_utils import insert_axis, grouped_bar
import numpy as np
import matplotlib as mpl
import scipy.linalg as la

def plot_Zdq(tf_dq, ax=None, save=False):
    if ax is None:
        # Plot Bode plot for each input-output pair
        fig, ax = plt.subplots(4, 2, figsize=(10, 8), sharex=True)

    # Counters for indexing plot count for each axis
    dd_cnt = 0
    qd_cnt = 0
    dq_cnt = 0
    qq_cnt = 0

    for ccm_dict in tf_dq:
        v = ccm_dict
        key = ccm_dict['axis'].lower()
        print(key)

        if key == 'dd':
            ax0, ax1 = ax[0, 0], ax[1, 0]
            idx = dd_cnt
            dd_cnt += 1
        elif key == 'dq':
            ax0, ax1 = ax[0, 1], ax[1, 1]
            idx = dq_cnt
            dq_cnt += 1
        elif key == 'qd':
            ax0, ax1 = ax[2, 0], ax[3, 0]
            idx = qd_cnt
            qd_cnt += 1
        elif key == 'qq':
            ax0, ax1 = ax[2, 1], ax[3, 1]
            idx = qq_cnt
            qq_cnt += 1

        # print(v)

        # Define the system matrices A, B, and C
        A = (v['ccm']).A  # Replace with your A matrix
        B = (v['ccm']).B  # Replace with your B matrix
        C = (v['ccm']).C  # Replace with your C matrix
        D = (v['ccm']).D  # Replace with your C matrix

        # Define the frequency range for which you want to plot the Bode plot
        # omega = np.logspace(-3, 4, 1000)  # Replace with your desired frequency range
        omega = np.logspace(-2, 5, 1000)  # Replace with your desired frequency range

        # Preallocate magnitude and phase arrays
        mag = np.zeros((len(omega), C.shape[0], B.shape[1]))
        phase = np.zeros_like(mag)

        # Compute the transfer function and Bode plot for each frequency
        for k, w in enumerate(omega):
            s = 1j * w
            # Calculate the transfer function matrix
            T = C @ inv(s * np.eye(A.shape[0]) - A) @ B
            # Store magnitude and phase for each input-output pair
            mag[k, :, :] = 20 * np.log10(abs(T))
            phase[k, :, :] = np.angle(T, deg=True)

        # Adjust these loops and indexing if your system has more inputs/outputs
        for i in range(C.shape[0]):
            for j in range(B.shape[1]):
                ax0.semilogx(omega / (2 * np.pi), mag[:, i, j], label=v['label'],color=('k',f'C{idx}')[idx != 0],ls=('--','-')[idx != 0],zorder=(5,3)[idx != 0])
                ax1.semilogx(omega / (2 * np.pi), phase[:, i, j], label=v['label'],color=('k',f'C{idx}')[idx != 0],ls=('--','-')[idx != 0],zorder=(5,3)[idx != 0])

        ax0.set_title('$Y_\\mathit{HSC}^\\mathit{' + str(key) + '}=i_o^' + key[1] + '/v_o^' + key[0] + '$')
        ax0.axvline(50, lw=0.75, color='k')
        ax1.axvline(50, lw=0.75, color='k')
        ax1.grid(ls=':')
        ax0.grid(ls=':')
        ax1.grid(ls=':')
        ax1.set(ylim=(-180, 180))

        if key[1] == 'd':
            ax1.set_ylabel('Phase (degrees)')
            ax0.set_ylabel('Magnitude (dB)')
        if key[0] == 'q':
            ax1.set_xlabel('Frequency (Hz)')

    ax1.legend(loc='lower right')

    ax0.set(xlim=((omega / (2 * np.pi)).min(), (omega / (2 * np.pi)).max()))

    fig.tight_layout()
    fig.align_ylabels()
    if isinstance(save, str):
        print(f'SAVING {save}')
        plt.savefig(save)
    else:
        plt.show()

    return

def participation_matrix_to_latex(matrix, path:str, name:str, state_names:list, selected_rows:list=None, selected_columns:list=None, mode_labels:list=None, threshold=0.02, caption='',absolute:bool = False,omit_threshold:float=0):
    filename = f'{path}\\{name}.tex'

    if absolute:
        matrix = abs(matrix)

    if selected_rows is None:
        selected_rows = [i for i in range(matrix.shape[0])]
    if selected_columns is None:
        selected_columns = [i for i in range(matrix.shape[0])]

    # Filter the matrix based on selected rows and columns
    filtered_matrix = matrix[np.ix_(selected_rows, selected_columns)]
    filtered_matrix = matrix[np.ix_(selected_rows, selected_columns)]
    state_names = [state_names[i] for i in selected_rows]

    # Start LaTeX booktab
    latex_content = "\\begin{table}[H]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{"+caption+"}\n"
    latex_content += "\\label{tab:"+name+"}\n"
    latex_content += "\\begin{tabular}{c|" + "c" * filtered_matrix.shape[1] + "}\n"
    latex_content += "\\toprule\n"

    # Add rows of the participation matrix
    for i, row in enumerate(filtered_matrix):
        row_content = [state_names[i]] + \
                      [f"\\textbf{{({val:.3f})}}" if val > threshold else f"({val:.3f})" for val in row]
        latex_content += " & ".join(row_content) + " \\\\\n"

    # End LaTeX booktab with footer
    latex_content += "\\hline\n"
    if mode_labels is None:
        footer = ["$\\lambda_{" + str(i + 1) + "}$" for i in selected_columns]
    else:
        footer = ["$\\lambda_{" + mode + "}$" for mode in mode_labels]
    latex_content += " & ".join([""] + footer) + "\\\\\n"
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}"
    latex_content += "\\end{table}\n"

    # Save to file
    with open(filename, 'w') as file:
        file.write(latex_content)
    #
    # # Example usage
    # participation_matrix = np.array([[0.01, 0.03], [0.02, 0.04], [0.015, 0.025]])
    # state_names = ["State 1", "State 2", "State 3"]
    # selected_rows = [0, 1]  # Selecting first and second states
    # selected_columns = [0, 1]  # Selecting both columns
    # threshold = 0.02
    #
    # participation_matrix_to_latex(participation_matrix, state_names, selected_rows, selected_columns, threshold,
    #                               "participation_matrix_table.tex")


def eigen_properties_to_latex(A, path:str, name:str,caption='', selection:list=None,dominating_states:list = None):
    # Compute eigenvalues
    eigenvalues = la.eigvals(A)

    # Start LaTeX booktab
    latex_content = "\\begin{table}[H]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{"+caption+"}\n"
    latex_content += "\\label{tab:eig_"+name+"}\n"
    latex_content += "\\begin{tabular}{l|rrrr|l}\n"
    latex_content += "\\toprule\n"
    latex_content += "\\# & $\\boldsymbol{\\lambda^{re}}$ & $\\boldsymbol{\\lambda^{im}}$ & $\\boldsymbol{f}$ & $\\boldsymbol{\\zeta}$ & \\textbf{Dominant states} \\\\\n"
    latex_content += "\\midrule\n"

    # Iterate over eigenvalues
    cnt = 0
    for i, lambda_ in enumerate(eigenvalues):
        if isinstance(selection,list):
            if i + 1 not in selection:
                continue
        sigma = lambda_.real
        omega = lambda_.imag
        f = omega / (2 * np.pi)
        zeta = -sigma / np.sqrt(sigma**2 + omega**2)

        # Format the row
        if np.iscomplex(lambda_):
            if lambda_.imag > 0:
                index_str = "$"+f"\\lambda_{{{i+1},{i+2}}}" + "$"
                lambda_str_re = f"{lambda_.real:.3f}"
                lambda_str_im = " $\\pm$ " + f"{abs(lambda_.imag):.3f}"
            else:
                continue

        else:
            index_str = "$"+ f"\\lambda_{{{i+1}}}" + "$"
            lambda_str_re = f"{lambda_.real:.3f}"
            lambda_str_im = f"{lambda_.imag:.3f}"

        f_str = f"{f:.2f}"
        zeta_str = f"{zeta:.2f}"

        if dominating_states is not None:
            dominating_state = dominating_states[cnt]
        else:
            dominating_state = ''

        latex_content += f"{index_str} & {lambda_str_re} & {lambda_str_im} & {f_str} & {zeta_str} & {dominating_state} \\\\\n"

        cnt += 1

    # End LaTeX booktab
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}"
    latex_content += "\\end{table}\n"

    # Save to file
    with open(f'{path}\\eigtab_{name}.tex', 'w') as file:
        file.write(latex_content)
    return


def plot_nyquist(tf_dq,unit_circle:bool=False,N:int=10000,w_start:int=-3,w_end:int=5,save: str = False):
    fig = plt.figure(dpi=150, figsize=(8, 7))
    # First subplot spanning the first two rows
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    # Second subplot in the third row
    ax2 = plt.subplot2grid((5, 1), (4, 0))

    # # Third subplot in the fourth row
    # ax3 = plt.subplot2grid((6, 1), (5, 0))

    ax1.set(xlabel='Real',ylabel='Imaginary')
    ax2.set(xlabel='Frequency [Hz]')

    max_stab_marg = 0

    for i in range(len(tf_dq)):
        idx = i
        clr = f'C{i}'

        tf = control.ss2tf(tf_dq[idx]['ccm'].A, tf_dq[idx]['ccm'].B, tf_dq[idx]['ccm'].C, tf_dq[idx]['ccm'].D)

        # Transfer function coefficients
        num = np.array(tf.num).ravel()
        den = np.array(tf.den).ravel()

        # Create transfer function - see docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html
        # get the system in transfer function
        sys = TransferFunction(num, den)

        # convert to zero-pole gain representation
        zpk = sys.to_zpk()

        # Transform from laplace domain into frequency domain
        s, w, x = symbols('s w x')
        k = zpk.gain
        num_ = 1
        den_ = 1
        for z in zpk.zeros:
            num_ *= s + z
        for p in zpk.poles:
            den_ *= s + p

        num_ = num_.subs(s, I * w)
        den_ = den_.subs(s, I * w)

        num__ = num_ * conjugate(den_)
        den__ = den_ * conjugate(den_)

        G = k * num__ / den__

        # Lambdify - docs: https://docs.sympy.org/latest/tutorials/intro-tutorial/basic_operations.html#lambdify
        expr = G.evalf(subs={w: x})
        Geval = lambdify(x, expr, "numpy")

        # pprint(num_)
        # pprint(den_)
        # print(G)

        # %%
        G_vals = []
        ws = np.logspace(w_start, w_end, N)
        for w_ in ws[::-1]:
            # for w_ in np.linspace(0, 12, 10000):
            G_vals.append(Geval(-w_))

        for w_ in ws:
            # for w_ in np.linspace(0, 12, 10000):
            G_vals.append(Geval(w_))

        if i == 0:
            ax1.scatter([-1], [0], marker='o', color='red')

        ax1.plot([v.real for v in G_vals], [v.imag for v in G_vals], color=clr, alpha=0.5)
        ax1.scatter([G_vals[0].real], [G_vals[0].imag], marker='+', s=75, color=clr, zorder=6, label='$\\omega=\\infty_-$')
        ax1.scatter([G_vals[-1].real], [G_vals[-1].imag], marker='x', s=50, color=clr, zorder=6,
                    label='$\\omega=\\infty_+$')
        ax1.scatter([G_vals[N - 1].real], [G_vals[N - 1].imag], marker='3', s=75, color=clr, zorder=6,
                    label='$\\omega=0_{-}$')
        ax1.scatter([G_vals[N].real], [G_vals[N].imag], marker='4', s=75, color=clr, zorder=6, label='$\\omega=0_{+}$')

        ax1.scatter([Geval(50*2*np.pi).real], [Geval(50*2*np.pi).imag], marker='*', s=75, color=clr, zorder=6)
        # ax1.scatter([Geval(-50*2*np.pi).real], [Geval(-50*2*np.pi).imag], marker='*', s=75, color=clr, zorder=6)

        # ax.set(xlim=(-.11e11,.02e11),ylim=(-0.2e11,0.2e11))
        ax1.legend()
        ax1.grid(ls=':')

        pos = np.array(G_vals[N:]).flatten()
        stab_marg = abs(pos + 1)
        phas_marg = np.degrees(np.angle(pos + 1))

        ax2.plot(ws / (2 * np.pi), stab_marg)
        # ax3.plot(ws / (2 * np.pi), phas_marg)

        max_stab_marg = max(max(stab_marg), max_stab_marg)
        ax2.set_ylim(0, max_stab_marg * 1.05)
        # ax3.set_ylim(-185, 185)

        for j, ax in enumerate([ax2]):
            ax.set_xscale('log')
            ax.grid(ls=':')
            ax.axvline(50, color='k', lw=0.75)
            ax.set_xlim(min(ws) / (2 * np.pi), max(ws) / (2 * np.pi))
            ax.set_ylabel(['$\\left|G(j\\omega) + 1\\right|$', '$\\mathrm{ang}\\left(G(j\\omega) + 1\\right)$'][j])

    # Define custom legend markers and labels
    legend_elements = [
        Line2D([0], [0], color='C0', label=r'$Y^{dd}(j\omega)$', markersize=10),
        Line2D([0], [0], color='C1', label=r'$Y^{qd}(j\omega)$', markersize=10),
        Line2D([0], [0], color='C2', label=r'$Y^{dq}(j\omega)$', markersize=10),
        Line2D([0], [0], color='C3', label=r'$Y^{qq}(j\omega)$', markersize=10),
        Line2D([0], [0], marker='+', color='w', markeredgecolor='k', label=r'$\omega = \infty_-$', markerfacecolor='w',markersize=10),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='k', label=r'$\omega = \infty_+$', markerfacecolor='w',markersize=10),
        Line2D([0], [0], marker='3', color='w', markeredgecolor='k', label=r'$\omega = 0_-$', markerfacecolor='w',markersize=10),
        Line2D([0], [0], marker='4', color='w', markeredgecolor='k', label=r'$\omega = 0_+$', markerfacecolor='w',markersize=10),
        Line2D([0], [0], marker='*', color='w', label=r'$\omega=50\cdot 2\pi $', markerfacecolor='k', markersize=12.5),
        Line2D([0], [0], marker='o', color='w', label='Critical point', markerfacecolor='red', markersize=7.5),

    ]

    if unit_circle:
        ax1.set_aspect('equal')
        ax1.plot(np.cos(np.linspace(0,2*np.pi,1000)), np.sin(np.linspace(0,2*np.pi,1000)), linewidth=0.75,color='k',ls=':')

    # Add the custom legend to the plot
    ax1.legend(handles=legend_elements, ncol=1,loc='upper left',bbox_to_anchor=(1.0, 1.0),fontsize=8)
    # ax2.set_yscale('log')
    # ax2.axes.xaxis.set_ticklabels([])

    fig.tight_layout()
    fig.align_ylabels()

    if isinstance(save, str):
        print(f'SAVING {save}')
        plt.savefig(save)
    else:
        plt.show()

    return

def plot_nyquist_sweep(tf_dq,unit_circle:bool=False,N:int=10000,w_start:int=-3,w_end:int=5,save: str = False):

    mpl.colormaps['Reds']
    fig = plt.figure(dpi=150, figsize=(6, 8))
    # First subplot spanning the first two rows
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    # Second subplot in the third row
    ax2 = plt.subplot2grid((5, 1), (4, 0))

    # # Third subplot in the fourth row
    # ax3 = plt.subplot2grid((6, 1), (5, 0))

    ax1.set(xlabel='Real',ylabel='Imaginary')
    ax2.set(xlabel='Frequency [Hz]')

    max_stab_marg = 0

    for i in range(len(tf_dq)):
        idx = i
        clr = mpl.colormaps['Reds']((i+1)/len(tf_dq))


        tf = control.ss2tf(tf_dq[idx]['ccm'].A, tf_dq[idx]['ccm'].B, tf_dq[idx]['ccm'].C, tf_dq[idx]['ccm'].D)

        # Transfer function coefficients
        num = np.array(tf.num).ravel()
        den = np.array(tf.den).ravel()

        # Create transfer function - see docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html
        # get the system in transfer function
        sys = TransferFunction(num, den)

        # convert to zero-pole gain representation
        zpk = sys.to_zpk()

        # Transform from laplace domain into frequency domain
        s, w, x = symbols('s w x')
        k = zpk.gain
        num_ = 1
        den_ = 1
        for z in zpk.zeros:
            num_ *= s + z
        for p in zpk.poles:
            den_ *= s + p

        num_ = num_.subs(s, I * w)
        den_ = den_.subs(s, I * w)

        num__ = num_ * conjugate(den_)
        den__ = den_ * conjugate(den_)

        G = k * num__ / den__

        # Lambdify - docs: https://docs.sympy.org/latest/tutorials/intro-tutorial/basic_operations.html#lambdify
        expr = G.evalf(subs={w: x})
        Geval = lambdify(x, expr, "numpy")

        # pprint(num_)
        # pprint(den_)
        # print(G)

        # %%
        G_vals = []
        ws = np.logspace(w_start, w_end, N)
        for w_ in ws[::-1]:
            # for w_ in np.linspace(0, 12, 10000):
            G_vals.append(Geval(-w_))

        for w_ in ws:
            # for w_ in np.linspace(0, 12, 10000):
            G_vals.append(Geval(w_))

        if i == 0:
            ax1.scatter([-1], [0], marker='o', color='red')

        ax1.plot([v.real for v in G_vals], [v.imag for v in G_vals], color=clr, alpha=0.75,label=tf_dq[idx]['label'])
        ax1.scatter([G_vals[0].real], [G_vals[0].imag], marker='+', s=75, color=clr, zorder=6) # , label='$\\omega=\\infty_-$'
        ax1.scatter([G_vals[-1].real], [G_vals[-1].imag], marker='x', s=50, color=clr, zorder=6) # ,label='$\\omega=\\infty_+$'
        ax1.scatter([G_vals[N - 1].real], [G_vals[N - 1].imag], marker='3', s=75, color=clr, zorder=6) # ,label='$\\omega=0_{-}$'
        ax1.scatter([G_vals[N].real], [G_vals[N].imag], marker='4', s=75, color=clr, zorder=6) # , label='$\\omega=0_{+}$'

        ax1.scatter([Geval(50*2*np.pi).real], [Geval(50*2*np.pi).imag], marker='*', s=75, color=clr, zorder=6)
        # ax1.scatter([Geval(-50*2*np.pi).real], [Geval(-50*2*np.pi).imag], marker='*', s=75, color=clr, zorder=6)

        # ax.set(xlim=(-.11e11,.02e11),ylim=(-0.2e11,0.2e11))
        ax1.legend()
        ax1.grid(ls=':')
        ax1.set(ylim=(-10,10),xlim=(-2,15))

        pos = np.array(G_vals[N:]).flatten()
        stab_marg = abs(pos + 1)
        phas_marg = np.degrees(np.angle(pos + 1))

        ax2.plot(ws / (2 * np.pi), stab_marg,color=clr)
        # ax3.plot(ws / (2 * np.pi), phas_marg)

        max_stab_marg = max(max(stab_marg), max_stab_marg)
        ax2.set_ylim(0, 40)
        # ax3.set_ylim(-185, 185)



        for j, ax in enumerate([ax2]):
            ax.set_xscale('log')
            ax.grid(ls=':')
            ax.axvline(50, color='k', lw=0.75)
            ax.set_xlim(min(ws) / (2 * np.pi), max(ws) / (2 * np.pi))
            ax.set_ylabel(['$\\left|G(j\\omega) + 1\\right|$', '$\\mathrm{ang}\\left(G(j\\omega) + 1\\right)$'][j])

    # Define custom legend markers and labels
    legend_elements = [
        Line2D([0], [0], color='C0', label=r'$Y^{dd}(j\omega)$', markersize=10),
        Line2D([0], [0], color='C1', label=r'$Y^{qd}(j\omega)$', markersize=10),
        Line2D([0], [0], color='C2', label=r'$Y^{dq}(j\omega)$', markersize=10),
        Line2D([0], [0], color='C3', label=r'$Y^{qq}(j\omega)$', markersize=10),
        Line2D([0], [0], marker='+', color='w', markeredgecolor='k', label=r'$\omega = \infty_-$', markerfacecolor='w',markersize=10),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='k', label=r'$\omega = \infty_+$', markerfacecolor='w',markersize=10),
        Line2D([0], [0], marker='3', color='w', markeredgecolor='k', label=r'$\omega = 0_-$', markerfacecolor='w',markersize=10),
        Line2D([0], [0], marker='4', color='w', markeredgecolor='k', label=r'$\omega = 0_+$', markerfacecolor='w',markersize=10),
        Line2D([0], [0], marker='*', color='w', label=r'$\omega=50\cdot 2\pi $', markerfacecolor='k', markersize=12.5),
        Line2D([0], [0], marker='o', color='w', label='Critical point', markerfacecolor='red', markersize=7.5),

    ]

    if unit_circle:
        ax1.set_aspect('equal')
        ax1.plot(np.cos(np.linspace(0,2*np.pi,1000)), np.sin(np.linspace(0,2*np.pi,1000)), linewidth=0.75,color='k',ls=':')

    # Add the custom legend to the plot
    ax1.legend(ncol=1,loc='upper right')
    # ax2.set_yscale('log')
    # ax2.axes.xaxis.set_ticklabels([])

    fig.tight_layout()
    fig.align_ylabels()

    if isinstance(save, str):
        print(f'SAVING {save}')
        plt.savefig(save)
    else:
        plt.show()

    return


def plot_eigs(A, leg_loc=None, mode='complex', xlim=None, ylim=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)
    marks = ['o', '*', '+', 's', 'd']
    lamb = np.linalg.eigvals(A)
    for i, l in enumerate(lamb):
        if np.real(l) > 0:
            print(i, l)
            ax.scatter([np.real(l)], [np.imag(l)], marker='3', color='red', s=200, alpha=1, zorder=4)

        if mode == 'complex':
            if np.imag(l) > 0:
                ax.scatter([np.real(l), np.real(l)], [np.imag(l), -np.imag(l)], label='$\\lambda_{' + str(i + 1) + '}$',
                           marker=marks[i // 12])
        else:
            if np.imag(l) > 0:
                ax.scatter([np.real(l), np.real(l)], [np.imag(l), -np.imag(l)], label='$\\lambda_{' + str(i + 1) + '}$',
                           marker=marks[i // 12])
            elif np.imag(l) < 0:
                continue
            else:
                ax.scatter([np.real(l)], [np.imag(l)], label='$\\lambda_{' + str(i + 1) + '}$', marker=marks[i // 12])

    ax.axhline(0, color='k', lw=0.75)
    ax.axvline(0, color='k', lw=0.75)
    # ax.set_title(self.filename)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if leg_loc is not None:
        ax.legend(loc=leg_loc, ncol=(1, 3)[len(lamb) > 24])
    else:
        ax.legend(ncol=(1, 3)[len(lamb) > 24])

    ax.grid()

    # plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_eig.pdf')
    plt.show()
    plt.close()

    return


def plot_eigvals_dict(parameter_name,eigenvalues_dict, save: str = None, y_offset=0,title=False,xlim:tuple=None,ylim:tuple=None,insert_axis_args1=None,insert_axis_kwargs1=None,insert_axis_args2=None,insert_axis_kwargs2=None,show_labels:list=[],force_labels:bool=False,**kwargs):
    def is_within_frame(z, x1=-100, x2=10, y1=-100, y2=100):
        a = complex(z).real
        b = complex(z).imag
        return x1 <= a <= x2 and y1 <= b <= y2

    # Prepare the colormap
    scr_values = list(eigenvalues_dict.keys())
    norm = plt.Normalize(min(scr_values), max(scr_values))
    colors = cm.cool(norm(scr_values))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    if insert_axis_args1 is not None:
        if insert_axis_kwargs1 is not None:
            ins_ax1,_ = insert_axis(ax, *insert_axis_args1,insert_axis_kwargs1)
        else:
            ins_ax1,_ = insert_axis(ax, *insert_axis_args1)
    if insert_axis_args2 is not None:
        if insert_axis_kwargs2 is not None:
            ins_ax2,_ = insert_axis(ax, *insert_axis_args2,insert_axis_kwargs2)
        else:
            ins_ax2,_ = insert_axis(ax, *insert_axis_args2)

    for idx, (scr, color) in enumerate(zip(scr_values, colors)):
        eigenvalues = eigenvalues_dict[scr]
        real_parts = [e.real for e in eigenvalues]
        imag_parts = [e.imag for e in eigenvalues]
        ax.scatter(real_parts, imag_parts, color=color)
        if insert_axis_args1 is not None:
            ins_ax1.scatter(real_parts, imag_parts, color=color)

        if insert_axis_args2 is not None:
            ins_ax2.scatter(real_parts, imag_parts, color=color)

        # Annotate the second eigenvalue (index 1) if its imaginary part is >= 0
        if ylim is None and xlim is None:
            if scr == list(eigenvalues_dict.keys())[0]:
                for i, eigval in enumerate(eigenvalues):
                    # if eigval.imag = 0:
                    # Adjust the offset as needed
                    if (i+1) in show_labels and ((xlim is not None and ylim is not None) or force_labels):
                        if not is_within_frame(eigval,*insert_axis_args1[0]) and not is_within_frame(eigval,*insert_axis_args2[0]):
                            ax.text(eigval.real, eigval.imag,
                                     '$\\lambda_{' + str(i + 1) + '}$', va='center', ha='center', color='k')
                        if is_within_frame(eigval,*insert_axis_args1[0]):
                            ins_ax1.text(eigval.real, eigval.imag,'$\\lambda_{' + str(i + 1) + '}$', va='center', ha='center', color='k')
                        if is_within_frame(eigval,*insert_axis_args2[0]) and not is_within_frame(eigval,*insert_axis_args1[0]):
                            ins_ax2.text(eigval.real, eigval.imag,'$\\lambda_{' + str(i + 1) + '}$', va='center', ha='center', color='k')
        else:
            if (scr == list(eigenvalues_dict.keys())[0] and xlim is not None and ylim is not None) or force_labels:
                for i, eigval in enumerate(eigenvalues):
                    if (i+1) in show_labels:
                        if is_within_frame(eigval, xlim[0],xlim[1],ylim[0],ylim[1]):
                            ax.text(eigval.real, eigval.imag,
                                    '$\\lambda_{' + str(i + 1) + '}$', va='center', ha='center', color='k')

    if 'additional' in kwargs:
        for k, v in kwargs['additional'].items():
            ins_ax1.scatter(v.real, v.imag, color='k',alpha=0.5)

    if 'hline' in kwargs:
        ins_ax1.fill_between(insert_axis_args1[0][:2],kwargs['hline']['vmin'],kwargs['hline']['vmax'],alpha=0.25,color='grey')
        ins_ax1.fill_between(insert_axis_args1[0][:2],-kwargs['hline']['vmin'],-kwargs['hline']['vmax'],alpha=0.25,color='grey')
        ins_ax1.axhline(kwargs['hline']['vmin'],color=kwargs['hline']['color'],lw=0.75,ls=':')
        ins_ax1.axhline(-kwargs['hline']['vmin'],color=kwargs['hline']['color'],lw=0.75,ls=':')
        ins_ax1.axhline(kwargs['hline']['vmax'],color=kwargs['hline']['color'],lw=0.75,ls=':')
        ins_ax1.axhline(-kwargs['hline']['vmax'],color=kwargs['hline']['color'],lw=0.75,ls=':')

    # Labels and title
    ax.set_xlabel('$\\sigma$')
    ax.set_ylabel('$j\\omega$')
    if title:
        plt.title(f'Eigenvalues in the Complex Plane by {parameter_name}')
    ax.axhline(y=0, color='k')  # x-axis
    ax.axvline(x=0, color='k')  # y-axis
    ax.grid(ls=':')
    if insert_axis_args1 is not None:
        ins_ax1.axhline(y=0, color='k')  # x-axis
        ins_ax1.axvline(x=0, color='k')  # y-axis
        ins_ax1.grid(ls=':')
    if insert_axis_args2 is not None:
        ins_ax2.axhline(y=0, color='k')  # x-axis
        ins_ax2.axvline(x=0, color='k')  # y-axis
        ins_ax2.grid(ls=':')

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)

    # if ylim is None and xlim is None:
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label=parameter_name)

    fig.tight_layout()

    # plt.legend()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
    return



def plot_participation_factors(P, names, tol=0.01,save: str = None):
    # find the absolute values
    p = abs(P)
    p_heatmap = np.where(p >= tol, p, np.nan)

    # Prepare modes and states
    ticks = [i for i in range(p.shape[0])]
    yticklabels = names
    xticklabels = ['$\\lambda_{' + f'{i + 1}' + '}$' for i in range(p.shape[0])]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), dpi=150)
    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.viridis
    cax = ax.imshow(p_heatmap, interpolation='nearest', cmap=cmap, norm=norm)

    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="3%", pad=0.1)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(xticklabels,fontsize=8)
    ax.set_yticklabels(yticklabels,fontsize=8)

    ax.set_aspect('equal')

    ax.set_xticks(np.arange(-.5, len(ticks), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ticks), 1), minor=True)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.colorbar(cax, cax=cbar_ax, extend='max')

    plt.tight_layout()

    if isinstance(save,str):
        plt.savefig(save,bbox_inches='tight')
    else:
        plt.show()

    return

def plot_participation_factor_bars(P,names,save:str=None,figsize=(6,11)):
    rows = len(names)//2
    fig,axs = plt.subplots(rows,2,dpi=150,sharex=True,sharey=True,figsize=figsize)

    for i in range(len(names)):
        ax = axs[i%rows,i//rows]
        ax.bar([i for i in range(len(names))],abs(P[:,i]),zorder=5)
        ax.set_xticks([i for i in range(len(names))])
        ax.set_xticklabels(names,rotation=90,fontsize=6)
        ax.grid(ls=':')
        ax.text(0.07,0.8,'$\\lambda_{'+f'{i+1}' + '}$', transform=ax.transAxes, va='center',ha='center',zorder=10)
    plt.tight_layout()
    if isinstance(save, str):
        plt.savefig(save,bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    return

def plot_participation_factor_bars_sample(P,names,selected_modes,save:str=None,figsize=(6,8),mode_labels:list= None,vmax:float= None):
    # names = [names[i] for i in selected_modes]
    rows = len(selected_modes)
    fig,axs = plt.subplots(rows,1,dpi=150,sharex=True,sharey=True,figsize=figsize)
    if mode_labels is None:
        mode_labels = selected_modes
    if vmax is None:
        vmax = P.max()

    for i in range(rows):
        if rows > 1:
            ax = axs[i%rows]
        else:
            ax = axs
        ax.bar([i for i in range(len(names))],abs(P[:,selected_modes[i]]),zorder=5)
        ax.set_xticks([i for i in range(len(names))])
        ax.set_xticklabels(names,rotation=90) # ,fontsize=6
        ax.grid(ls=':')
        ax.set_ylabel('$\\lambda_{'+mode_labels[i]+ '}$')
        # ax.text(0.07,0.8,'$\\lambda_{'+f'{i+1}' + '}$', transform=ax.transAxes, va='center',ha='center',zorder=10)
    ax.set_ylim(0, vmax * 1.05)
    plt.tight_layout()
    if isinstance(save, str):
        plt.savefig(save,bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    return

def plot_participation_factor_bars_sample2(P:pd.DataFrame,selected_modes,save:str=None,figsize=(6,8),mode_labels:list= None,vmax:float= None):
    # names = [names[i] for i in selected_modes]
    rows = len(selected_modes)

    names = list(P.index)

    fig,axs = plt.subplots(rows,1,dpi=150,sharex=True,sharey=True,figsize=figsize)

    if mode_labels is None:
        mode_labels = selected_modes
    if vmax is None:
        vmax = P.max()

    for i in range(rows):
        ax = axs[i%rows]
        grouped_bar(ax,P)

        ax.bar([i for i in range(len(names))],abs(P[:,selected_modes[i]]),zorder=5)
        ax.set_xticks([i for i in range(len(names))])
        ax.set_xticklabels(names,rotation=90) # ,fontsize=6
        ax.grid(ls=':')
        ax.set_ylabel('$\\lambda_{'+mode_labels[i]+ '}$')
        # ax.text(0.07,0.8,'$\\lambda_{'+f'{i+1}' + '}$', transform=ax.transAxes, va='center',ha='center',zorder=10)

    ax.set_ylim(0, vmax * 1.05)

    plt.tight_layout()
    if isinstance(save, str):
        plt.savefig(save,bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    return



def complex_to_real_expanded(matrix):
    rows, cols = matrix.shape
    real_matrix = np.zeros((2 * rows, 2 * cols))

    for i in range(rows):
        for j in range(cols):
            real_matrix[2 * i, 2 * j] = matrix[i, j].real
            real_matrix[2 * i, 2 * j + 1] = -matrix[i, j].imag
            real_matrix[2 * i + 1, 2 * j] = matrix[i, j].imag
            real_matrix[2 * i + 1, 2 * j + 1] = matrix[i, j].real

    return real_matrix

def show_matrix(M,ceil = 1e6,title:str= None):
    # print(pd.DataFrame(M))

    M = np.where(M != 0, M, np.nan)
    plt.imshow(np.where(M<ceil,1,np.nan))
    plt.imshow(np.where(M>=ceil,1,np.nan),cmap='bwr_r',alpha=0.5)

    if title is not None:
        plt.title(title)

    plt.show()
    plt.close()
    return

def calc_RXL(scr, xr, Vbase=300e3, Sbase=1e9, Fbase=50):
    Zbase = Vbase ** 2 / Sbase
    R = np.cos(np.arctan(xr)) / scr / Zbase
    X = np.sin(np.arctan(xr)) / scr / Zbase
    L = np.sin(np.arctan(xr)) / scr / Zbase / (2 * np.pi * Fbase)

    return R, X, L


def state_space_simulation(A, B, C, D, x0, u, t):
    """
    Simulate a state-space system.

    :param A, B, C, D: State-space matrices.
    :param x0: Initial state vector.
    :param u: Input function (time -> input vector).
    :param t: Time points for the simulation.
    :return: Time points, state history, output history.
    """
    print('\n#=================== RUNNING DYNAMIC SIMULATION ===================#\n')

    def system_dynamics(t, x):
        return A @ x + B @ u(t)

    sol = solve_ivp(system_dynamics, [t[0], t[-1]], x0, t_eval=t, method='RK45')

    x = sol.y
    y = np.array([C @ x[:, i] + D @ u(t[i]) for i in range(len(t))]).T

    return sol.t, x, y


def x_plot(t, x_, t_sim=None, x_sim=None):
    fig, ax = plt.subplots(5, 1, dpi=150, sharex=True)
    # ax0 = ax[0].twinx()
    ax1 = ax[1].twinx()
    ax2 = ax[2].twinx()
    ax3 = ax[3].twinx()
    ax4 = ax[4].twinx()

    ax[0].plot(t, x_[:, 0] % (2 * np.pi), color='C0')
    ax[1].plot(t, x_[:, 1], color='C0')
    ax1.plot(t, x_[:, 2], color='C1')
    ax[2].plot(t, x_[:, 3], color='C0')
    ax2.plot(t, x_[:, 4], color='C1')
    ax[3].plot(t, x_[:, 5], color='C0')
    ax3.plot(t, x_[:, 6], color='C1')
    ax[4].plot(t, x_[:, 7], color='C0')
    ax4.plot(t, x_[:, 8], color='C1')

    if x_sim is not None:
        ax[0].plot(t_sim, x_sim[:, 0] % (2 * np.pi), color='C1')
        ax[1].plot(t_sim, x_sim[:, 1], color='C0', ls=':')
        ax1.plot(t_sim, x_sim[:, 2], color='C1', ls=':')
        ax[2].plot(t_sim, x_sim[:, 3], color='C0', ls=':')
        ax2.plot(t_sim, x_sim[:, 4], color='C1', ls=':')
        ax[3].plot(t_sim, x_sim[:, 5], color='C0', ls=':')
        ax3.plot(t_sim, x_sim[:, 6], color='C1', ls=':')
        ax[4].plot(t_sim, x_sim[:, 7], color='C0', ls=':')
        ax4.plot(t_sim, x_sim[:, 8], color='C1', ls=':')

    for i, ax_ in enumerate([ax1, ax1, ax2, ax3, ax4]):
        ax[i].set(ylabel=['$\\delta$', '$P$', '$\\zeta_V^{d}$', '$\\zeta_C^{d}$', '$i_o^{d}$'][i])
        ax[i].grid(ls=':')
        ax[i].tick_params(axis='y', colors='C0')
        ax_.tick_params(axis='y', colors='C1')
        if i >= 1:
            ax_.set(ylabel=['$\\delta$', '$Q$', '$\\zeta_V^{q}$', '$\\zeta_C^{q}$', '$i_o^{q}$'][i])

    ax[0].set(xlim=(t.min(), t.max()))

    fig.tight_layout()
    fig.align_ylabels()

    return fig, ax

Ts = lambda d: np.array([[np.cos(d),-np.sin(d)],[np.sin(d),np.cos(d)]])


def load_ccm(vd0=1, vq0=0, id0=0, iq0=0, d0=0, scr=10, xr=10,x_vi=0, system_input=['v_gd', 'v_gq'], range_=range(7),
             system_output=['i_od', 'i_oq'], ccm_name='ssm_ccm', Vbase=300e3, Sbase = 1e9, path = r'C:\Users\bvilm\PycharmProjects\StateSpaceModelling\statespacemodels', **kwargs):

    if ccm_name == 'ssm_ccm - full - simple':
        range_=range(4)

    file_path = f'{path}\\{ccm_name}.xlsx'

    vD0,vQ0 = Ts(d0) @ np.array([vd0,vq0]).T

    if scr == 0:
        L_tfr = 0
        R_tfr = 0
    else:
        R_tfr, X_tfr, L_tfr = calc_RXL(scr, xr, Vbase=Vbase, Sbase=Sbase, Fbase=50)

    HSC_init = {'V_od': vd0, 'V_oq': vq0,'V_oD':vD0,'V_oQ':vQ0, 'I_od': id0, 'I_oq': iq0, 'R_tfr': R_tfr,
                'L_tfr': L_tfr, 'x_vi': x_vi,'d0':d0}

    HSC_init.update(kwargs)

    HSC = [StateSpaceSystem(file_path, 'HSC', i, vals=HSC_init, ref=True, comp_id=1) for i in range_]
    # HSC2 = [StateSpaceSystem(file_path,'HSC',i,vals=HSC2_init,ref=True,comp_id=2) for i in range(1,5)]

    # ccm = ComponentConnectionMethod(HSC1,system_input=['v_od','v_oq'],system_output=['omega','i_od','i_oq'])
    ccm = ComponentConnectionMethod(HSC, system_input=system_input, system_output=system_output)

    return ccm, HSC_init


def load_init_conditions(path=r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\validation', file=r'HSC_validation_1', t0=1.5, t_offset=-.1, t_dur=0.6):

    loader = PowerSourceDataLoader(file, directory=path)
    t_, vdq = loader.get_measurement_data(f'Main(0):vdq_HSC1', t0=t0 + t_offset, t1=t0 + t_offset + t_dur)
    t_, idq = loader.get_measurement_data(f'Main(0):idq_HSC1', t0=t0 + t_offset, t1=t0 + t_offset + t_dur)
    # t_, delta = loader.get_measurement_data(f'Main(0):x_',t0=t0+t_offset,t1=t0+t_offset+t_dur)
    # t_, w  = loader.get_measurement_data(f'Main(0):idq_HSC1',t0=t0+t_offset,t1=t0+t_offset+t_dur)

    vd0, vq0 = vdq[0, :]
    id0, iq0 = idq[0, :]
    # d0 = delta[0]
    # w0 = w[0]

    path = r'C:\Users\bvilm\Dropbox\DTU\11_master thesis\pscad\validation'
    file = r'HSC_validation_1'
    t0, x0 = loader.get_measurement_data(f'Main(0):x_HSC1', t0=t0 + t_offset, t1=t0 + t_offset + t_dur)

    return vd0, vq0, id0, iq0, x0[0, 0], (t0, x0), vdq, idq


def calculate_passivity_index(frequency_response):
    """
    Calculate the passivity index for each frequency point.

    :param frequency_response: A NxNxM matrix representing the system's frequency response.
    :return: A list of passivity indices for each frequency.
    """
    num_frequencies = frequency_response.shape[2]
    passivity_indices = []

    for i in range(num_frequencies):
        # Extract the matrix for this frequency
        H = frequency_response[:, :, i]

        # Calculate the Hermitian part
        H_hermitian = 0.5 * (H + H.conj().T)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(H_hermitian)

        # Passivity index for this frequency is the minimum eigenvalue
        passivity_index = np.min(eigenvalues)
        passivity_indices.append(passivity_index)

    return passivity_indices

def get_passivity(tf_dq, ax=None, save=False):
    if ax is None:
        # Plot Bode plot for each input-output pair
        fig, ax = plt.subplots(1,1, figsize=(6*1.2,4*1.2))

    # Define number of evaluation points in frequency and start and end frequency
    omega = np.logspace(-2, 5, 1000)  # Replace with your desired frequency range

    # Preallocate magnitude and phase arrays
    G = np.zeros((2,2,len(omega)),dtype=np.complex128)
    p = np.zeros((1,1,len(omega)))

    # Counters for indexing plot count for each axis
    for ccm_dict in tf_dq:
        v = ccm_dict
        key = ccm_dict['axis'].lower()
        print(key)


        # Define the system matrices A, B, and C
        A = (v['ccm']).A  # Replace with your A matrix
        B = (v['ccm']).B  # Replace with your B matrix
        C = (v['ccm']).C  # Replace with your C matrix
        D = (v['ccm']).D  # Replace with your C matrix

        # Define the frequency range for which you want to plot the Bode plot
        # omega = np.logspace(-3, 4, 1000)  # Replace with your desired frequency range
        omega = np.logspace(-2, 5, 1000)  # Replace with your desired frequency range

        # Preallocate magnitude and phase arrays
        mag = np.zeros((len(omega), C.shape[0], B.shape[1]))
        phase = np.zeros_like(mag)

        # Compute the transfer function and Bode plot for each frequency
        for k, w in enumerate(omega):
            s = 1j * w
            # Calculate the transfer function matrix
            T = C @ inv(s * np.eye(A.shape[0]) - A) @ B
            # Store magnitude and phase for each input-output pair

            if key == 'dd':
                G[0,0,k] = T[0][0]
            elif key == 'dq':
                G[0,1,k] = T[0][0]
            elif key == 'qd':
                G[1,0,k] = T[0][0]
            elif key == 'qq':
                G[1,1,k] = T[0][0]

    p = calculate_passivity_index(G)

    # breakpoint()
    #
    #
    # clr = mpl.colormaps['Reds']((i + 1) / len(tf_dq))
    #
    # # Adjust these loops and indexing if your system has more inputs/outputs
    # for i in range(C.shape[0]):
    #     for j in range(B.shape[1]):
    #         ax.semilogx(omega / (2 * np.pi), mag[:, i, j], label=v['label'],color=('k',f'C{idx}')[idx != 0],ls=('--','-')[idx != 0],zorder=(5,3)[idx != 0])
    #         ax.semilogx(omega / (2 * np.pi), phase[:, i, j], label=v['label'],color=('k',f'C{idx}')[idx != 0],ls=('--','-')[idx != 0],zorder=(5,3)[idx != 0])
    #
    # fig.tight_layout()
    # fig.align_ylabels()
    # if isinstance(save, str):
    #     print(f'SAVING {save}')
    #     plt.savefig(save)
    # else:
    #     plt.show()

    return p


