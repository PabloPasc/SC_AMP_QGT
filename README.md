# SC_AMP_QGT
Includes code used to produce all figures in "Quantitative Group Testing and Pooled Data in the Linear Regime with Sublinear Tests" by Nelvin Tan, Pablo Pascual Cobo and Ramji Venkataramanan - add arXiv link.

All plots are generated using Python files.


# Required Packages
In order to run the files, the following Python libraries are required: _numpy_, _scipy_, _math_, _numba_, _cvxpy_ and _matplotlib_. In addition, library _tikzplotlib_ can be used to generate tikz files corresponding to each of the plots.

# Python Scripts
## pool_amp.py
Includes functions for matrix creation and transformations, as well as the denoising function, state evolution (SE) and approximate message passing (AMP) for pooled data testing. This script also includes functions to implement linear programming (LP), convex optimization (CVX) and iterative hard thresholding (IHT). 

## amp_qgt.py
Includes functions for matrix and vector creation and transformations, as well as the denoising function and approximate message passing (AMP) for quantitative group testing (QGT). This script also includes functions to implement linear programming (LP), convex programming (CVX) for QGT.

## se_qgt.py
Includes functions to compute the corresponding state evolution (SE) to AMP QGT. 

## sc_amp_qgt.py
Includes functions for spatially coupled approximate message passing (SC-AMP) for quantitative group testing (QGT). This script also includes functions to implement spatially coupled convex programming (SC CVX) for QGT.

## sc_se_qgt.py
Includes functions to compute the corresponding spatially coupled state evolution (SE) to SC-AMP QGT. 

## sc_amp_pool.py
Includes functions for spatially coupled approximate message passing (SC-AMP) and state evolution (SC-SE) for matrix iterates, used for pooled data testing.

## sc_amp_qgt_fig2.py
Generates potential functions plot in Fig. 2 for $\pi=0.1$, $\sigma=1\times 10^{-30}$, for $\delta\in${0.02, 0.05, 0.1, 0.2, 0.3}.

## sc_amp_qgt_fig3.py
Generates correlation plots for AMP and SC AMP QGT noiseless, with defective probability $\pi=0.3$ compared to iid-SE and SC-SE (Fig. 3a) and iid and SC LP (Fig. 3b).

## sc_amp_qgt_fig4.py
Generates FPR vs FNR plots for AMP, SC-AMP and LP/CVX QGT with defective probability $\pi=0.3$, for noiseless, $\delta=0.38$ (Fig. 4a) and $\sigma=0.04, \delta=0.46$ (Fig. 4b).

## sc_amp_qgt_fig5.py
Generates correlation plots for AMP and SC AMP with noiseless pooled data, compared to iid-SE and SC-SE (Fig. 5a) and iid and SC LP (Fig. 5b).

## sc_amp_qgt_fig6.py
Generates correlation plots for AMP and SC AMP with noiseless pooled data, compared to columnwise SC-AMP QGT on each column of the pooled data.

