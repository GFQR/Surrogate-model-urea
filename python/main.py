import os
import argparse
import pathlib
from xml.parsers.expat import model
import numpy as np

import db
import rescale
import regression
import plot
import utils

cwd = pathlib.Path.cwd()
rescale_bool = False


# ----------------------------------------------------------------------------
def main():
    '''
    Orchestration module
    '''
    os.system('cls||clear')

    # cmd line arguments
    parser = argparse.ArgumentParser(
        description="Surrogate model for hyperpolarizability of urea"
    )
    parser.add_argument(
        "--charge", "-q",
        type=float,
        help="Choose charge (q) for the plot beta.vs.theta at fixed q"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Choose input db file (input.db)"
    )
    parser.add_argument(
        "--ridge", "-r",
        type=str,
        help="Choose on/off for regularization"
    )
    args = parser.parse_args()

    # title
    print("-----------------------------------------------")
    print("Surrogate model for hyperpolarizability of urea")
    print("-----------------------------------------------")
    print()
    print("input database: ", args.input)
    print("charge for visualization: ", args.charge)
    print()


    # Read the database
    db_file = cwd / "../data" / args.input
    Xbeta_set = db.read_db(db_file)

    # original data
    Xbeta_np = np.array(Xbeta_set)
    q = Xbeta_np[:,0]
    theta = Xbeta_np[:,1]
    beta = Xbeta_np[:,2]

    # rescale q / no need to rescale trigonometric function
    if args.ridge == "on":
        qresc, qmean, qstd = rescale.std_scaling(q)
        print("rescale on")
        print()
    else:
        qresc, qmean, qstd = q, 0, 1
        print("rescale off")
        print()

    # user defines q_scalar and script find qresc_scalar to use in plot and utils
    q_scalar = args.charge # user defined: what q to plot?
    # q_index = np.argmax(q == q_scalar)
    qresc_scalar = (q_scalar - qmean) / qstd # find the corresponding qresc_scalar to use in plot and utils

    # derived expansion terms
    #const = np.ones(qresc.shape)
    q2resc = np.square(qresc)
    cost = np.cos(theta)
    cos2t = np.cos(2*theta)
    sint = np.sin(theta)
    sin2t = np.sin(2*theta)
    qresc_cost = qresc*cost
    qresc_cos2t = qresc*cos2t
    qresc_sint = qresc*sint
    qresc_sin2t = qresc*sin2t
    # new terms (March 8, 2026)
    qresc2_cost = qresc*qresc*cost
    qresc2_sint = qresc*qresc*sint
    cos3t = np.cos(3*theta)
    sin3t = np.sin(3*theta)
    qresc_cos3t = qresc*cos3t
    qresc_sin3t = qresc*sin3t


    X_col = np.column_stack((qresc, q2resc, cost, sint, cos2t, sin2t,
                             qresc_cost, qresc_sint, qresc_cos2t, qresc_sin2t,
                             qresc2_cost, qresc2_sint, cos3t, sin3t, qresc_cos3t,
                             qresc_sin3t))
    
    R2, coeff, intercept, mse, beta_pred, model = regression.harmt_polq(X_col, beta, alpha = 0)

    # 1D and 2D tables with regression predictions
    fixq_theta_1D = utils.table_regression_1D(model, qresc_scalar)
    qresc_2D, theta_2D, beta_pred_2D = utils.table_regression_2D(model, qresc)
    dalton_data = np.column_stack((qresc, theta, beta)) 
    
    # plots for fix q and for the whole 2D space
    plot.P_2D(dalton_data, model, q_scalar, qresc_scalar)   
    plot.P_3D(dalton_data, (qresc_2D, theta_2D, beta_pred_2D))

    # predictions and metrics
    print("R2:", R2)
    print("mse:", mse)
    print("intercept:", intercept)
    if not rescale_bool:
        print("Coefficients:", coeff)
        
    else:
        print("To see coefficients, turn rescale_bool=False")

    # find the maximum beta and corresponding theta for the fixed q_scalar
    theta_max1D, beta_max1D = utils.find_max1D(fixq_theta_1D)
    print(f"fixed q: {q_scalar:.3f}, maximum beta(theta = {theta_max1D*180/np.pi:.1f} deg) = {beta_max1D:.3f}")

    # find the maximum beta and corresponding q and theta in the whole 2D space
    q_max2D, theta_max2D, beta_max2D = utils.find_max2D(beta_pred_2D, qresc_2D, theta_2D)
    q_max_2D_orig = q_max2D*qstd + qmean # find the corresponding q in original scale
    print(f"maximum beta(charge = {q_max_2D_orig:.3f}, theta = {theta_max2D*180/np.pi:.1f} deg) = {beta_max2D:.3f}")

    input("Press Enter to close plots...")

if __name__ == "__main__":
    main()