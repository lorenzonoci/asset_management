import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from pre_proc import create_batches, get_labels, compute_windows
from pre_proc import scaler
from lstm import LSTM
import tensorflow as tf


def pf_optim(risk_aversion, mu, sigma):

    # Setup workspace
    """
    setup(self, P=None, q=None, A=None, l=None, u=None, **settings):
            Setup OSQP solver problem of the form
            minimize     1/2 x' * P * x + q' * x
            subject to   l <= A * x <= u
    """
    k = len(mu)                         # dimension of the problem

    problem = osqp.OSQP()
    A = np.concatenate((np.ones((1, k)), np.eye(k)), axis=0)
    sA = sparse.csr_matrix(A)
    # short selling
    l = np.hstack([1, -1 + np.zeros(k)])
    u = np.ones(k + 1)
    sCov = sparse.csr_matrix(risk_aversion*sigma)

    problem.setup(sCov, -mu, sA, l, u)

    # Solve problem
    res = problem.solve()

    return res.x


def optimize_portfolio(df_close, mu, cov, frequency, iter, risk_aversion=1):
    '''
    Optimize portfolio weights and compute profit and losses
    :param df_close: returns DataFrame
    :param mu: vector of expected returns
    :param cov: covariance matrix
    :param frequency: frequency of optimization
    :param iter:
    :param risk_aversion:
    :return: portfolio weights and profit and losses
    '''

    optim_portfolio = pf_optim(risk_aversion, mu, cov)
    pnl = np.dot(df_close.iloc[iter*frequency: (iter+1)*frequency], optim_portfolio)
    return optim_portfolio, pnl


def show_results(pnls):
    ax = pd.DataFrame(pnls).cumsum().plot()
    ax.legend(['profit and loss'])
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative returns")
    plt.show()


def compute_performance(port_ret):
    annualized_return = np.mean(np.log(port_ret+1))*252
    # AnnualizedVolatility
    annualized_volatility = np.std(np.log(port_ret+1))*np.sqrt(252)
    # Information Ratio
    information_ratio = annualized_return/annualized_volatility
    performance = [annualized_return, annualized_volatility, information_ratio]
    return performance

def regularize_covariance_matrix(cov, gamma=0.000001):
    regularized_cov = cov + np.diag(np.diag(gamma * np.ones_like(cov)))
    return regularized_cov


def backtesting_optim_portfolio(df_close, test_data, test_labels, other_data, lstm_model, sess, risk_adv, output_window, net_config, regularize_cov=False, bayes=False):
    '''

    :param df_close: DataFrame of returns
    :param test_data:
    :param test_labels:
    :param other_data: other past data with which re-optimize the lstm model
    :param lstm_model: the lstm based neural network model
    :param sess: tf Session
    :param risk_adv: risk adversion
    :param output_window: output window of predictions
    :param net_config: configuration file object
    :param regularize_cov: if True, apply regularization on the diagonal of covariance matrix
    :param bayes: if True, combines past data (as a prior) and lstm predictions
    :return: portfolio weights and profit and losses
    '''
    all_weights = []
    pnls = []
    tot_days = test_data.shape[0]
    indexes_reoptimzation = get_indexes(net_config.frequency_optimize_lstm, net_config.frequency, tot_days)
    i = 0
    while i < int(tot_days / net_config.frequency):
        feed_dict = {
            lstm_model.inputs: test_data[i * net_config.frequency:i * net_config.frequency + 1, :, :],
            lstm_model.dropout_prob: 1.0
        }
        preds = sess.run(lstm_model.predicted_outputs, feed_dict=feed_dict)
        preds = preds.reshape(output_window, preds.shape[-1])
        preds = scaler.inverse_transform(preds)

        mus = np.mean(pd.DataFrame(preds).pct_change().dropna().values, axis=0)
        covs = np.cov(pd.DataFrame(preds).pct_change().dropna().values.T)

        if bayes:
            past_days = test_data[i * net_config.frequency:i * net_config.frequency + 1, :, :].reshape(
                [net_config.windows_size, preds.shape[-1]])
            past_days = scaler.inverse_transform(past_days)
            prior_ret = np.mean(pd.DataFrame(past_days[-40:]).pct_change().dropna().values, axis=0)
            prior_covs = np.cov(pd.DataFrame(past_days[-40:]).pct_change().dropna().values.T)

            prior_ret_weight = 0.5
            mus = prior_ret_weight*prior_ret + (1-prior_ret_weight)*mus

            prior_cov_weight = 0.5
            covs = prior_cov_weight*prior_covs + (1-prior_cov_weight)*covs

        if regularize_cov:
            covs = regularize_covariance_matrix(covs)

        weights, pnl = optimize_portfolio(df_close, mus, covs,
                                          frequency=net_config.frequency, iter=i,
                                          risk_aversion=risk_adv)
        i = i + 1
        all_weights.append(weights)
        pnls.append(pnl)

        if i in indexes_reoptimzation:
            train_data = np.concatenate([other_data,
                                         test_data[0:i*net_config.frequency, 0, :]],
                                        axis=0)

            train_labels = get_labels(train_data, window_size=net_config.windows_size, output_window=output_window)
            train_data = train_data[:-output_window]
            train_data = compute_windows(train_data, windows_size=net_config.windows_size)
            td = create_batches(train_data,
                                batch_size=net_config.batch_size)
            tl = create_batches(train_labels,
                                batch_size=net_config.batch_size)
            LSTM.train_lstm(sess, lstm_model, td, tl, test_data[i*net_config.frequency:, :, :], test_labels[i * net_config.frequency:, :, :],
                            n_epochs=net_config.lstm_epochs, dropout_prob=net_config.dropout_prob, initialize=True)

    pnls = np.concatenate(pnls, axis=0)
    return np.stack(all_weights, axis=0), pnls


def get_indexes(frequency_reoptimize, frequency_portfolio, tot_days):
    tot_iter = tot_days / frequency_portfolio
    n_iter = tot_days/frequency_reoptimize
    step_size = int(tot_iter / n_iter)
    return np.arange(step_size, tot_iter, step_size, dtype=int)


def optimal_risk_aversion(risk_aversion_vec, df_close, test_data, test_labels, other_data, lstm_model, output_window, sess, net_config):
    '''
    compute optimal portfolio weights and performances with given vector of risk aversion coefficients
    return the best risk aversion coefficient, the portfolio weights obtained with such coefficient and
    its performances.
    :param risk_aversion_vec:
    :param mus_list:
    :param covs_list:
    :return:
    '''
    port_ret_vec = []
    performance_df = []
    optimized_weights_riskAversion = []
    for risk_aversion in risk_aversion_vec:
        lstm_model.saver.restore(sess, tf.train.latest_checkpoint('lstm-price/'))
        all_weights, port_ret = backtesting_optim_portfolio(
            df_close, test_data, test_labels, other_data, lstm_model, sess, risk_aversion, output_window, net_config)
        optimized_weights_riskAversion.append(all_weights)
        port_ret_vec.append(port_ret)
        # show_results(port_ret)
        performance = compute_performance(port_ret)
        performance_df.append(performance)

    # plot performances for each case
    performance_df = pd.DataFrame(performance_df)
    performance_df = performance_df.rename(columns={performance_df.columns[0]: "Annualized Return", performance_df.columns[1]: "Annualized Std", performance_df.columns[2]: "Information Ratio"})

    performance_df = performance_df.rename(index=dict(zip(performance_df.index, risk_aversion_vec)))

    return performance_df.values


def plot_performance_results(performance_df, risk_aversion_vec):
    # plot performance results
    performance_df = performance_df.rename(index=dict(zip(performance_df.index,risk_aversion_vec)))
    performance_df.plot()
    plt.title('Best Risk Aversion parameter')
    plt.ylabel('Performances')
    plt.xlabel('Risk Aversion parameter')
    plt.legend(['Annualized Return', 'Annualized StdDev', 'Information Ratio'], loc='upper right')
    plt.show()
