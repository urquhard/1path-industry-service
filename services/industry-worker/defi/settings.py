import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import copy
from pydantic import BaseSettings
from collections import Counter


import warnings
warnings.filterwarnings('ignore')


class Settings(BaseSettings):
    
    
    @staticmethod
    def estimate_and_sample_cov_mat(resid_frame, returns, Beta, INDUSTRY_RETURNS, x_frame, day = '2022-06-01'):
    
        # estimate mat = beta * industry_returns.cov() * beta' + diag_mat(resid.cov())
        # sample mat = returns.cov()

        b = copy.deepcopy(x_frame[x_frame.date == day])
        b.drop(columns = ['date'], inplace = True)
        tmp_tokens = list(b.index)
        ind_copy = copy.deepcopy(INDUSTRY_RETURNS.loc[:day])
        factors = list(b.columns)
        ind_copy = ind_copy[factors]
        test_resid_frame = resid_frame[b.index].loc[:day]
        b = np.matrix(b)
        ind = np.matrix(ind_copy.cov().mul(365)) 
        res_diag = np.diag(test_resid_frame.cov().mul(365))
        diag_mat = np.diag(res_diag)
        estimate_mat = b * ind * b.transpose() + diag_mat
        estimate_frame = pd.DataFrame(index = test_resid_frame.columns, columns = test_resid_frame.columns)  
        tokens_dict = {}
        i = 0
        for token in test_resid_frame.columns:
            tokens_dict[token] = i
            i += 1
        for token in test_resid_frame.columns:
            estimate_frame.loc[token] = estimate_mat[tokens_dict[token]]
        returns_copy = copy.deepcopy(returns[:day])
        sample_mat = returns_copy[tmp_tokens].cov().mul(365)
        estimate_frame = estimate_frame.astype('float')
        risk = res_diag / np.diag(estimate_frame)

        return estimate_frame, sample_mat, risk
    
    @staticmethod
    def users_weights(wallet, x_frame, geckoId_to_symbol, resid_frame):

        # calculates the weights of the user's portfolio at each point in time, based on data that we can evaluate. Returns a dict with weights and rebalancing times

        final_dict = {}
        w_dict = {}
        change_dates = []
        w_dates = []
        resid_dates = []

        for day in list(wallet.day.unique()):
            w_dates.append(str(day).split(' ')[0])

        for day in list(resid_frame.index):
            resid_dates.append(str(day).split(' ')[0])

        resid_frame['ind'] = resid_dates
        resid_frame.set_index('ind', inplace = True)


        amount_list = []

        tokens = []
        tmp_tokens = list(x_frame[x_frame.date == w_dates[0]].index.unique())
        for t in tmp_tokens:
            tokens.append(geckoId_to_symbol.loc[t].symbol)
        r = copy.deepcopy(resid_frame.loc[:w_dates[0]].cov().mul(365))


        tmp_frame = copy.deepcopy(wallet[wallet.day == w_dates[0]])
        tmp_frame = tmp_frame[tmp_frame['token'].isin(tokens)]
        tmp_frame.drop(columns = ['wallet_address', 'day'], inplace = True)
        t = list(tmp_frame.token)
        r_tokens = list(r.loc[t][t].iloc[0].dropna().index)
        tmp_frame = tmp_frame[tmp_frame['token'].isin(r_tokens)]


        sum_value = tmp_frame['value'].sum()
        for token in tmp_frame['token'].unique():

            tmp_frame2 = tmp_frame[tmp_frame['token'] == token]
            if tmp_frame2.shape[0] > 1:
                tmp_frame3 = pd.DataFrame(0.0, columns = ['amount', 'value', 'token'], index = [0])
                tmp_frame3.loc[0] = [tmp_frame2['amount'].sum(), tmp_frame2['value'].sum(), token]
                tmp_frame2 = tmp_frame3

            w_dict[token] = tmp_frame2['value'].values[0] / sum_value
            amount_list.append(tmp_frame2['amount'].values[0])
        final_dict[w_dates[0]] = w_dict



        for j in tqdm(range(1, len(w_dates))):

            w_dict = {}
            tokens = []
            tmp_amount_list = []
            tmp_tokens = list(x_frame[x_frame.date == w_dates[j]].index.unique())
            for t in tmp_tokens:
                tokens.append(geckoId_to_symbol.loc[t].symbol)

            r = copy.deepcopy(resid_frame.loc[:w_dates[j]].cov().mul(365))


            tmp_frame = copy.deepcopy(wallet[wallet.day == w_dates[j]])
            tmp_frame = tmp_frame[tmp_frame['token'].isin(tokens)]
            tmp_frame.drop(columns = ['wallet_address', 'day'], inplace = True)
            t = list(tmp_frame.token)
            r_tokens = list(r.loc[t][t].iloc[0].dropna().index)
            tmp_frame = tmp_frame[tmp_frame['token'].isin(r_tokens)]

            for token in tmp_frame['token'].unique():

                tmp_frame2 = tmp_frame[tmp_frame['token'] == token]
                if tmp_frame2.shape[0] > 1:
                    tmp_frame3 = pd.DataFrame(0.0, columns = ['amount', 'value', 'token'], index = [0])
                    tmp_frame3.loc[0] = [tmp_frame2['amount'].sum(), tmp_frame2['value'].sum(), token]
                    tmp_frame2 = tmp_frame3

                tmp_amount_list.append(tmp_frame2['amount'].values[0])

            tmp_amount_list.sort()
            amount_list.sort()

            if tmp_amount_list == amount_list:
                final_dict[w_dates[j]] = final_dict[w_dates[j-1]]
                continue
            else:
                tmp_amount_list = []
                sum_value = tmp_frame['value'].sum()

                for token in tmp_frame['token'].unique():

                    tmp_frame2 = tmp_frame[tmp_frame['token'] == token]
                    if tmp_frame2.shape[0] > 1:
                        tmp_frame3 = pd.DataFrame(0.0, columns = ['amount', 'value', 'token'], index = [0])
                        tmp_frame3.loc[0] = [tmp_frame2['amount'].sum(), tmp_frame2['value'].sum(), token]
                        tmp_frame2 = tmp_frame3

                    w_dict[token] = tmp_frame2['value'].values[0] / sum_value
                    tmp_amount_list.append(tmp_frame2['amount'].values[0])
                amount_list = tmp_amount_list    
                final_dict[w_dates[j]] = w_dict
                change_dates.append(w_dates[j])

        return final_dict, change_dates

    
    @staticmethod
    def user_performance(wallet_dict, returns):
        
        user_performance_frame = pd.DataFrame(0.0, index = wallet_dict.keys(), columns = returns.columns)
        for date in wallet_dict.keys():
            tmp_tokens = list(wallet_dict[date].keys())
            for token in tmp_tokens:
                user_performance_frame.loc[date][token] = wallet_dict[date][token] * returns.loc[date][token]
        
        return user_performance_frame
    
    @staticmethod
    def risks_for_assets_in_portfolio(estimate_mat, portfolio_dict = {'ETH': 1.0}):
    
        # Using the Euler theorem, decomposes the risk of a portfolio by tokens in this portfolio. Returns a dataframe with margin, quantity and percentage risks for each token
    
        rows = list(estimate_mat.dropna().index)
        estimate_mat = estimate_mat.loc[rows][rows]

        risk_frame = pd.DataFrame(0.0, index = estimate_mat.columns, columns = ['marginal', 'not marginal', 'percent'])

        portf_mat = pd.DataFrame(0.0, columns = estimate_mat.columns, index = [0])
        tokens = portf_mat.columns

        for key in portfolio_dict.keys():
            if key in list(estimate_mat.columns):
                portf_mat[key] = portfolio_dict[key]

        all_risk = np.matrix(portf_mat) * np.matrix(estimate_mat) * np.matrix(portf_mat).transpose()
        all_risk = all_risk[0, 0]
        marginal_val = np.matrix(portf_mat) * np.matrix(estimate_mat)
        m_val_list = []

        for i in range(marginal_val.shape[1]):
            m_val_list.append(marginal_val[0, i] / (all_risk ** 0.5))

        risk_frame['marginal'] = m_val_list
        for i in range(len(m_val_list)):
            risk_frame.loc[tokens[i], 'not marginal'] = (portf_mat[tokens[i]] * m_val_list[i]).values[0] 
        for i in range(len(m_val_list)):
            risk_frame.loc[tokens[i], 'percent'] = (portf_mat[tokens[i]] * m_val_list[i] / all_risk ** 0.5).values[0]
        risk_frame['names'] = list(risk_frame.index)
        return risk_frame
    
    @staticmethod
    def portfolio_riks_contribution(x_frame, INDUSTRY_RETURNS, resid_frame, day = '2022-06-01',
                                     portfolio_dict = {'ETH':1.0}):
                                     
        # Using the Euler theorem, decomposes the risk of the portfolio into categories and individual risks of tokens. Returns a dataframe with margin, quantity and percentage risk for each category
    

        tokens = list(portfolio_dict.keys())
        factors = copy.deepcopy(x_frame[x_frame.date == day].loc[tokens])
        factors.drop(columns = ['date'], inplace = True)
        f = list(factors.columns)
        omega = INDUSTRY_RETURNS[f].loc[:day].cov().mul(365)
        D = resid_frame.loc[:day].cov().mul(365).loc[tokens][tokens]
        D = np.diag(D)
        D = np.diag(D)
        weights = pd.DataFrame(list(portfolio_dict.values()), index = tokens, columns = [0])
        weights = weights.T
        w = np.matrix(weights)
        factors_t = copy.deepcopy(factors)

        columns_to_new_factors = []
        for i in range(len(tokens)):
            columns_to_new_factors.append('e_' + tokens[i])
        add_tokens_factors = pd.DataFrame(0.0, index = tokens, columns = columns_to_new_factors)
        for i in range(add_tokens_factors.shape[0]):
            for j in range(add_tokens_factors.shape[1]):
                if i == j:
                    tmp_column = 'e_' + tokens[j]
                    add_tokens_factors.iloc[j][tmp_column] = 1.0
        for col in add_tokens_factors.columns:
            factors_t[col] = add_tokens_factors[col]

        b_t = np.matrix(factors_t)
        O_f = np.block([[omega, np.zeros((omega.shape[0], len(tokens)))], [np.zeros((len(tokens), omega.shape[0])), D]])
        sigma = (w * b_t * O_f * (w * b_t).transpose()).transpose()[0, 0] ** 0.5

        risk_frame = pd.DataFrame(0.0, index = factors_t.columns, columns = ['marginal', 'factor', 'percent'])
        marginal_vector = w * b_t * O_f
        factors_vector = np.multiply( w * b_t * O_f, w * b_t)

        for i in range(len(factors.columns)):
            risk_frame.iloc[i]['marginal'] = marginal_vector[0, i]
        for i in range(len(factors_t.columns)):
            risk_frame.iloc[i]['factor'] = factors_vector[0, i] / sigma
        risk_frame['percent'] = risk_frame['factor'] / sigma
        risk_frame['names'] = list(risk_frame.index)
        risk_frame_copy = copy.deepcopy(risk_frame.iloc[:17])
        risk_frame_copy.loc['tokens_risk'] = [0.0, risk_frame.iloc[17:].sum()['factor'],
                                              risk_frame.iloc[17:].sum()['percent'], 'tokens_risk']

        return risk_frame_copy
    
    @staticmethod
    def optimize_risk_in_portfolio(Beta, x_frame, industry_returns, resid_frame, returns, final_dict,
                                   geckoId_to_symbol, rebalancing_frequency = 14):

        dates = list(final_dict.keys())
        new_portfolio = {}
        change_dates = []

        gts = {}
        for ind in geckoId_to_symbol.index:
            gts[ind] = geckoId_to_symbol.loc[ind].symbol

        resid_frame_copy = copy.deepcopy(resid_frame)
        resid_frame_copy.rename(columns = gts, inplace = True)
        x_frame_copy = copy.deepcopy(x_frame)
        x_frame_copy.rename(index = gts, inplace = True)
        returns_copy = copy.deepcopy(returns)
        returns_copy.rename(columns = gts, inplace = True)
        swap_freq = 0

        for i in tqdm(range(len(dates))):


            if (i == 0) or ((i > 0) and list(new_portfolio[dates[i-1]].keys()) != \
                                                    list(final_dict[dates[i]].keys())) or (i == \
                                                                                           swap_freq + rebalancing_frequency):

                estimate_mat, sample_mat, risk = \
                settings.estimate_and_sample_cov_mat(resid_frame, returns, Beta, industry_returns, x_frame,
                                                     day = dates[i])
                estimate_mat_copy = copy.deepcopy(estimate_mat)
                estimate_mat_copy.rename(columns = gts, inplace = True)
                estimate_mat_copy.rename(index = gts, inplace = True)

                risk_frame = settings.portfolio_riks_contribution(x_frame_copy, industry_returns, resid_frame_copy,
                                                                  day = dates[i], 
                                                                  portfolio_dict = final_dict[dates[i]])
                tmp_risk_frame = copy.deepcopy(risk_frame)['marginal']
                factors = list(set(tmp_risk_frame[tmp_risk_frame < 0].index) -\
                               set(['VMR', 'factor_log_mc','factor_log_mc_tvl', 'Mbeta', 'factor_momentum', 'other_chains',
                                    'other_categories']))
                unique_tokens_to_factor = {}
                all_unique_tokens = []
                for f in range(len(factors)):
                    unique_tokens = []
                    x_copy = copy.deepcopy(x_frame_copy[x_frame_copy.date == dates[i]][factors[f]])

                    if f == 0:

                        for token in list(final_dict[dates[i]].keys()):
                            if x_copy[token] != 0:
                                unique_tokens.append(token)

                        current_token = unique_tokens[0]
                        current_risk = estimate_mat_copy.loc[current_token][current_token] ** 0.5
                        for token in unique_tokens:
                            risk = estimate_mat_copy.loc[token][token] ** 0.5
                            if risk <= current_risk:
                                current_token = token
                                current_risk = risk


                        unique_tokens_to_factor[factors[f]] = current_token
                        all_unique_tokens.append(current_token)
                    else:

                        for token in list(final_dict[dates[i]].keys()):
                            if x_copy[token] != 0 and token not in all_unique_tokens:
                                unique_tokens.append(token)
                        if len(unique_tokens) == 0:
                            for token in list(final_dict[dates[i]].keys()):
                                 if x_copy[token] != 0:
                                    unique_tokens.append(token) 

                        current_token = unique_tokens[0]
                        current_risk = estimate_mat_copy.loc[current_token][current_token] ** 0.5
                        for token in unique_tokens:
                            risk = estimate_mat_copy.loc[token][token] ** 0.5
                            if risk <= current_risk:
                                current_token = token
                                current_risk = risk

                        unique_tokens_to_factor[factors[f]] = current_token
                        all_unique_tokens.append(current_token)

                tmp_portfolio = copy.deepcopy(final_dict[dates[i]])

                for f in range(len(factors)):

                    tmp_value = risk_frame['marginal'][factors[f]]
                    iteration = 0

                    while tmp_value < 0:

                        if tmp_portfolio[unique_tokens_to_factor[factors[f]]] > 0.3:
                            break

                        tmp_portfolio[unique_tokens_to_factor[factors[f]]] += 0.01
                        sum_value = 0
                        for token in list(set(tmp_portfolio.keys()) - set([unique_tokens_to_factor[factors[f]]])):
                            sum_value += tmp_portfolio[token]

                        for token in list(set(tmp_portfolio.keys()) - set([unique_tokens_to_factor[factors[f]]])):
                            tmp_portfolio[token] /= sum_value
                            tmp_portfolio[token] *= (1 - tmp_portfolio[unique_tokens_to_factor[factors[f]]])

                        risk_frame = \
                        settings.portfolio_riks_contribution(x_frame_copy, industry_returns, resid_frame_copy,
                                                             day = dates[i], 
                                                             portfolio_dict = tmp_portfolio)

                        tmp_value = risk_frame['marginal'][factors[f]]
                        iteration += 1
                        if iteration == 30:
                            break

                new_portfolio[dates[i]] = tmp_portfolio
                change_dates.append(dates[i])
                swap_freq = i

            else:

                tmp_dict = {}
                ret_copy = copy.deepcopy(returns_copy.loc[dates[swap_freq]:dates[i]])
                tmp_ret_copy = ret_copy[list(new_portfolio[dates[swap_freq]].keys())]
                tmp_ret_copy = tmp_ret_copy.add(1).cumprod()
                for token in tmp_ret_copy.columns:
                    tmp_ret_copy.loc[(tmp_ret_copy.index == list(tmp_ret_copy.index)[-1]), token] *= \
                    new_portfolio[dates[i-1]][token]
                tmp_frame = tmp_ret_copy.iloc[-1]
                tmp_sum = tmp_frame.sum()
                for ind in tmp_frame.index:
                    tmp_dict[ind] = (tmp_frame[ind] / tmp_sum)
                new_portfolio[dates[i]] = tmp_dict

        return new_portfolio, change_dates
    
    @staticmethod
    def returns_decomposition(returns, x_frame, resid_frame, INDUSTRY_RETURNS, start_date = '2022-06-01', 
                          end_date = '2023-01-01', token_list = ['BAKE']):
    
        # decomposes returns of one token by model factors
        fators_to_token_dict = {}
        returns_dict = {}

        dates = list(returns[start_date:end_date].index)
        factors = list(INDUSTRY_RETURNS.columns)
        for token in token_list:

            tmp_df = pd.DataFrame(0.0, index = dates, columns = factors)
            tmp_df['resid'] = 0.0

            for day in dates:

                X_copy = copy.deepcopy((x_frame[x_frame.date == day])[factors])
                factors_vec = np.multiply(np.matrix(INDUSTRY_RETURNS.loc[day]), np.matrix(X_copy.loc[token]))
                for i in range(factors_vec.shape[1]):
                    tmp_df.loc[day, factors[i]] = factors_vec[0, i]
                tmp_df.loc[day, 'resid'] = resid_frame.loc[day, token]
            fators_to_token_dict[token] = tmp_df
            returns_dict[token] = returns[start_date:end_date][token]

        return fators_to_token_dict, returns_dict
    
    @staticmethod
    def portfolio_returns_decomposition(final_dict, industry_returns, x_frame, resid_frame, geckoId_to_symbol):
    
        gts = {}
        for ind in geckoId_to_symbol.index:
            gts[ind] = geckoId_to_symbol.loc[ind].symbol

        x_frame_copy = copy.deepcopy(x_frame)
        resid_frame_copy = copy.deepcopy(resid_frame)
        x_frame_copy.rename(index = gts, inplace = True)
        resid_frame_copy.rename(columns = gts, inplace = True)

        factors = list(industry_returns.columns)
        portfolio_risk_frame = pd.DataFrame(0.0, index = final_dict.keys(), columns = factors)
        portfolio_risk_frame['tokens_risk'] = 0.0

        for day in tqdm(final_dict.keys()):

            x_copy = \
            copy.deepcopy((x_frame_copy[x_frame_copy.date == day])[factors].loc[list(final_dict[day].keys())])
            for token in x_copy.index:
                x_copy.loc[token] *= final_dict[day][token]

            for f in factors:
                x_copy[f] *= industry_returns.loc[day][f]
            tmp_frame = x_copy.sum()
            for i in range(len(factors)):
                portfolio_risk_frame.loc[day, factors[i]] = tmp_frame[factors[i]]

            weights = pd.DataFrame(0.0, index = [day], columns = final_dict[day].keys())
            for col in weights.columns:
                weights[col] = final_dict[day][col]

            portfolio_risk_frame.loc[day, 'tokens_risk'] = \
            (np.matrix(resid_frame_copy.loc[day][list(final_dict[day].keys())]) * \
             np.matrix(weights).transpose())[0, 0]


        return portfolio_risk_frame
    
    # переписать нормально эту функцию
    @staticmethod
    def tokens_impact_of_portfolio(user_performance_frame):
    
        # the impact of tokens on the movement of the entire portfolio
        tmp_df = user_performance_frame.sum()
        tmp = tmp_df[tmp_df != 0.0].index
        tmp_df = user_performance_frame[tmp]
        ttt = (1 + tmp_df).cumprod()
        user_portf = (1 + user_performance_frame.sum(axis=1)).cumprod()
        user_portf /= user_portf.iloc[0]
        ttt /= ttt.iloc[0]

        t_t = ttt.min()
        ind1 = t_t[t_t >= 0.97].index
        t_t = ttt[ind1].max()
        ind2 = t_t[t_t <= 1.03].index
        tmp_ind = list(set(tmp_df.columns) - set(ind2))
        ttt = ttt[tmp_ind]
        ttt['portfolio'] = (1 + user_performance_frame.sum(axis=1)).cumprod().values
        ttt['portfolio'] /= ttt['portfolio'].iloc[0]
        
        kkk = copy.deepcopy(ttt)
        dates = list(ttt.index)
        for i in range(len(ttt) - 1):
            tmp5 = ttt.iloc[i+1] / ttt.iloc[i]
            tmp_tokens = list(tmp5[tmp5 == 1.0].index)
            for token in tmp_tokens:
                kkk.loc[dates[i+1], token] = np.nan
        
        return kkk
    
    @staticmethod
    def ID_to_symbol(Beta):
    
        # returns a dict with gecko_id to token symbol
    
        geckoId_to_symbol = {}
        for ID in tqdm(Beta.gecko_id.unique()):
            tmp_ticker = Beta[Beta.gecko_id == ID].symbol.unique()
            geckoId_to_symbol[ID] = tmp_ticker[0]
        c = Counter(list(geckoId_to_symbol.values()))
        list_of_tokens = []
        c = dict(c.items())
        for key in c.keys():
            if c[key] >= 2:
                list_of_tokens.append(key)
        for key in geckoId_to_symbol.keys():
            if geckoId_to_symbol[key] in list_of_tokens:
                geckoId_to_symbol[key] = key + '_' + geckoId_to_symbol[key]

        return geckoId_to_symbol
    
    @staticmethod
    def portfolio_weights(resid_frame, returns, x_frame, weights, geckoId_to_symbol):
    
        returns_dates = []
        for ind in returns.index:
            returns_dates.append(str(ind).split(' ')[0])
        final_date = list(weights.keys())[0]
        final_ind = returns_dates.index(final_date)
        dates = returns_dates[final_ind + 1 - 365 : final_ind + 1]
        final_portfolio = {}

        gts = {}
        for ind in geckoId_to_symbol.index:
            gts[ind] = geckoId_to_symbol.loc[ind].symbol

        resid_frame_copy = copy.deepcopy(resid_frame)
        resid_frame_copy.rename(columns = gts, inplace = True)
        x_frame_copy = copy.deepcopy(x_frame)
        x_frame_copy.rename(index = gts, inplace = True)

        tokens_to_date = {}

        tmp_tokens = list(set.intersection(set(resid_frame_copy.columns), set(weights[final_date].keys())))

        for i in range(len(dates)):


            r = copy.deepcopy(resid_frame_copy[tmp_tokens].loc[:dates[i]].cov().mul(365))
            r = r[~r.loc[r.columns[0]].isna()]

            final_tokens = list(set.intersection(set(r.index),
                                            set(x_frame_copy[x_frame_copy.date == dates[i]].dropna().index)))

            tmp_dict = {}
            tokens_to_drop = list(set(weights[final_date].keys()) - set(final_tokens))
            tmp_sum = 0
            for token in final_tokens:
                tmp_sum += weights[final_date][token]
                tmp_dict[token] = weights[final_date][token]
            for token1 in final_tokens:
                coef = tmp_dict[token1] / tmp_sum
                for token2 in tokens_to_drop:
                    tmp_dict[token1] += coef * weights[final_date][token2]

            final_portfolio[dates[i]] = tmp_dict

        return final_portfolio
    
    @staticmethod
    def print_portfolio_stats(industry_returns, final_dict, returns, geckoId_to_symbol, resid_frame, 
                              Beta, x_frame):


        start_date = list(final_dict.keys())[0]
        end_date = list(final_dict.keys())[-1]

        gts = {}
        for ind in geckoId_to_symbol.index:
            gts[ind] = geckoId_to_symbol.loc[ind].symbol

        returns_copy = copy.deepcopy(returns)
        returns_copy.rename(columns = gts, inplace = True)

        user_performance_frame = pd.DataFrame(0.0, index = final_dict.keys(), columns = returns_copy.columns)
        for date in final_dict.keys():
            tmp_tokens = list(final_dict[date].keys())
            for token in tmp_tokens:
                try:
                    user_performance_frame.\
                    loc[(pd.to_datetime(date) + timedelta(days = 2)).strftime('%Y-%m-%d')][token] = final_dict[date][token] *\
                    returns_copy.loc[(pd.to_datetime(date) + timedelta(days = 2)).strftime('%Y-%m-%d')][token]
                except:
                    pass

        s_d = (pd.to_datetime(start_date) + timedelta(days = 2)).strftime('%Y-%m-%d')
        tmp_df = copy.deepcopy(industry_returns.loc[s_d:end_date])
        ind = list(user_performance_frame[2:].index)
        tmp_df['index'] = ind
        tmp_df.set_index('index', inplace = True)
        u = (1 + user_performance_frame.sum(axis=1)).cumprod()
        m = (1 + tmp_df['const']).cumprod()
        m = pd.concat([pd.Series([1.0], index = [(pd.to_datetime(m.index[0]) - timedelta(days = 1)).strftime('%Y-%m-%d')]), m])
        m = pd.concat([pd.Series([1.0], index = [(pd.to_datetime(m.index[0]) - timedelta(days = 1)).strftime('%Y-%m-%d')]), m])
        d = u-m


        estimate_mat, sample_mat, risk = \
        settings.estimate_and_sample_cov_mat(resid_frame, returns, Beta, industry_returns, x_frame, day = end_date)

        estimate_mat_copy = copy.deepcopy(estimate_mat)
        x_frame_copy = copy.deepcopy(x_frame)
        resid_frame_copy = copy.deepcopy(resid_frame)
        x_frame_copy.rename(index = gts, inplace = True)
        resid_frame_copy.rename(columns = gts, inplace = True)
        estimate_mat_copy.rename(columns = gts, inplace = True)
        estimate_mat_copy.rename(index = gts, inplace = True)

        risk_frame = settings.portfolio_riks_contribution(x_frame_copy, industry_returns, resid_frame_copy, 
                                                          day = end_date, portfolio_dict = final_dict[end_date])
        x = risk_frame.sum().factor

        l = list(np.diag(estimate_mat_copy) ** 0.5)
        l.pop(l.index(max(l)))
        l.pop(l.index(max(l)))
        min_risk = min(l)
        max_risk = max(l)

        y1 = 5.0
        y2 = 1.0
        x1 = min_risk
        x2 = max_risk

        y = (((y2 - y1) * (x - x1)) / (x2 - x1)) + y1


        u_per_print = 'Your portfolio performance is ' + str(round(((u[-1] - u[0]) / u[0])*100, 2)) + '%'
        u_to_m_per_print = \
        'Compared to the market, your portfolio performance is ' + str(round(((u[-1] - m[-1]) / u[-1]) * 100, 2)) + '%'
        u_to_m_av_per_print = 'Compared to the market, your average performance is ' + \
        str(round(d.mean() * 100, 2)) + '%'
        sharpe_print = 'Sharpe ratio of the portfolio is ' + str(round(u.mean() / (u.std() * (365 ** 0.5)), 2))
        diversification_print = 'The diversification rating of the portfolio is ' + str(round(y, 2)) + ' out of 5. This rating indicates how well your portfolio is diversified across various risk factors, with 5 representing perfect diversification. For a comprehensive analysis of the risk in your portfolio, please access the "Risk Profile" tab.'

        print(u_per_print)
        print(u_to_m_per_print)
        print(u_to_m_av_per_print)
        print(sharpe_print)
        print(diversification_print)

        return u_per_print + u_to_m_per_print + u_to_m_av_per_print + sharpe_print + diversification_print
    
    @staticmethod
    def risk_comparison(resid_frame, returns, Beta, industry_returns, x_frame, old_dict, new_dict, geckoId_to_symbol):
    
        gts = {}
        for ind in geckoId_to_symbol.index:
            gts[ind] = geckoId_to_symbol.loc[ind].symbol

        date = list(old_dict.keys())[-1]

        estimate_mat, sample_mat, risk = \
        settings.estimate_and_sample_cov_mat(resid_frame, returns, Beta, industry_returns, x_frame, 
                                             day = date)

        estimate_mat_copy = copy.deepcopy(estimate_mat)
        estimate_mat_copy.rename(columns = gts, inplace = True)
        estimate_mat_copy.rename(index = gts, inplace = True)

        old_df = pd.DataFrame(old_dict[date], index = [date])
        new_df = pd.DataFrame(new_dict[date], index = [date])

        o_r = (np.matrix(old_df) * np.matrix(estimate_mat_copy.loc[list(old_df.columns)][list(old_df.columns)]) * \
        np.matrix(old_df).transpose())[0, 0]
        n_r = (np.matrix(new_df) * np.matrix(estimate_mat_copy.loc[list(new_df.columns)][list(new_df.columns)]) * \
        np.matrix(new_df).transpose())[0, 0] 

        old_risk = o_r ** 0.5
        new_risk = n_r ** 0.5

        percent_risk = round((1 - new_risk / old_risk) * 100, 2)

        if percent_risk <= 0:
            str_to_print = 'Well done on maintaining a balanced portfolio! Remember to regularly monitor the market for any changes and adjust your strategy accordingly. Stay proactive and make informed decisions to maximize your investment success'
        else:
            number_to_increase = 0
            number_to_decrease = 0
            for token in old_dict[date].keys():
                if old_dict[date][token] <= new_dict[date][token]:
                    number_to_increase += 1
                else:
                    number_to_decrease += 1
            str_to_print = 'To reduce the total risk by ' + str(percent_risk) + ' percent and potentially enhance the portfolio performance, 1path recommends adjusting the allocation of your tokens. Specifically, consider lowering the share of ' + str(number_to_decrease) + ' tokens and increasing shares of ' + str(number_to_increase) + ' tokens.' 

        return str_to_print
    

settings = Settings()