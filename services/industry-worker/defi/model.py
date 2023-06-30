import pandas as pd
import numpy as np
import math as m
import statsmodels.api as sm  # 0.13.2!!!
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .settings import settings

import warnings
warnings.filterwarnings('ignore')


def factor_model(beta, categories_columns, blockchain_columns, style_factors, weight_type, power_wight):
    
    categories_in_reg = list(set(categories_columns) - {'Other'})
    chains_in_reg = list(set(blockchain_columns) - {'Other'})
    factors_in_reg = style_factors
    endog = ['current_return']
    resid_dict = {}
    fitted_dict = {}
    x_dict = {}
    pvalue_dict = {}
    x_frames = []

    industry_returns = pd.DataFrame(0.0, columns=['const'] + factors_in_reg +
                                    chains_in_reg + categories_in_reg, index=beta.date.unique())

    se = pd.DataFrame(0.0, columns=['const'] + factors_in_reg + chains_in_reg + categories_in_reg,
                      index=beta.date.unique())

    res_test = pd.DataFrame(0.0, columns=['R_2', 'R_2_adj',
                                          'F_test_regression', 'F_test_chains',
                                          'F_test_categories', 'F_test_styles',
                                          'residual_std'],
                            index=beta.date.unique())
    vif_values = pd.DataFrame(0.0, columns=['const'] + categories_in_reg + chains_in_reg + factors_in_reg,
                              index=beta.date.unique())

    resid_frame = pd.DataFrame(m.nan, columns=beta.gecko_id.unique(), index=beta.date.unique())
    fitted_frame = pd.DataFrame(m.nan, columns=beta.gecko_id.unique(), index=beta.date.unique())

    for t in tqdm(beta.date.unique()):

        x = beta[beta.date == t][categories_in_reg + chains_in_reg + factors_in_reg + endog + ['quarantine_flag'] +
                                 ['gecko_id']]
        if weight_type == 'power':
            x['W'] = np.power((np.exp(x['factor_log_mc']).replace([-np.inf, np.inf], np.nan)), power_wight)
        elif weight_type == 'log':
            x['W'] = x['factor_log_mc'].replace([-np.inf, np.inf], np.nan)
        elif weight_type == 'equal':
            x['W'] = 1
        x = x.set_index('gecko_id')
        
        # --- If we have style factors then do z-score calculations
        if factors_in_reg:
            x[factors_in_reg] = x[factors_in_reg].replace([-np.inf, np.inf], np.nan)
            x.loc[x['quarantine_flag'].isna(), factors_in_reg + ['W']] = np.nan

            for factor in factors_in_reg:
                x[factor] -= ((x[factor] * x.W).sum() / x.W.sum())

            x[factors_in_reg] = x[factors_in_reg].div(x[factors_in_reg].std(axis=0), axis=1)
            x[factors_in_reg] = x[factors_in_reg].clip(-3, 3)

            for factor in factors_in_reg:
                x[factor] -= ((x[factor] * x.W).sum() / x.W.sum())

            x[factors_in_reg] = x[factors_in_reg].div(x[factors_in_reg].std(axis=0), axis=1)
            x[factors_in_reg] = x[factors_in_reg].clip(-3, 3)

        x = x.dropna()
        y = x['current_return']
        w = x.W.fillna(0.0)
        x.drop(['current_return', 'W', 'quarantine_flag'], axis=1, inplace = True)
        x = sm.add_constant(x)
        if len(x) > 10:
#             res_model = sm.WLS(y, x, weights=1 / w ** 2)
            res_model = sm.WLS(y, x, weights= w**2)
            res = res_model.fit(cov_type='HC0')
            industry_returns = industry_returns[list(x.columns)]
            industry_returns.loc[t] = list(res.params)
            se.loc[t] = list(res.bse)
            resid_dict[t] = res.resid
            fitted_dict[t] = res.fittedvalues


            # Regression statistics and analysis
            vif_df = pd.DataFrame(columns=['value'])
            for i in range(len(x.columns)):
                vif_df.loc[x.columns[i]] = variance_inflation_factor(x, i)

            # P-values of F-test of linear restriction - ALL category returns are mutually 0
            if categories_in_reg:
                hypothesis_categories = ','.join('(' + pd.Series(categories_in_reg) + ' = 0)')
                f_test_categories = res.f_test(hypothesis_categories).pvalue
            else: f_test_categories = np.nan

            # P-values of F-test of linear restriction - ALL chains returns are mutually 0
            if chains_in_reg:
                hypothesis_chains = ','.join('(' + pd.Series(chains_in_reg) + ' = 0)')
                f_test_chains = res.f_test(hypothesis_chains).pvalue
            else: f_test_chains = np.nan

            # P-values of F-test of linear restriction - ALL style factor returns are mutually 0
            if factors_in_reg:
                hypothesis_style = ','.join('(' + pd.Series(factors_in_reg) + ' = 0)')
                f_test_factors = res.f_test(hypothesis_style).pvalue
            else: f_test_factors = np.nan

            # P-values of t-test of coefficients significance
            pvalue_dict[t] = res.pvalues

            # R-square

            r2 = res.rsquared
            r2_adj = res.rsquared_adj

            # Variables transformation
            x_tokens = list(x.index)
            b = beta[beta.date == t]
            b = b[b['gecko_id'].isin(x_tokens)]
            x['other_chains'] = list(b['Other_chain'])
            x['other_categories'] = list(b['Other_category'])
            for cat in categories_in_reg:
                x[cat] += list(x['other_categories'])
            for chain in chains_in_reg:
                x[chain] += list(x['other_chains'])
            x_dict[t] = x

            res_test.loc[t] = [r2, r2_adj, res.f_pvalue,
                               f_test_categories, f_test_chains,
                               f_test_factors, res.resid.std()]
            vif_values.loc[t] = vif_df.T.values[0]
        p_value_data = pd.DataFrame(pvalue_dict).T
    for date in tqdm(resid_dict.keys()):
        for token in resid_dict[date].index:
            resid_frame.loc[date][token] = resid_dict[date].loc[token]
            fitted_frame.loc[date][token] = fitted_dict[date].loc[token]

    resid_frame = resid_frame.dropna(how='all', axis=1)
    fitted_frame = fitted_frame.dropna(how='all', axis=1)
    industry_returns['other_chains'] = -industry_returns[chains_in_reg].sum(axis=1)
    industry_returns['other_categories'] = -industry_returns[categories_in_reg].sum(axis=1)
    
    for key in x_dict.keys():
        tmp_data = x_dict[key]
        tmp_data['date'] = np.array([key] * len(tmp_data))
        x_frames.append(tmp_data)
    x_frame = pd.concat(x_frames)
    
    geckoId_to_symbol = settings.ID_to_symbol(beta)
    geckoId_to_symbol = pd.DataFrame.from_dict(geckoId_to_symbol, orient='index')
    geckoId_to_symbol.rename(columns={0: "symbol"}, inplace = True)
    
    columns_to_frame = ['const', 'Bridge', 'Lending', 'Dexes', 'Derivatives', 'Yield', 'Avalanche', 'Polygon', 
                        'Binance', 'Ethereum', 'factor_log_mc', 'factor_log_mc_tvl', 'VMR', 'factor_momentum',
                        'Mbeta', 'other_chains', 'other_categories']
    
    x_frame = x_frame[columns_to_frame + ['date']]
    industry_returns = industry_returns[columns_to_frame]

    return industry_returns, x_frame, fitted_frame, resid_frame, p_value_data, res_test, vif_values, se, geckoId_to_symbol