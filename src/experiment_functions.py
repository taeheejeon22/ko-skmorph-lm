# Functions for stats
import copy
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests, MultiComparison
from scipy.stats import chisquare, chi2_contingency, kstest, norm, anderson, spearmanr, kendalltau, shapiro, wilcoxon, mannwhitneyu
# from pymer4.models import Lmer



# normality test
def do_kstest(data: pd.DataFrame, var_name: str, dtype=None, cdf=norm.cdf, log=False):
    if log==False:
        data[var_name] = np.log(pd.to_numeric(data[var_name]) + 1)
    else:
        data[var_name] = pd.to_numeric(data[var_name])
    data_without_nan = data[~data[var_name].isnull()]

    if dtype != None:
        data_without_nan[var_name] = data_without_nan[var_name].astype(dtype)

    statistic, p_value = kstest(data_without_nan[var_name], cdf)
    print(f"KS test statistic: {statistic:.4f}, p-value: {p_value:.4f}")
    return statistic, p_value

def do_SWtest(data: pd.DataFrame, var_name: str, dtype=None):
    data[var_name] = pd.to_numeric(data[var_name])
    data_without_nan = data[~data[var_name].isnull()]

    if dtype != None:
        data_without_nan[var_name] = data_without_nan[var_name].astype(dtype)

    result = shapiro(data_without_nan[var_name])
    # print the test statistic and critical values
    print(result)
    return result



# Wilcoxon signed-rank test
def do_wilcoxon_test(data_a, data_b):
    stat, p_value = wilcoxon(data_a, data_b)
    return stat, p_value


# Mann-Whitney U test (Independent (not-paired) Samples)
def do_MWU_test(data_a, data_b):
    stat, p = mannwhitneyu(data_a, data_b, alternative='two-sided')
    return stat, p


# # mixed effects model (pymer4)
# def do_Lmer(formula: str, DV: str, df: pd.DataFrame, family: str = "gaussian", factors: dict = None):
#     # factors = {"문장_성분": factors}
#     df_without_nan = df[~df[DV].isnull()]

#     model = Lmer(formula, data=df_without_nan, family=family)
#     model.fit(factors=factors)

#     return model


# mixed effects model (statsmodel)
def stars(p):
    if (p < 0.05) and (p >= 0.01):
        return "*"
    elif p < 0.01 and (p >= 0.001):
        return "**"
    elif p < 0.001:
        return "***"
    elif (p >= 0.05) and (p < 0.1): # marginally significant
        return "."
    else:
        return ""


def show_pvalues_for_statsmodel(mdf):
    print("\n".join([f"{mdf.params.index[i]} {stars(mdf.pvalues[i])}" for i in range(len(mdf.params))]))


def do_mixedlm(formula: str, DV: str, group_name: str, re_formula: str, df: pd.DataFrame, reml: bool=True, method=None):
    # formula
        # coding scheme: C(var, Sum)
        # set a reference level: "last_syl_dur ~ sex + C(문장_성분, Treatment(reference=-1))"
    # groups: random effects 열
    # reml: 모델 완성하고 싶으면 True. AIC, BIC 비교하고 싶으면 False.
    # method: optimization method (['bfgs', 'lbfgs', 'cg', 'newton']) # https://www.statsmodels.org/dev/_modules/statsmodels/regression/mixed_linear_model.html#MixedLM.fit

    df_without_nan = df[~df[DV].isnull()]
    groups = df_without_nan[group_name]

    md = smf.mixedlm(formula=formula, data=df_without_nan, groups=groups, re_formula=re_formula)
    mdf = md.fit(reml=reml, method=method)

    # print("\n".join([f"{mdf.params.index[i]} {stars(mdf.pvalues[i])}" for i in range(len(mdf.params))]))
    show_pvalues_for_statsmodel(mdf=mdf)

    if mdf.converged == False:
        print("\n####### Not converged!!! #######")

    return mdf


    # statsmodels로 mixed logistic regression
def do_GLM(formula: str, DV: str, group_name: str, df: pd.DataFrame, reml: bool=True):
    # formula
        # coding scheme: C(var, Sum)
        # set a reference level: "last_syl_dur ~ sex + C(문장_성분, Treatment(reference=-1))"
    # groups: random effects 열
    # reml: 모델 완성하고 싶으면 True. AIC, BIC 비교하고 싶으면 False.

    df_without_nan = df[~df[DV].isnull()]
    # groups = df_without_nan[group_name]

    md = sm.GLM.from_formula(formula=formula, data=df_without_nan, family=sm.families.Binomial(), groups=group_name)
    mdf = md.fit(reml=reml)

    show_pvalues_for_statsmodel(mdf=mdf)

    return mdf


    # statsmodel 결과 저장
regex_float = re.compile("\-?\d+\.\d+")
def write_statsmodel(model, path: str, mode="LMM"):
    if mode == "LMM":
        result_df = model.summary().tables[1].loc[:, ["Coef.", "Std.Err.", "z"]]
    elif mode == "GLM":
        result_df = model.summary2().tables[1].loc[:, ["Coef.", "Std.Err.", "z"]]

    # summary = summary_col(model, stars=True)
    #
    # result_df["p_value"] = [regex_float.sub("", summary.tables[0].iloc[i, 0]) for i in range(len(summary.tables[0])) if
    #                         i % 2 == 0]

    # pvalues = model.summary().tables[1].loc[:, ["P>|z|"]]
    # pvalues = pvalues[:-1].astype({"P>|z|": "float"})
    #
    # star_list = list()
    # for i in range(len(pvalues)):
    #     star_list.append(stars(pvalues.iloc[i, 0]))
    #
    # result_df["p-value"] = star_list + [""]


    pvalues = model.pvalues.to_list()
    result_df["p-value"] = [stars(x) for x in pvalues]

    # 자릿 수 맞추기
    # result_df = result_df.round(3)
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        result_df[col] = result_df[col].map('{:.3f}'.format)

    # 출력
    result_df.to_excel(path)


def write_statsmodel_v2(df_1: pd.DataFrame, df_2: pd.DataFrame, formula: str, DV: str, path, group_name="file", mode="LMM", method=None):
    # df_1: base dataframe
    # df_2: alternative dataframe
    # formula: regression model formula
    # DV: dependent variable
    # path: path to write the result
    # group_name: random effect
    # mode: LMM / GLM

    if mode == "LMM":
        if method == None:
            model_1 = do_mixedlm(formula=formula, DV=DV, group_name=group_name, df=df_1, reml=True)
            model_2 = do_mixedlm(formula=formula, DV=DV, group_name=group_name, df=df_2, reml=True)
        else:
            model_1 = do_mixedlm(formula=formula, DV=DV, group_name=group_name, df=df_1, reml=True, method=method)
            model_2 = do_mixedlm(formula=formula, DV=DV, group_name=group_name, df=df_2, reml=True, method=method)
        result_df_1 = model_1.summary().tables[1].loc[:, ["Coef.", "Std.Err.", "z"]]
        result_df_2 = model_2.summary().tables[1].loc[:, ["Coef.", "Std.Err.", "z"]]

        # If not converged, raise an error
        if (model_1.converged == False) or (model_1.converged == False):
            raise ValueError("not converged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    elif mode == "GLM":
        if method == None:
            model_1 = do_GLM(formula=formula, DV=DV, df=df_1, group_name=group_name)
            model_2 = do_GLM(formula=formula, DV=DV, df=df_2, group_name=group_name)
        else:
            model_1 = do_GLM(formula=formula, DV=DV, df=df_1, group_name=group_name, method=method)
            model_2 = do_GLM(formula=formula, DV=DV, df=df_2, group_name=group_name, method=method)
        result_df_1 = model_1.summary2().tables[1].loc[:, ["Coef.", "Std.Err.", "z"]]
        result_df_2 = model_2.summary2().tables[1].loc[:, ["Coef.", "Std.Err.", "z"]]

        # If not converged, raise an error
        if (model_1.converged == False) or (model_1.converged == False):
            raise ValueError("not converged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # pvalue 얻기
    pvalues_1 = model_1.pvalues.to_list()
    pvalues_2 = model_2.pvalues.to_list()

    result_df_1["p-value"] = [stars(x) for x in pvalues_1]
    result_df_2["p-value"] = [stars(x) for x in pvalues_2]

    # 자릿 수 맞추기
    numeric_cols_1 = result_df_1.select_dtypes(include=[np.number]).columns
    numeric_cols_2 = result_df_2.select_dtypes(include=[np.number]).columns
    # for col in numeric_cols_1:
    #     result_df_1[col] = result_df_1[col].map('{:.3f}'.format)

    for i in range(len(numeric_cols_1)):
        result_df_1[numeric_cols_1[i]] = result_df_1[numeric_cols_1[i]].map('{:.3f}'.format)
        result_df_2[numeric_cols_2[i]] = result_df_2[numeric_cols_2[i]].map('{:.3f}'.format)

    # 통합
    regex_main_variable = re.compile("C\(.+ Sum\)")

    indexes = result_df_1.index.to_list()
    last_row_index = [ix for ix in range(len(indexes)) if regex_main_variable.search(indexes[ix])][-1]

    new_row = result_df_2.iloc[last_row_index,]
    # new_row_index = new_row.name

    result_df_0_1 = copy.deepcopy(result_df_1.iloc[:last_row_index + 1])
    result_df_0_2 = copy.deepcopy(result_df_1.iloc[last_row_index + 1:])

    result_df = result_df_0_1.append(new_row, ignore_index=False)
    result_df_final = pd.concat([result_df, result_df_0_2], axis=0)

    # 출력
    result_df_final.to_excel(path)