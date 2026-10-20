"""Question templates + ground-truth functions.

Each template is a (question-text, ground-truth-function, answer-type) triple
plus 4 paraphrases that keep the SAME meaning (so they must map to the SAME
ground-truth value). The ground-truth function is a deterministic pandas
computation — never the LLM — and is the reference the agent's answer is scored
against (report Eq. 1).

Coverage required by the report section 3.3:
    aggregation · filtering · group comparison · trend · multi-step

A template:
    id             stable id, e.g. "titanic.q03"
    kind           one of the five question types (a static feature later)
    text           canonical phrasing
    paraphrases    4 alternative phrasings, same meaning
    answer_type    "numeric" | "categorical"
    truth          callable(tables: dict[str, DataFrame]) -> value
    n_tables       how many tables the answer needs (static feature)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

Tables = dict[str, pd.DataFrame]


@dataclass
class Template:
    id: str
    dataset: str
    kind: str          # aggregation | filtering | group_comparison | trend | multi_step
    text: str
    paraphrases: list[str]
    answer_type: str   # numeric | categorical
    truth: Callable[[Tables], object]
    n_tables: int = 1


def _iqr_filter(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return s[(s >= lo) & (s <= hi)]


# =========================================================================== #
# TITANIC — clean single table
# =========================================================================== #
def _titanic_templates() -> list[Template]:
    def col(t, *names):
        """Titanic mirrors vary in casing; pick the first present column."""
        df = t["main"]
        for n in names:
            if n in df.columns:
                return df[n]
        raise KeyError(names)

    def survived(t):
        return col(t, "Survived", "survived")

    def pclass(t):
        return col(t, "Pclass", "Class", "pclass")

    def fare(t):
        return col(t, "Fare", "fare")

    def age(t):
        return col(t, "Age", "age")

    def sex(t):
        return col(t, "Sex", "sex")

    T = []
    T.append(Template(
        "titanic.q01", "titanic", "aggregation",
        "What is the overall survival rate of passengers?",
        ["What fraction of passengers survived?",
         "What proportion of all passengers lived?",
         "Across everyone on board, what share survived?",
         "What is the mean of the survival indicator over all passengers?"],
        "numeric", lambda t: round(float(survived(t).mean()), 6)))
    T.append(Template(
        "titanic.q02", "titanic", "aggregation",
        "How many passengers are in the dataset?",
        ["What is the total passenger count?",
         "How many rows of passengers are there?",
         "Count the passengers in the table.",
         "What is the number of passenger records?"],
        "numeric", lambda t: int(len(t["main"]))))
    T.append(Template(
        "titanic.q03", "titanic", "filtering",
        "How many passengers were in 1st class (Pclass 1)?",
        ["Count the first-class passengers.",
         "How many travellers held a class-1 ticket?",
         "What is the number of passengers whose class is 1?",
         "Among all passengers, how many were first class?"],
        "numeric", lambda t: int((pclass(t) == 1).sum())))
    T.append(Template(
        "titanic.q04", "titanic", "group_comparison",
        "Which passenger class had the highest survival rate?",
        ["In which ticket class did the largest share survive?",
         "Which class of passenger was most likely to survive?",
         "By class, which had the greatest survival proportion?",
         "Which Pclass shows the top survival rate?"],
        "categorical",
        lambda t: int(t["main"].assign(_s=survived(t), _c=pclass(t))
                      .groupby("_c")["_s"].mean().idxmax())))
    T.append(Template(
        "titanic.q05", "titanic", "group_comparison",
        "What is the difference in survival rate between female and male passengers?",
        ["By how much does female survival exceed male survival?",
         "What is the gap in survival proportion between women and men?",
         "Female minus male survival rate — what is it?",
         "How much higher was the survival share for females than males?"],
        "numeric",
        lambda t: round(float(
            t["main"].assign(_s=survived(t), _x=sex(t)).groupby("_x")["_s"].mean()
            .reindex(["female", "male"]).diff().iloc[-1] * -1), 6)))
    T.append(Template(
        "titanic.q06", "titanic", "aggregation",
        "What is the mean fare after removing fare outliers using the 1.5*IQR rule?",
        ["Average the fares once 1.5-IQR outliers are dropped.",
         "What is the trimmed mean fare under the standard IQR outlier rule?",
         "Remove fare outliers (1.5 IQR) and report the mean fare.",
         "After IQR outlier removal, what is the average ticket fare?"],
        "numeric", lambda t: round(float(_iqr_filter(fare(t).dropna()).mean()), 6)))
    T.append(Template(
        "titanic.q07", "titanic", "aggregation",
        "What is the median age of passengers (ignoring missing ages)?",
        ["What is the median passenger age, excluding unknown ages?",
         "Among passengers with a recorded age, what is the median?",
         "Report the median age over non-missing age values.",
         "What age is the 50th percentile of known ages?"],
        "numeric", lambda t: round(float(age(t).dropna().median()), 6)))
    T.append(Template(
        "titanic.q08", "titanic", "group_comparison",
        "What is the mean fare for 1st-class passengers after removing fare outliers "
        "with the 1.5*IQR rule?",
        ["For first-class only, average the fares after 1.5-IQR outlier removal.",
         "Mean class-1 fare, IQR outliers excluded — what is it?",
         "Among 1st-class passengers, what is the trimmed-mean fare (1.5 IQR)?",
         "Drop fare outliers (1.5 IQR), then report the first-class mean fare."],
        "numeric",
        lambda t: round(float(_iqr_filter(
            fare(t)[pclass(t) == 1].dropna()).mean()), 6)))
    T.append(Template(
        "titanic.q09", "titanic", "filtering",
        "How many female passengers survived?",
        ["Count the women who survived.",
         "How many surviving passengers were female?",
         "What is the number of female survivors?",
         "Among survivors, how many were women?"],
        "numeric", lambda t: int(((sex(t) == "female") & (survived(t) == 1)).sum())))
    T.append(Template(
        "titanic.q10", "titanic", "multi_step",
        "Among passengers older than 30 (known age), what is the survival rate "
        "of 1st-class passengers?",
        ["For over-30s with a known age, what share of first-class passengers survived?",
         "Restrict to known-age passengers above 30; among those in class 1, "
         "what is the survival rate?",
         "Survival proportion of 1st-class, over-30, known-age passengers?",
         "Take passengers aged over 30 with recorded age, then compute class-1 "
         "survival rate."],
        "numeric",
        lambda t: round(float(
            t["main"].assign(_s=survived(t), _c=pclass(t), _a=age(t))
            .query("_a > 30 and _c == 1")["_s"].mean()), 6)))
    return T


# =========================================================================== #
# ADULT CENSUS — clean single table, categoricals
# =========================================================================== #
def _adult_templates() -> list[Template]:
    m = lambda t: t["main"]  # noqa: E731
    T = []
    T.append(Template(
        "adult.q01", "adult_census", "aggregation",
        "How many individuals are in the dataset?",
        ["What is the total record count?",
         "How many rows are there?",
         "Count the individuals in the table.",
         "What is the number of people recorded?"],
        "numeric", lambda t: int(len(m(t)))))
    T.append(Template(
        "adult.q02", "adult_census", "aggregation",
        "What is the mean age across all individuals?",
        ["What is the average age?",
         "Report the mean of the age column.",
         "On average, how old are the individuals?",
         "What is the arithmetic mean age?"],
        "numeric", lambda t: round(float(m(t)["age"].mean()), 6)))
    T.append(Template(
        "adult.q03", "adult_census", "filtering",
        "How many individuals earn more than 50K (income '>50K')?",
        ["Count the people with income above 50K.",
         "How many records have the high-income label?",
         "What is the number of individuals in the >50K bracket?",
         "How many earn over 50K per year?"],
        "numeric",
        lambda t: int(m(t)["income"].astype(str).str.contains(">50K").sum())))
    T.append(Template(
        "adult.q04", "adult_census", "aggregation",
        "What is the mean weekly hours worked?",
        ["What is the average of hours_per_week?",
         "On average, how many hours are worked per week?",
         "Report the mean weekly working hours.",
         "What is the mean hours-per-week value?"],
        "numeric", lambda t: round(float(m(t)["hours_per_week"].mean()), 6)))
    T.append(Template(
        "adult.q05", "adult_census", "group_comparison",
        "Which education level has the highest share of >50K earners?",
        ["For which education category is the high-income proportion greatest?",
         "Which schooling level most often earns above 50K?",
         "By education, which has the top >50K rate?",
         "Which education group has the largest fraction earning >50K?"],
        "categorical",
        lambda t: str(m(t).assign(
            _hi=m(t)["income"].astype(str).str.contains(">50K"))
            .groupby("education")["_hi"].mean().idxmax())))
    T.append(Template(
        "adult.q06", "adult_census", "group_comparison",
        "What is the difference in mean age between the >50K and <=50K groups?",
        ["How much older on average is the high-income group than the low-income one?",
         "Mean age of >50K minus mean age of <=50K — what is it?",
         "By income bracket, what is the gap in average age?",
         "What is the age difference between the two income groups, high minus low?"],
        "numeric",
        lambda t: round(float(
            m(t).assign(_hi=m(t)["income"].astype(str).str.contains(">50K"))
            .groupby("_hi")["age"].mean().reindex([True, False]).diff().iloc[-1] * -1
        ), 6)))
    T.append(Template(
        "adult.q07", "adult_census", "filtering",
        "How many individuals have more than 12 years of education (education_num > 12)?",
        ["Count records with education_num above 12.",
         "How many people have education_num greater than 12?",
         "What is the number of individuals whose education-num exceeds 12?",
         "Among all, how many have over 12 education-years?"],
        "numeric", lambda t: int((m(t)["education_num"] > 12).sum())))
    T.append(Template(
        "adult.q08", "adult_census", "aggregation",
        "What is the most common occupation?",
        ["Which occupation appears most frequently?",
         "What is the modal occupation?",
         "Which job category is the most represented?",
         "What occupation has the highest count?"],
        "categorical", lambda t: str(m(t)["occupation"].mode(dropna=True).iloc[0])))
    T.append(Template(
        "adult.q09", "adult_census", "aggregation",
        "What is the mean weekly hours after removing hours outliers with the "
        "1.5*IQR rule?",
        ["Average hours_per_week once 1.5-IQR outliers are dropped.",
         "Trimmed-mean weekly hours under the standard IQR rule?",
         "Remove hours outliers (1.5 IQR) and report mean weekly hours.",
         "After IQR outlier removal, what is the average weekly hours?"],
        "numeric", lambda t: round(float(
            _iqr_filter(m(t)["hours_per_week"].dropna()).mean()), 6)))
    T.append(Template(
        "adult.q10", "adult_census", "multi_step",
        "Among individuals working more than 40 hours per week, what share earn >50K?",
        ["For people working over 40 weekly hours, what fraction are high-income?",
         "Restrict to >40 hours/week; what is the >50K rate there?",
         "High-income proportion among those working more than 40 hours weekly?",
         "Take individuals above 40 weekly hours, then compute the >50K share."],
        "numeric",
        lambda t: round(float(
            m(t)[m(t)["hours_per_week"] > 40]["income"].astype(str)
            .str.contains(">50K").mean()), 6)))
    return T


# =========================================================================== #
# NYC TAXI — larger, outliers, mixed types
# =========================================================================== #
def _nyc_templates() -> list[Template]:
    m = lambda t: t["main"]  # noqa: E731
    T = []
    T.append(Template(
        "nyc.q01", "nyc_taxi", "aggregation",
        "What is the mean fare amount (ignoring missing fares)?",
        ["What is the average fare_amount over non-missing rows?",
         "Report the mean fare, excluding missing values.",
         "On average, what is the fare amount?",
         "What is the mean of recorded fare amounts?"],
        "numeric", lambda t: round(float(m(t)["fare_amount"].dropna().mean()), 6)))
    T.append(Template(
        "nyc.q02", "nyc_taxi", "aggregation",
        "What is the total number of trips?",
        ["How many taxi trips are recorded?",
         "Count the trips.",
         "What is the trip count?",
         "How many rows of trips are there?"],
        "numeric", lambda t: int(len(m(t)))))
    T.append(Template(
        "nyc.q03", "nyc_taxi", "filtering",
        "How many trips had more than 2 passengers?",
        ["Count trips with passenger count above 2.",
         "How many rides carried more than two passengers?",
         "What is the number of trips where passengers > 2?",
         "Among trips, how many had over 2 passengers?"],
        "numeric", lambda t: int((m(t)["passengers"] > 2).sum())))
    T.append(Template(
        "nyc.q04", "nyc_taxi", "aggregation",
        "What is the mean fare amount after removing fare outliers with the "
        "1.5*IQR rule?",
        ["Average fare_amount once 1.5-IQR outliers are dropped.",
         "Trimmed-mean fare under the standard IQR outlier rule?",
         "Remove fare outliers (1.5 IQR) and report the mean fare.",
         "After IQR outlier removal, what is the average fare?"],
        "numeric", lambda t: round(float(
            _iqr_filter(m(t)["fare_amount"].dropna()).mean()), 6)))
    T.append(Template(
        "nyc.q05", "nyc_taxi", "group_comparison",
        "During which pickup hour is the mean fare highest?",
        ["Which hour of day has the greatest average fare?",
         "By pickup hour, when is mean fare at its maximum?",
         "At what pickup_hour is the average fare largest?",
         "Which hour shows the top mean fare?"],
        "numeric", lambda t: int(
            m(t).groupby("pickup_hour")["fare_amount"].mean().idxmax())))
    T.append(Template(
        "nyc.q06", "nyc_taxi", "aggregation",
        "What is the median trip distance?",
        ["What is the 50th percentile of trip_distance?",
         "Report the median distance travelled.",
         "What distance is the median trip?",
         "What is the middle value of trip distances?"],
        "numeric", lambda t: round(float(m(t)["trip_distance"].median()), 6)))
    T.append(Template(
        "nyc.q07", "nyc_taxi", "group_comparison",
        "What is the mean tip amount for card payments (payment_type normalised, "
        "case-insensitive 'card')?",
        ["Average tip on card-paid trips, matching payment type case-insensitively.",
         "For card payments (any casing), what is the mean tip?",
         "Mean tip_amount where the payment type is card, ignoring case?",
         "What is the average tip when paying by card (case-insensitive)?"],
        "numeric", lambda t: round(float(
            m(t)[m(t)["payment_type"].astype(str).str.lower() == "card"]
            ["tip_amount"].mean()), 6)))
    T.append(Template(
        "nyc.q08", "nyc_taxi", "filtering",
        "How many trips have a non-missing fare amount?",
        ["Count trips with a recorded fare.",
         "How many rows have a present fare_amount?",
         "What is the number of trips whose fare is not missing?",
         "How many trips are not missing the fare value?"],
        "numeric", lambda t: int(m(t)["fare_amount"].notna().sum())))
    T.append(Template(
        "nyc.q09", "nyc_taxi", "aggregation",
        "What is the mean number of passengers per trip?",
        ["What is the average passenger count?",
         "On average, how many passengers per trip?",
         "Report the mean passengers value.",
         "What is the mean of the passengers column?"],
        "numeric", lambda t: round(float(m(t)["passengers"].mean()), 6)))
    T.append(Template(
        "nyc.q10", "nyc_taxi", "multi_step",
        "For trips longer than 3 miles, what is the mean fare after removing fare "
        "outliers with the 1.5*IQR rule (outliers computed on the filtered subset)?",
        ["Among trips over 3 miles, trimmed-mean fare (1.5 IQR on that subset)?",
         "Filter to >3 mile trips, drop 1.5-IQR fare outliers, then average the fare.",
         "Mean fare of long (>3 mi) trips after IQR outlier removal within that group?",
         "Take trips beyond 3 miles, remove fare outliers (1.5 IQR), report mean fare."],
        "numeric", lambda t: round(float(_iqr_filter(
            m(t)[m(t)["trip_distance"] > 3]["fare_amount"].dropna()).mean()), 6)))
    return T


# =========================================================================== #
# MISSINGNESS — heavy missing values
# =========================================================================== #
def _missingness_templates() -> list[Template]:
    m = lambda t: t["main"]  # noqa: E731
    T = []
    T.append(Template(
        "miss.q01", "missingness", "aggregation",
        "What fraction of 'income' values are missing?",
        ["What share of the income column is null?",
         "What proportion of income entries are missing?",
         "How much of 'income' is absent, as a fraction?",
         "What is the missing rate of the income field?"],
        "numeric", lambda t: round(float(m(t)["income"].isna().mean()), 6)))
    T.append(Template(
        "miss.q02", "missingness", "aggregation",
        "What is the mean age over non-missing ages?",
        ["Average the recorded ages, ignoring missing ones.",
         "What is the mean of present age values?",
         "Among rows with a known age, what is the mean age?",
         "Report mean age excluding nulls."],
        "numeric", lambda t: round(float(m(t)["age"].dropna().mean()), 6)))
    T.append(Template(
        "miss.q03", "missingness", "filtering",
        "How many rows have a missing 'region'?",
        ["Count rows where region is null.",
         "How many records lack a region value?",
         "What is the number of rows with absent region?",
         "How many entries are missing the region field?"],
        "numeric", lambda t: int(m(t)["region"].isna().sum())))
    T.append(Template(
        "miss.q04", "missingness", "aggregation",
        "How many rows are complete (no missing values in any column)?",
        ["Count the fully populated rows.",
         "How many records have no nulls at all?",
         "What is the number of complete-case rows?",
         "How many rows have every field present?"],
        "numeric", lambda t: int(m(t).dropna().shape[0])))
    T.append(Template(
        "miss.q05", "missingness", "group_comparison",
        "Which region has the highest mean income (over non-missing income)?",
        ["For which region is average recorded income greatest?",
         "By region, which has the top mean income?",
         "Which region shows the largest average income among present values?",
         "Which region's mean income (non-missing) is highest?"],
        "categorical", lambda t: str(
            m(t).dropna(subset=["income", "region"])
            .groupby("region")["income"].mean().idxmax())))
    T.append(Template(
        "miss.q06", "missingness", "aggregation",
        "What is the mean churn rate (mean of the 'churned' column)?",
        ["What fraction of rows are churned?",
         "What is the average of the churned indicator?",
         "What share of records churned?",
         "Report the mean churned value."],
        "numeric", lambda t: round(float(m(t)["churned"].mean()), 6)))
    T.append(Template(
        "miss.q07", "missingness", "filtering",
        "How many rows have a known income greater than 50000?",
        ["Count rows whose recorded income exceeds 50000.",
         "How many present-income rows are above 50000?",
         "What is the number of rows with income (non-missing) over 50000?",
         "Among rows with income, how many exceed 50000?"],
        "numeric", lambda t: int((m(t)["income"].dropna() > 50000).sum())))
    T.append(Template(
        "miss.q08", "missingness", "aggregation",
        "What is the median tenure in months?",
        ["What is the 50th percentile of tenure_months?",
         "Report the median tenure.",
         "What tenure is the median?",
         "What is the middle value of tenure_months?"],
        "numeric", lambda t: round(float(m(t)["tenure_months"].median()), 6)))
    T.append(Template(
        "miss.q09", "missingness", "group_comparison",
        "What is the difference in churn rate between rows with missing income and "
        "rows with present income (missing minus present)?",
        ["How much does churn differ when income is missing vs present (missing-present)?",
         "Churn rate for missing-income rows minus that for present-income rows?",
         "By income-missingness, what is the churn gap (missing minus present)?",
         "Difference in mean churn between null-income and non-null-income rows?"],
        "numeric", lambda t: round(float(
            m(t).assign(_miss=m(t)["income"].isna())
            .groupby("_miss")["churned"].mean().reindex([True, False]).diff().iloc[-1]
            * -1), 6)))
    T.append(Template(
        "miss.q10", "missingness", "multi_step",
        "Among rows with both age and income present, what is the mean income for "
        "those older than 40?",
        ["For complete age+income rows over age 40, what is mean income?",
         "Restrict to rows with age and income; among age>40, average income?",
         "Mean income of over-40s having both age and income recorded?",
         "Take rows where age and income are present and age>40, report mean income."],
        "numeric", lambda t: round(float(
            m(t).dropna(subset=["age", "income"]).query("age > 40")["income"].mean()
        ), 6)))
    return T


# =========================================================================== #
# OLIST — multi-table, needs joins
# =========================================================================== #
def _olist_templates() -> list[Template]:
    def joined(t):
        oi = t["order_items"].merge(t["orders"], on="order_id", how="left")
        return oi.merge(t["customers"], on="customer_id", how="left")

    T = []
    T.append(Template(
        "olist.q01", "olist", "aggregation",
        "How many orders are there in total?",
        ["What is the total order count?",
         "Count the orders.",
         "How many rows are in the orders table?",
         "What is the number of distinct orders?"],
        "numeric", lambda t: int(len(t["orders"])), n_tables=1))
    T.append(Template(
        "olist.q02", "olist", "aggregation",
        "What is the mean item price across all order items?",
        ["What is the average price in order_items?",
         "Report the mean item price.",
         "On average, what does an order item cost?",
         "What is the mean of the price column in order items?"],
        "numeric", lambda t: round(float(t["order_items"]["price"].mean()), 6),
        n_tables=1))
    T.append(Template(
        "olist.q03", "olist", "filtering",
        "How many orders have status 'delivered'?",
        ["Count the delivered orders.",
         "How many orders were delivered?",
         "What is the number of orders with delivered status?",
         "Among orders, how many are delivered?"],
        "numeric", lambda t: int((t["orders"]["status"] == "delivered").sum()),
        n_tables=1))
    T.append(Template(
        "olist.q04", "olist", "group_comparison",
        "Which product category has the highest total revenue (sum of price)?",
        ["For which category is summed price greatest?",
         "Which product category earns the most revenue overall?",
         "By category, which has the top total price?",
         "Which category's total item price is highest?"],
        "categorical", lambda t: str(
            t["order_items"].groupby("product_category")["price"].sum().idxmax()),
        n_tables=1))
    T.append(Template(
        "olist.q05", "olist", "multi_step",
        "What is the total item revenue (sum of price) for orders placed by customers "
        "in state 'SP'?",
        ["Summed item price for orders from SP-state customers?",
         "For customers in SP, what is the total order-item price?",
         "Join items to orders to customers; total price where state is SP?",
         "What is the SP-state customers' total item revenue?"],
        "categorical",  # large number; compared as exact int-ish -> treat categorical
        lambda t: round(float(
            _olist_join(t).query("state == 'SP'")["price"].sum()), 2),
        n_tables=3))
    T.append(Template(
        "olist.q06", "olist", "group_comparison",
        "Which customer state has the most orders?",
        ["Which state places the greatest number of orders?",
         "By state, which has the highest order count?",
         "What state accounts for the most orders?",
         "Which state has the top order volume?"],
        "categorical", lambda t: str(
            t["orders"].merge(t["customers"], on="customer_id", how="left")
            ["state"].value_counts().idxmax()),
        n_tables=2))
    T.append(Template(
        "olist.q07", "olist", "aggregation",
        "What is the mean freight value across all order items?",
        ["What is the average freight?",
         "Report the mean of the freight column.",
         "On average, what is the freight charge per item?",
         "What is the mean freight value?"],
        "numeric", lambda t: round(float(t["order_items"]["freight"].mean()), 6),
        n_tables=1))
    T.append(Template(
        "olist.q08", "olist", "filtering",
        "How many order items belong to the 'electronics' category?",
        ["Count the electronics order items.",
         "How many items are in the electronics category?",
         "What is the number of electronics-category items?",
         "Among order items, how many are electronics?"],
        "numeric", lambda t: int(
            (t["order_items"]["product_category"] == "electronics").sum()),
        n_tables=1))
    T.append(Template(
        "olist.q09", "olist", "trend",
        "In which order month is the total item revenue highest?",
        ["Which month has the greatest summed item price?",
         "By order month, when is total price at its maximum?",
         "What order_month shows the top revenue?",
         "Which month earns the most item revenue?"],
        "numeric", lambda t: int(
            t["order_items"].merge(t["orders"], on="order_id", how="left")
            .groupby("order_month")["price"].sum().idxmax()),
        n_tables=2))
    T.append(Template(
        "olist.q10", "olist", "multi_step",
        "Among delivered orders only, what is the mean item price for the "
        "'electronics' category?",
        ["For delivered orders, average electronics item price?",
         "Mean price of electronics items on delivered orders?",
         "Restrict to delivered orders; what is mean electronics price?",
         "Take delivered orders and electronics items, report mean price."],
        "numeric", lambda t: round(float(
            t["order_items"].merge(t["orders"], on="order_id", how="left")
            .query("status == 'delivered' and product_category == 'electronics'")
            ["price"].mean()), 6),
        n_tables=2))
    return T


def _olist_join(t):
    oi = t["order_items"].merge(t["orders"], on="order_id", how="left")
    return oi.merge(t["customers"], on="customer_id", how="left")


_BUILDERS = {
    "titanic": _titanic_templates,
    "adult_census": _adult_templates,
    "nyc_taxi": _nyc_templates,
    "missingness": _missingness_templates,
    "olist": _olist_templates,
}


def templates_for(dataset: str) -> list[Template]:
    return _BUILDERS[dataset]()


def all_templates() -> list[Template]:
    out: list[Template] = []
    for ds in _BUILDERS:
        out.extend(_BUILDERS[ds]())
    return out


def expand_questions(dataset: str) -> list[dict]:
    """Expand each template into 1 canonical + 4 paraphrase question instances.

    Returns one record per (template, phrasing) with a stable question_id. The
    ground-truth VALUE is shared across all phrasings of a template (that is the
    whole point of paraphrase-consistency labelling)."""
    records = []
    for tpl in templates_for(dataset):
        phrasings = [tpl.text] + tpl.paraphrases
        for p_idx, text in enumerate(phrasings):
            records.append({
                "question_id": f"{tpl.id}.p{p_idx}",
                "template_id": tpl.id,
                "dataset": tpl.dataset,
                "kind": tpl.kind,
                "answer_type": tpl.answer_type,
                "n_tables": tpl.n_tables,
                "paraphrase_index": p_idx,
                "is_canonical": p_idx == 0,
                "text": text,
            })
    return records
