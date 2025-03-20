import math
import streamlit as st
import pandas as pd
import numpy as np

class PricingModel:
    def __init__(self, base_cost, margin, volume_discount, risk_factors, cashflow_model, customer_value):
        self.base_cost = base_cost
        self.margin = margin
        self.volume_discount = volume_discount
        self.risk_factors = risk_factors
        self.cashflow_model = cashflow_model
        self.customer_value = customer_value

    def get_volume_discount(self, order_quantity):
        """Determine volume discount based on thresholds."""
        if order_quantity >= 400:
            return self.volume_discount.get(400, 0)
        elif order_quantity >= 300:
            return self.volume_discount.get(300, 0)
        elif order_quantity >= 200:
            return self.volume_discount.get(200, 0)
        else:
            return 0

    def cost_plus_pricing(self, order_quantity):
        """Computes the cost-plus price:
           Price = COGS / (1 - margin) with an applied volume discount (if any)."""
        raw_price = self.base_cost / (1 - self.margin)
        vol_disc = self.get_volume_discount(order_quantity)
        return raw_price * (1 - vol_disc)

    def tiered_pricing(self, order_quantity):
        """For tiered pricing we start from the raw cost-plus base, then apply:
             - the volume discount (if set) and
             - an extra discount if order_quantity thresholds are met.
           If the volume discount is 0, no extra discount is applied."""
        raw_price = self.base_cost / (1 - self.margin)
        vol_disc = self.get_volume_discount(order_quantity)
        price_after_vol = raw_price * (1 - vol_disc)
        if vol_disc == 0:
            return raw_price
        else:
            if order_quantity >= 400:
                return price_after_vol * 0.9
            elif order_quantity >= 300:
                return price_after_vol * 0.95
            else:
                return price_after_vol

    def value_based_pricing(self, order_quantity):
        """Value-based pricing model using the customer perceived value."""
        return self.customer_value

    def evaluate_deal(self, order_quantity):
        """
        Evaluates pricing models and computes gross profits.
        Returns:
            raw_results: Prices after applying risk adjustments.
            final_results: Prices after applying cash flow adjustments.
            gross_profits: Gross profit calculated as (Final Price - COGS) * order_quantity.
            best_option: The best pricing model (lowest final price).
        """
        cost_plus = self.cost_plus_pricing(order_quantity)
        tiered = self.tiered_pricing(order_quantity)
        value_based = self.value_based_pricing(order_quantity)

        # Apply risk factor adjustments
        total_risk_factor = sum(self.risk_factors.values()) / 100  # converting percentage to multiplier
        raw_results = {
            "Cost-Plus Pricing": cost_plus * (1 + total_risk_factor),
            "Tiered Pricing": tiered * (1 + total_risk_factor),
            "Value-Based Pricing": value_based * (1 + total_risk_factor)
        }

        # Save raw results before applying cash flow adjustments
        final_results = raw_results.copy()

        # Apply cash flow model impact
        if self.cashflow_model == "upfront":
            final_results = {k: v * 0.95 for k, v in final_results.items()}
        elif self.cashflow_model == "milestone":
            final_results = {k: v * 1.02 for k, v in final_results.items()}
        elif self.cashflow_model == "delayed":
            final_results = {k: v * 1.05 for k, v in final_results.items()}

        # Compute Gross Profit based on final prices:
        gross_profits = {k: (v - self.base_cost) * order_quantity for k, v in final_results.items()}

        # Select best pricing model based on the lowest final price (per unit)
        best_option = min(final_results, key=final_results.get)
        return raw_results, final_results, gross_profits, best_option

# Sidebar: User Inputs

st.sidebar.header("User Inputs")

# Pricing Model Inputs
base_cost = st.sidebar.number_input("COGS per Unit (€)", value=1179)
margin = st.sidebar.number_input("Profit Margin (%)", min_value=0, max_value=100, value=44) / 100
customer_value = st.sidebar.number_input("Customer Perceived Value (€)", value=2500)

# Order Quantity Input
order_quantity = st.sidebar.number_input("Order Quantity", value=300)

# Volume Discount Inputs as User Defined
st.sidebar.header("Volume Discount Inputs")
discount_200 = st.sidebar.number_input("Discount at Order Quantity 200 (%)", min_value=0, max_value=100, value=0) / 100
discount_300 = st.sidebar.number_input("Discount at Order Quantity 300 (%)", min_value=0, max_value=100, value=0) / 100
discount_400 = st.sidebar.number_input("Discount at Order Quantity 400 (%)", min_value=0, max_value=100, value=0) / 100
volume_discount = {200: discount_200, 300: discount_300, 400: discount_400}

# Risk Factor Inputs
st.sidebar.subheader("Risk Factors (as % Impact)")
supply_chain_risk = st.sidebar.number_input("Supply Chain Risk (%)", min_value=0, max_value=10, value=5)
regulatory_risk = st.sidebar.number_input("Regulatory Compliance Risk (%)", min_value=0, max_value=10, value=3)
payment_risk = st.sidebar.number_input("Payment Delay Risk (%)", min_value=0, max_value=10, value=4)
competition_risk = st.sidebar.number_input("Competitive Market Pressure (%)", min_value=0, max_value=10, value=2)

risk_factors = {
    "Supply Chain Risk": supply_chain_risk,
    "Regulatory Risk": regulatory_risk,
    "Payment Risk": payment_risk,
    "Competition Risk": competition_risk,
}

# Cash Flow Options
st.sidebar.subheader("Cash Flow Management Strategy")
cashflow_model = st.sidebar.selectbox("Select Payment Structure", ["upfront", "milestone", "delayed"])

# Supply Chain Inputs for EOQ Calculation
st.sidebar.header("Supply Chain Inputs")
annual_demand = st.sidebar.number_input("Annual Demand (units)", value=10000)
ordering_cost = st.sidebar.number_input("Ordering Cost per Order (€)", value=50)
holding_cost = st.sidebar.number_input("Inventory Holding Cost per Unit (€)", value=2)

# Compute EOQ
if holding_cost > 0:
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
else:
    eoq = 0
st.sidebar.write(f"Calculated EOQ: {eoq:.2f} units")

# Create Tabs for Results, Sensitivity Analysis, and Supply Chain
tabs = st.tabs(["Results", "Sensitivity Analysis", "Supply Chain"])

# Instantiate Pricing Model
pricing_model = PricingModel(base_cost, margin, volume_discount, risk_factors, cashflow_model, customer_value)

with tabs[0]:
    st.title("ERD Pricing Model Evaluation - Results")
    raw_results, final_results, gross_profits, best_pricing_option = pricing_model.evaluate_deal(order_quantity)
    # Create a summary dataframe
    df = pd.DataFrame({
        "Raw Price per Unit (€)": raw_results,
        "Final Price per Unit (€)": final_results,
        "Gross Profit (€)": gross_profits
    })
    st.dataframe(df)
    st.success(f"Best Pricing Model: {best_pricing_option}")

with tabs[1]:
    st.title("Sensitivity Analysis")
    st.write("Analysis over a range of Order Quantities")
    quantities = np.arange(100, 501, 50)
    raw_cp = []
    final_cp = []
    value_based_prices = []
    best_option = []
    
    for qty in quantities:
        raw, final, _, best = pricing_model.evaluate_deal(qty)
        raw_cp.append(raw["Cost-Plus Pricing"])
        final_cp.append(final["Cost-Plus Pricing"])
        value_based_prices.append(final["Value-Based Pricing"])
        best_option.append(best)
    
    sa_df = pd.DataFrame({
        "Order Quantity": quantities,
        "Raw Cost-Plus Price (€)": raw_cp,
        "Final Cost-Plus Price (€)": final_cp,
        "Value-Based Pricing (€)": value_based_prices,
        "Best Option": best_option
    })
    
    st.dataframe(sa_df)
    st.line_chart(sa_df.set_index("Order Quantity")[["Raw Cost-Plus Price (€)", "Final Cost-Plus Price (€)", "Value-Based Pricing (€)"]])

with tabs[2]:
    st.title("Supply Chain EOQ Calculation")
    st.write("Supply Chain inputs and EOQ result are provided in the sidebar.")
    st.write(f"Calculated Economic Order Quantity (EOQ): {eoq:.2f} units")
