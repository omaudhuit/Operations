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

    def cost_plus_pricing(self, order_quantity):
        """Cost-plus pricing model with margin consideration.
           Selling price is computed as COGS / (1 - margin) and a volume discount is applied."""
        base_price = self.base_cost / (1 - self.margin)
        return base_price * (1 - self.volume_discount.get(order_quantity, 0))

    def tiered_pricing(self, order_quantity):
        """Tiered pricing based on volume."""
        base_price = self.cost_plus_pricing(order_quantity)
        if order_quantity >= 400:
            return base_price * 0.9
        elif order_quantity >= 300:
            return base_price * 0.95
        else:
            return base_price

    def value_based_pricing(self, order_quantity):
        """Value-based pricing model using the customer perceived value."""
        return self.customer_value

    def evaluate_deal(self, order_quantity):
        """Evaluates pricing models, applies adjustments, and computes gross profits.
           Gross Profit is computed as (Adjusted Price per Unit - COGS) * Order Quantity."""
        cost_plus = self.cost_plus_pricing(order_quantity)
        tiered = self.tiered_pricing(order_quantity)
        value_based = self.value_based_pricing(order_quantity)

        # Apply risk factor adjustments
        total_risk_factor = sum(self.risk_factors.values()) / 100  # converting percentage to multiplier
        adjusted_results = {
            "Cost-Plus Pricing": cost_plus * (1 + total_risk_factor),
            "Tiered Pricing": tiered * (1 + total_risk_factor),
            "Value-Based Pricing": value_based * (1 + total_risk_factor)
        }

        # Apply cash flow model impact
        if self.cashflow_model == "upfront":
            adjusted_results = {k: v * 0.95 for k, v in adjusted_results.items()}  # 5% discount
        elif self.cashflow_model == "milestone":
            adjusted_results = {k: v * 1.02 for k, v in adjusted_results.items()}  # 2% increase
        elif self.cashflow_model == "delayed":
            adjusted_results = {k: v * 1.05 for k, v in adjusted_results.items()}  # 5% increase

        # Compute Gross Profit for each pricing model as (Adjusted Price - COGS) * order_quantity
        gross_profits = {k: (v - self.base_cost) * order_quantity for k, v in adjusted_results.items()}

        # Select best pricing model based on the lowest adjusted price (per unit)
        best_option = min(adjusted_results, key=adjusted_results.get)
        return adjusted_results, gross_profits, best_option

# Sidebar: User Inputs

st.sidebar.header("User Inputs")

# Pricing Model Inputs
base_cost = st.sidebar.number_input("COGS per Unit (€)", value=2000)
margin = st.sidebar.number_input("Profit Margin (%)", min_value=0, max_value=100, value=30) / 100
customer_value = st.sidebar.number_input("Customer Perceived Value (€)", value=2500)

# Order Quantity Input
order_quantity = st.sidebar.number_input("Order Quantity", value=300)

# Volume discount structure
volume_discount = {200: 0.02, 300: 0.05, 400: 0.1}

# Risk Factor Inputs (using number_input instead of slider)
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
    # Evaluate pricing for the selected order quantity
    pricing_results, gross_profits, best_pricing_option = pricing_model.evaluate_deal(order_quantity)
    df = pd.DataFrame({
        "Price per Unit (€)": pricing_results,
        "Gross Profit (€)": gross_profits
    })
    st.dataframe(df)
    st.success(f"Best Pricing Model: {best_pricing_option}")

with tabs[1]:
    st.title("Sensitivity Analysis")
    st.write("Analysis over a range of Order Quantities")
    quantities = np.arange(100, 501, 50)
    cost_plus_prices = []
    tiered_prices = []
    value_based_prices = []
    best_option = []
    
    for qty in quantities:
        results, _, best = pricing_model.evaluate_deal(qty)
        cost_plus_prices.append(results["Cost-Plus Pricing"])
        tiered_prices.append(results["Tiered Pricing"])
        value_based_prices.append(results["Value-Based Pricing"])
        best_option.append(best)
    
    sa_df = pd.DataFrame({
        "Order Quantity": quantities,
        "Cost-Plus Pricing": cost_plus_prices,
        "Tiered Pricing": tiered_prices,
        "Value-Based Pricing": value_based_prices,
        "Best Option": best_option
    })
    
    st.dataframe(sa_df)
    st.line_chart(sa_df.set_index("Order Quantity")[["Cost-Plus Pricing", "Tiered Pricing", "Value-Based Pricing"]])

with tabs[2]:
    st.title("Supply Chain EOQ Calculation")
    st.write("Supply Chain inputs and EOQ result are provided in the sidebar.")
    st.write(f"Calculated Economic Order Quantity (EOQ): {eoq:.2f} units")
