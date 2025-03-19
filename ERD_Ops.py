import streamlit as st
import pandas as pd
import numpy as np

class PricingModel:
    def __init__(self, base_cost, margin, volume_discount, risk_factors, cashflow_model):
        self.base_cost = base_cost
        self.margin = margin
        self.volume_discount = volume_discount
        self.risk_factors = risk_factors
        self.cashflow_model = cashflow_model

    def cost_plus_pricing(self, order_quantity):
        """Cost-plus pricing model with margin consideration."""
        return (self.base_cost * (1 + self.margin)) * (1 - self.volume_discount.get(order_quantity, 0))

    def tiered_pricing(self, order_quantity):
        """Tiered pricing based on volume."""
        return self.cost_plus_pricing(order_quantity) * (0.9 if order_quantity >= 400 else 0.95 if order_quantity >= 300 else 1)

    def evaluate_deal(self, order_quantity):
        """Evaluates different pricing models and incorporates risk and cash flow variations."""
        cost_plus = self.cost_plus_pricing(order_quantity)
        tiered = self.tiered_pricing(order_quantity)

        # Apply risk factor adjustments
        total_risk_factor = sum(self.risk_factors.values()) / 100  # Convert to percentage
        adjusted_results = {"Cost-Plus Pricing": cost_plus * (1 + total_risk_factor),
                            "Tiered Pricing": tiered * (1 + total_risk_factor)}

        # Apply cash flow model impact
        if self.cashflow_model == "upfront":
            adjusted_results = {k: v * 0.95 for k, v in adjusted_results.items()}  # 5% discount
        elif self.cashflow_model == "milestone":
            adjusted_results = {k: v * 1.02 for k, v in adjusted_results.items()}  # 2% increase
        elif self.cashflow_model == "delayed":
            adjusted_results = {k: v * 1.05 for k, v in adjusted_results.items()}  # 5% increase

        # Select best pricing model
        best_option = min(adjusted_results, key=adjusted_results.get)
        return adjusted_results, best_option

# Sidebar: User Inputs
st.sidebar.header("User Inputs")

# COGS and Margin Inputs
base_cost = st.sidebar.number_input("COGS per Unit (€)", value=2000)
margin = st.sidebar.slider("Profit Margin (%)", 0, 100, 30) / 100

# Order Quantity Input
order_quantity = st.sidebar.number_input("Order Quantity", value=300)

# Volume discount structure
volume_discount = {200: 0.02, 300: 0.05, 400: 0.1}

# Risk Factor Inputs
st.sidebar.subheader("Risk Factors (as % Impact)")
supply_chain_risk = st.sidebar.slider("Supply Chain Risk (%)", 0, 10, 5)
regulatory_risk = st.sidebar.slider("Regulatory Compliance Risk (%)", 0, 10, 3)
payment_risk = st.sidebar.slider("Payment Delay Risk (%)", 0, 10, 4)
competition_risk = st.sidebar.slider("Competitive Market Pressure (%)", 0, 10, 2)

risk_factors = {
    "Supply Chain Risk": supply_chain_risk,
    "Regulatory Risk": regulatory_risk,
    "Payment Risk": payment_risk,
    "Competition Risk": competition_risk,
}

# Cash Flow Options
st.sidebar.subheader("Cash Flow Management Strategy")
cashflow_model = st.sidebar.selectbox("Select Payment Structure", ["upfront", "milestone", "delayed"])

# Create Tabs for Results and Sensitivity Analysis
tabs = st.tabs(["Results", "Sensitivity Analysis"])

# Instantiate Pricing Model
pricing_model = PricingModel(base_cost, margin, volume_discount, risk_factors, cashflow_model)

with tabs[0]:
    st.title("ERD Pricing Model Evaluation - Results")
    # Model Evaluation for the selected order_quantity
    pricing_results, best_pricing_option = pricing_model.evaluate_deal(order_quantity)
    df = pd.DataFrame.from_dict(pricing_results, orient='index', columns=['Price per Unit (€)'])
    st.dataframe(df)
    st.success(f"Best Pricing Model: {best_pricing_option}")

with tabs[1]:
    st.title("Sensitivity Analysis")
    st.write("Analysis over a range of Order Quantities")
    # Run sensitivity analysis over a range of order quantities
    quantities = np.arange(100, 501, 50)
    sensitivity_data = {"Order Quantity": quantities}
    cost_plus_prices = []
    tiered_prices = []
    best_option = []
    
    for qty in quantities:
        results, best = pricing_model.evaluate_deal(qty)
        cost_plus_prices.append(results["Cost-Plus Pricing"])
        tiered_prices.append(results["Tiered Pricing"])
        best_option.append(best)
    
    sa_df = pd.DataFrame({
        "Order Quantity": quantities,
        "Cost-Plus Pricing": cost_plus_prices,
        "Tiered Pricing": tiered_prices,
        "Best Option": best_option
    })
    
    st.dataframe(sa_df)
    st.line_chart(sa_df.set_index("Order Quantity")[["Cost-Plus Pricing", "Tiered Pricing"]])
