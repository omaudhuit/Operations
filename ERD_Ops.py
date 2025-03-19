import streamlit as st
import pandas as pd

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

# Streamlit UI
st.title("ERD Pricing Model Evaluation")

# User Inputs
base_cost = st.number_input("Manufacturing Cost per Unit (€)", value=2000)
margin = st.slider("Profit Margin (%)", 0, 100, 30) / 100
order_quantity = st.number_input("Order Quantity", value=300)

# Volume discount structure
volume_discount = {200: 0.02, 300: 0.05, 400: 0.1}

# Risk Factor Inputs
st.subheader("Risk Factors (as % Impact)")
supply_chain_risk = st.slider("Supply Chain Risk (%)", 0, 10, 5)
regulatory_risk = st.slider("Regulatory Compliance Risk (%)", 0, 10, 3)
payment_risk = st.slider("Payment Delay Risk (%)", 0, 10, 4)
competition_risk = st.slider("Competitive Market Pressure (%)", 0, 10, 2)

risk_factors = {
    "Supply Chain Risk": supply_chain_risk,
    "Regulatory Risk": regulatory_risk,
    "Payment Risk": payment_risk,
    "Competition Risk": competition_risk,
}

# Cash Flow Options
st.subheader("Cash Flow Management Strategy")
cashflow_model = st.selectbox("Select Payment Structure", ["upfront", "milestone", "delayed"])

# Model Evaluation
pricing_model = PricingModel(base_cost, margin, volume_discount, risk_factors, cashflow_model)
pricing_results, best_pricing_option = pricing_model.evaluate_deal(order_quantity)

# Display Results
st.subheader("Pricing Evaluation Results")
df = pd.DataFrame.from_dict(pricing_results, orient='index', columns=['Price per Unit (€)'])
st.dataframe(df)

st.success(f"**Best Pricing Model:** {best_pricing_option}")
