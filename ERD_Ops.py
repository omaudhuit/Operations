import math
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class PricingModel:
    def __init__(self, base_cost, margin, volume_discount, risk_factors, cashflow_model, 
                 customer_value, cashflow_upfront, cashflow_milestone, cashflow_delayed):
        self.base_cost = base_cost
        self.margin = margin
        self.volume_discount = volume_discount
        self.risk_factors = risk_factors
        self.cashflow_model = cashflow_model
        self.customer_value = customer_value
        self.cashflow_upfront = cashflow_upfront      # e.g., 5% discount => 0.05
        self.cashflow_milestone = cashflow_milestone  # e.g., 2% surcharge => 0.02
        self.cashflow_delayed = cashflow_delayed      # e.g., 5% surcharge => 0.05

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
           Raw Price = COGS / (1 - Margin); then if applicable, apply volume discount."""
        raw_price = self.base_cost / (1 - self.margin)
        vol_disc = self.get_volume_discount(order_quantity)
        return raw_price * (1 - vol_disc)

    def tiered_pricing(self, order_quantity):
        """For tiered pricing we start from the raw cost-plus base, then:
             - Apply the volume discount, and if a discount is in effect,
             - Apply an extra discount based on order quantity thresholds.
           If no volume discount is set, extra discount is not applied."""
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
            raw_results: Prices per unit after risk adjustments (before cash flow adjustments). 
                         For Cost‑Plus Pricing, the raw price excludes volume discount.
            final_results: Prices per unit after applying cash flow adjustments.
                           For Cost‑Plus Pricing, the final price includes the volume discount.
            vat_results: Final prices including VAT (24%).
            gross_profits: Gross profit calculated as (Final Adjusted Price per Unit - COGS) * Order Quantity.
            best_option: The best pricing model (based on highest gross profit).
        """
        # Compute baseline raw price (no volume discount) for cost-plus pricing
        cp_raw_no_disc = self.base_cost / (1 - self.margin)
        # Compute cost-plus final price including volume discount (using our function)
        cp_final_with_disc = self.cost_plus_pricing(order_quantity)
        
        # Calculate pricing for the other models using the existing functions:
        tiered = self.tiered_pricing(order_quantity)
        value_based = self.value_based_pricing(order_quantity)
        
        # Calculate total risk factor multiplier from risk factors
        total_risk_factor = sum(self.risk_factors.values()) / 100
        
        # Build raw results:
        # For Cost-Plus Pricing, use baseline raw (without volume discount) and then adjust for risk.
        raw_results = {
            "Cost-Plus Pricing": cp_raw_no_disc * (1 + total_risk_factor),
            "Tiered Pricing": tiered * (1 + total_risk_factor),
            "Value-Based Pricing": value_based * (1 + total_risk_factor)
        }
        
        # Build final results:
        # For Cost-Plus Pricing, include the volume discount.
        final_results = {
            "Cost-Plus Pricing": cp_final_with_disc * (1 + total_risk_factor),
            "Tiered Pricing": tiered * (1 + total_risk_factor),
            "Value-Based Pricing": value_based * (1 + total_risk_factor)
        }
        
        # Apply cash flow adjustments to final results
        if self.cashflow_model == "upfront":
            final_results = {k: v * (1 - self.cashflow_upfront) for k, v in final_results.items()}
        elif self.cashflow_model == "milestone":
            final_results = {k: v * (1 + self.cashflow_milestone) for k, v in final_results.items()}
        elif self.cashflow_model == "delayed":
            final_results = {k: v * (1 + self.cashflow_delayed) for k, v in final_results.items()}
        
        # Compute VAT on final prices (24% VAT in Greece)
        vat_results = {k: v * 1.24 for k, v in final_results.items()}
        
        # Compute gross profit: (Final Adjusted Price per Unit - COGS) * Order Quantity
        gross_profits = {k: (v - self.base_cost) * order_quantity for k, v in final_results.items()}

        # Round all numerical results to 2 decimals.
        raw_results = {k: round(v, 2) for k, v in raw_results.items()}
        final_results = {k: round(v, 2) for k, v in final_results.items()}
        vat_results = {k: round(v, 2) for k, v in vat_results.items()}
        gross_profits = {k: round(v, 2) for k, v in gross_profits.items()}
        
        # Select best pricing option based on highest gross profit.
        best_option = max(gross_profits, key=gross_profits.get)
        return raw_results, final_results, vat_results, gross_profits, best_option

# Sidebar: User Inputs
st.sidebar.header("User Inputs")

# Pricing Model Inputs
base_cost = st.sidebar.number_input("COGS per Unit (€)", value=1179)
margin = st.sidebar.number_input("Profit Margin (%)", min_value=0, max_value=100, value=44) / 100
customer_value = st.sidebar.number_input("Customer Perceived Value (€)", value=2500)

# Order Quantity Input
order_quantity = st.sidebar.number_input("Order Quantity", value=1)

# Volume Discount Inputs as User Defined
st.sidebar.header("Volume Discount Inputs")
discount_200 = st.sidebar.number_input("Discount at Order Quantity 200 (%)", min_value=0, max_value=100, value=0) / 100
discount_300 = st.sidebar.number_input("Discount at Order Quantity 300 (%)", min_value=0, max_value=100, value=0) / 100
discount_400 = st.sidebar.number_input("Discount at Order Quantity 400 (%)", min_value=0, max_value=100, value=0) / 100
volume_discount = {200: discount_200, 300: discount_300, 400: discount_400}

# Risk Factor Inputs
st.sidebar.subheader("Risk Factors (as % Impact)")
supply_chain_risk = st.sidebar.number_input("Supply Chain Risk (%)", min_value=0, max_value=10, value=0)
regulatory_risk = st.sidebar.number_input("Regulatory Compliance Risk (%)", min_value=0, max_value=10, value=0)
payment_risk = st.sidebar.number_input("Payment Delay Risk (%)", min_value=0, max_value=10, value=0)
competition_risk = st.sidebar.number_input("Competitive Market Pressure (%)", min_value=0, max_value=10, value=0)
risk_factors = {
    "Supply Chain Risk": supply_chain_risk,
    "Regulatory Risk": regulatory_risk,
    "Payment Risk": payment_risk,
    "Competition Risk": competition_risk,
}

# Cash Flow Inputs
st.sidebar.subheader("Cash Flow Management Strategy")
cashflow_model = st.sidebar.selectbox("Select Payment Structure", ["upfront", "milestone", "delayed"])
upfront_discount = st.sidebar.number_input("Upfront Cash Flow Discount (%)", min_value=0, max_value=100, value=0) / 100
milestone_surcharge = st.sidebar.number_input("Milestone Cash Flow Surcharge (%)", min_value=0, max_value=100, value=0) / 100
delayed_surcharge = st.sidebar.number_input("Delayed Cash Flow Surcharge (%)", min_value=0, max_value=100, value=0) / 100

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
pricing_model = PricingModel(base_cost, margin, volume_discount, risk_factors, cashflow_model, 
                             customer_value, upfront_discount, milestone_surcharge, delayed_surcharge)

with tabs[0]:
    st.title("ERD Pricing Model Evaluation - Results")
    raw_results, final_results, vat_results, gross_profits, best_pricing_option = pricing_model.evaluate_deal(order_quantity)
    
    # Display the calculated values in a table, including a VAT column
    df = pd.DataFrame({
        "Raw Price per Unit (€)": raw_results,
        "Final Price per Unit (€)": final_results,
        "Final Price with VAT (€)": vat_results,
        "Gross Profit (€)": gross_profits
    })
    st.dataframe(df)
    
    # Detailed explanation sections using expanders
    with st.expander("Explanation of Cost-Plus Pricing Calculation"):
        st.write("1. **Raw Price Calculation:**")
        st.write("   - The raw price is computed as: **COGS / (1 - Margin)**.")
        st.write(f"   - For COGS = {base_cost} and Margin = {margin*100:.0f}%, Raw Price = {base_cost} / (1 - {margin:.2f}) ≈ {base_cost/(1-margin):.2f}.")
        st.write("2. **Volume Discount Application:**")
        st.write("   - A volume discount is applied if the order quantity exceeds set thresholds. "
                 "For example, if a discount is defined for that range, the raw price is reduced accordingly.")
    
    with st.expander("Explanation of Tiered Pricing Calculation"):
        st.write("1. **Tiered Pricing Raw Calculation:**")
        st.write("   - It starts with the same raw price: **COGS / (1 - Margin)**.")
        st.write("2. **Additional Tiered Discount:**")
        st.write("   - An extra discount is applied on top of any volume discount if the order quantity is high:")
        st.write("     - If order quantity is ≥ 400, an extra 10% discount is applied.")
        st.write("     - If order quantity is ≥ 300, an extra 5% discount is applied.")
        st.write("   - If no volume discount is active, no extra discount is applied, and the raw price is used.")
    
    with st.expander("Explanation of Value-Based Pricing Calculation"):
        st.write("1. **Customer Perceived Value:**")
        st.write("   - The value-based pricing method uses the customer’s perceived value directly as the price per unit.")
        st.write("   - It is not derived from COGS or a margin; rather it reflects what customers are willing to pay.")
        st.write("   - This approach is useful when a premium price is supported by market differentiation.")
    
    with st.expander("Explanation of Risk Factor & Cash Flow Adjustments"):
        st.write("1. **Risk Factor Adjustment:**")
        st.write("   - Prices are increased by the sum of all risk factors (expressed as a percentage). "
                 "For example, with a total risk of 10%, prices are multiplied by 1.10.")
        st.write("2. **Cash Flow Management Adjustment:**")
        st.write("   - After risk adjustments, a cash flow strategy is applied based on the user-defined inputs:")
        st.write(f"     - 'upfront' applies a {upfront_discount*100:.0f}% discount (multiplies by 1 - {upfront_discount:.2f}).")
        st.write(f"     - 'milestone' increases prices by {milestone_surcharge*100:.0f}% (multiplies by 1 + {milestone_surcharge:.2f}).")
        st.write(f"     - 'delayed' increases prices by {delayed_surcharge*100:.0f}% (multiplies by 1 + {delayed_surcharge:.2f}).")
    
    with st.expander("Explanation of Gross Profit Calculation"):
        st.write("Gross Profit is calculated as:")
        st.write("   - **Gross Profit = (Final Adjusted Price per Unit - COGS) * Order Quantity**")
        st.write("   - This represents the per-unit profit (after all adjustments) multiplied by the number of units ordered.")
    
    st.success(f"Best Pricing Model (based on highest Gross Profit): {best_pricing_option}")

with tabs[1]:
    st.title("Sensitivity Analysis")
    st.write("Explore how changes in key parameters impact Gross Profit.")
    param = st.selectbox("Select parameter to vary", ["Order Quantity", "Margin", "Risk Factor", "Order Quantity and Margin"])
    
    if param == "Order Quantity":
        quantities = np.arange(100, 1001, 50)
        sim_list = []
        for qty in quantities:
            _, _, _, gross, _ = pricing_model.evaluate_deal(qty)
            sim_list.append({
                "Order Quantity": qty,
                "Cost-Plus Gross Profit": gross["Cost-Plus Pricing"],
                "Tiered Gross Profit": gross["Tiered Pricing"],
                "Value-Based Gross Profit": gross["Value-Based Pricing"]
            })
        sim_df = pd.DataFrame(sim_list)
        st.dataframe(sim_df)
        st.line_chart(sim_df.set_index("Order Quantity")[["Cost-Plus Gross Profit", "Tiered Gross Profit", "Value-Based Gross Profit"]])
    
    elif param == "Margin":
        margins = np.linspace(0.1, 0.9, 20)
        sim_list = []
        for m in margins:
            temp_model = PricingModel(base_cost, m, volume_discount, risk_factors, cashflow_model,
                                      customer_value, upfront_discount, milestone_surcharge, delayed_surcharge)
            _, _, _, gross, _ = temp_model.evaluate_deal(order_quantity)
            sim_list.append({
                "Margin": m,
                "Cost-Plus Gross Profit": gross["Cost-Plus Pricing"],
                "Tiered Gross Profit": gross["Tiered Pricing"],
                "Value-Based Gross Profit": gross["Value-Based Pricing"]
            })
        sim_df = pd.DataFrame(sim_list)
        st.dataframe(sim_df)
        st.line_chart(sim_df.set_index("Margin")[["Cost-Plus Gross Profit", "Tiered Gross Profit", "Value-Based Gross Profit"]])
    
    elif param == "Risk Factor":
        risk_values = np.linspace(0, 20, 21)
        sim_list = []
        for r in risk_values:
            risk_dict = {
                "Supply Chain Risk": r,
                "Regulatory Risk": r,
                "Payment Risk": r,
                "Competition Risk": r
            }
            temp_model = PricingModel(base_cost, margin, volume_discount, risk_dict, cashflow_model,
                                      customer_value, upfront_discount, milestone_surcharge, delayed_surcharge)
            _, _, _, gross, _ = temp_model.evaluate_deal(order_quantity)
            sim_list.append({
                "Total Risk (%)": r,
                "Cost-Plus Gross Profit": gross["Cost-Plus Pricing"],
                "Tiered Gross Profit": gross["Tiered Pricing"],
                "Value-Based Gross Profit": gross["Value-Based Pricing"]
            })
        sim_df = pd.DataFrame(sim_list)
        st.dataframe(sim_df)
        st.line_chart(sim_df.set_index("Total Risk (%)")[["Cost-Plus Gross Profit", "Tiered Gross Profit", "Value-Based Gross Profit"]])
    
    elif param == "Order Quantity and Margin":
        # 2D Sensitivity analysis: vary both order quantity and margin, showing Cost-Plus Gross Profit
        quantities = np.arange(100, 1001, 50)
        margins = np.linspace(0.1, 0.9, 20)
        heat_data = np.zeros((len(margins), len(quantities)))
        for i, m in enumerate(margins):
            for j, qty in enumerate(quantities):
                temp_model = PricingModel(base_cost, m, volume_discount, risk_factors, cashflow_model,
                                          customer_value, upfront_discount, milestone_surcharge, delayed_surcharge)
                _, _, _, gross, _ = temp_model.evaluate_deal(qty)
                heat_data[i, j] = gross["Cost-Plus Pricing"]
        # Create a heatmap using seaborn
        fig, ax = plt.subplots()
        sns.heatmap(heat_data, xticklabels=quantities, yticklabels=np.round(margins,2),
                    cmap="YlGnBu", ax=ax)
        ax.set_xlabel("Order Quantity")
        ax.set_ylabel("Margin")
        ax.set_title("Cost-Plus Gross Profit Heatmap")
        st.pyplot(fig)

with tabs[2]:
    st.title("Supply Chain EOQ Calculation")
    st.write("Supply Chain inputs and EOQ result are provided in the sidebar.")
    st.write(f"Calculated Economic Order Quantity (EOQ): {eoq:.2f} units")
    
    # Explanation of EOQ impact on pricing
    with st.expander("How EOQ Impacts Pricing"):
        st.write("Optimizing your order quantity using the EOQ model helps minimize the total costs associated with ordering and holding inventory.")
        st.write("A lower EOQ indicates that you can replenish stock more frequently with lower holding costs, often enabling better negotiations for volume pricing or discounts from suppliers.")
        st.write("On the sales side, an optimal EOQ helps control overall costs, which can lead to more competitive pricing and higher gross margins.")
        st.write("Essentially, while EOQ is a supply chain metric, its effect on the cost structure indirectly influences your pricing strategy and profitability.")
