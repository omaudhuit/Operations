import math
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class PricingModel:
    
    def __init__(self, base_cost, margin, volume_discount, risk_factors, cashflow_model, 
                 customer_value, cashflow_upfront, cashflow_milestone, cashflow_delayed,
                 upfront_payment_pct, milestone_payment_pct, final_payment_pct):
        self.base_cost = base_cost
        self.margin = margin
        self.volume_discount = volume_discount
        self.risk_factors = risk_factors
        self.cashflow_model = cashflow_model
        self.customer_value = customer_value
        self.cashflow_upfront = cashflow_upfront      # used for discount on upfront portion
        self.cashflow_milestone = cashflow_milestone    # surcharge on milestone portion
        self.cashflow_delayed = cashflow_delayed        # surcharge on final payment (delayed)
        # Milestone payment structure parameters (each as a fraction, e.g., 0.3 for 30%)
        self.upfront_payment_pct = upfront_payment_pct
        self.milestone_payment_pct = milestone_payment_pct
        self.final_payment_pct = final_payment_pct

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
        """
        Computes the cost-plus price with volume discount applied.
        Price = (COGS / (1 - Margin)) * (1 - Volume Discount)
        """
        raw_price = self.base_cost / (1 - self.margin)
        vol_disc = self.get_volume_discount(order_quantity)
        return raw_price * (1 - vol_disc)

    def tiered_pricing(self, order_quantity):
        """
        For tiered pricing we start with the raw cost-plus base, then:
         - Apply the volume discount, and if a discount is in effect,
         - Apply an extra discount based on order quantity thresholds.
           If no volume discount is set, extra discount is not applied.
        """
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
            gross_profits: Gross profit = (Final Adjusted Price per Unit - COGS) * Order Quantity.
            best_option: The best pricing model (based on highest gross profit).
        """
        # Compute baseline raw price (without volume discount) for cost-plus pricing.
        cp_raw_no_disc = self.base_cost / (1 - self.margin)
        # Compute cost-plus final price including volume discount.
        cp_final_with_disc = self.cost_plus_pricing(order_quantity)
        
        # Pricing for the other models:
        tiered = self.tiered_pricing(order_quantity)
        value_based = self.value_based_pricing(order_quantity)
        
        # Calculate total risk factor multiplier from risk factors.
        total_risk_factor = sum(self.risk_factors.values()) / 100
        
        # Build raw results:
        raw_results = {
            "Cost-Plus Pricing": cp_raw_no_disc * (1 + total_risk_factor),
            "Tiered Pricing": tiered * (1 + total_risk_factor),
            "Value-Based Pricing": value_based * (1 + total_risk_factor)
        }
        
        # Build final results (before cash flow adjustments):
        final_results = {
            "Cost-Plus Pricing": cp_final_with_disc * (1 + total_risk_factor),
            "Tiered Pricing": tiered * (1 + total_risk_factor),
            "Value-Based Pricing": value_based * (1 + total_risk_factor)
        }
        
        # Apply cash flow adjustments:
        if self.cashflow_model == "upfront":
            final_results = {k: v * (1 - self.cashflow_upfront) for k, v in final_results.items()}
        elif self.cashflow_model == "milestone":
            # Weighted sum for milestone structure with an additional discount on the upfront component.
            final_results = {k: v * (
                self.upfront_payment_pct * (1 - self.cashflow_upfront) +
                self.milestone_payment_pct * (1 + self.cashflow_milestone) +
                self.final_payment_pct * (1 + self.cashflow_delayed)
            ) for k, v in final_results.items()}
        elif self.cashflow_model == "delayed":
            final_results = {k: v * (1 + self.cashflow_delayed) for k, v in final_results.items()}
        
        # Compute VAT on final prices (24% VAT in Greece).
        vat_results = {k: v * 1.24 for k, v in final_results.items()}
        
        # Compute gross profit: (Final Adjusted Price per Unit - COGS) * Order Quantity.
        gross_profits = {k: (v - self.base_cost) * order_quantity for k, v in final_results.items()}
        
        # Round all numeric results to 2 decimals.
        raw_results = {k: round(v, 2) for k, v in raw_results.items()}
        final_results = {k: round(v, 2) for k, v in final_results.items()}
        vat_results = {k: round(v, 2) for k, v in vat_results.items()}
        gross_profits = {k: round(v, 2) for k, v in gross_profits.items()}
        
        best_option = max(gross_profits, key=gross_profits.get)
        return raw_results, final_results, vat_results, gross_profits, best_option

# ---------------------------
# Sidebar: User Inputs

st.sidebar.header("User Inputs")

# Pricing Model Inputs
base_cost = st.sidebar.number_input("COGS per Unit (€)", value=1179)
margin = st.sidebar.number_input("Profit Margin (%)", min_value=0, max_value=100, value=44) / 100
customer_value = st.sidebar.number_input("Customer Perceived Value (€)", value=2500)
order_quantity = st.sidebar.number_input("Order Quantity", value=1)

# Volume Discount Inputs
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

# Milestone Payment Arrangement Inputs
st.sidebar.header("Milestone Payment Arrangement")
upfront_payment_pct = st.sidebar.number_input("Upfront Payment (%)", min_value=0, max_value=100, value=30) / 100
milestone_payment_pct = st.sidebar.number_input("Milestone Payment (%)", min_value=0, max_value=100, value=40) / 100
final_payment_pct = st.sidebar.number_input("Final Payment (%)", min_value=0, max_value=100, value=30) / 100

# Supply Chain Inputs for EOQ Calculation
st.sidebar.header("Supply Chain Inputs")
annual_demand = st.sidebar.number_input("Annual Demand (units)", value=10000)
ordering_cost = st.sidebar.number_input("Ordering Cost per Order (€)", value=50)
holding_cost = st.sidebar.number_input("Inventory Holding Cost per Unit (€)", value=2)
if holding_cost > 0:
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
else:
    eoq = 0
st.sidebar.write(f"Calculated EOQ: {eoq:.2f} units")

# Working Capital Inputs for Cash Conversion Cycle
st.sidebar.header("Working Capital Inputs")
days_inventory = st.sidebar.number_input("Days Inventory Outstanding (DIO)", value=45)
days_receivables = st.sidebar.number_input("Days Sales Outstanding (DSO)", value=30)
days_payables = st.sidebar.number_input("Days Payable Outstanding (DPO)", value=40)

# ---------------------------
# Create Tabs for Results, Sensitivity Analysis, Supply Chain, and Cash Conversion Cycle
tabs = st.tabs(["Results", "Sensitivity Analysis", "Supply Chain", "Cash Conversion Cycle"])

# Instantiate Pricing Model with the new milestone parameters
pricing_model = PricingModel(base_cost, margin, volume_discount, risk_factors, cashflow_model, 
                             customer_value, upfront_discount, milestone_surcharge, delayed_surcharge,
                             upfront_payment_pct, milestone_payment_pct, final_payment_pct)

# Results Tab
with tabs[0]:
    st.title("ERD Pricing Model Evaluation - Results")
    raw_results, final_results, vat_results, gross_profits, best_pricing_option = pricing_model.evaluate_deal(order_quantity)
    
    # Calculate Final Gross Margin (%) for each model:
    # Formula: ((Final Price per Unit - COGS) / Final Price per Unit) * 100
    final_gross_margin = {
        k: round(((final_results[k] - base_cost) / final_results[k] * 100), 2) if final_results[k] != 0 else 0
        for k in final_results
    }
    
    df = pd.DataFrame({
        "Raw Price per Unit (€)": raw_results,
        "Final Price per Unit (€)": final_results,
        "Final Price with VAT (€)": vat_results,
        "Gross Profit (€)": gross_profits,
        "Final Gross Margin (%)": final_gross_margin
    })
    st.dataframe(df)
    
    with st.expander("Explanation of Cost-Plus Pricing Calculation"):
        st.write("1. **Raw Price Calculation:**")
        st.write("   - Computed as: **COGS / (1 - Margin)**, which does not include volume discount.")
        st.write(f"   - For COGS = {base_cost} and Margin = {margin*100:.0f}%, the raw price is ≈ {base_cost/(1-margin):.2f} €.")
        st.write("2. **Volume Discount Application:**")
        st.write("   - The volume discount is applied only in the final price per unit.")
    
    with st.expander("Explanation of Tiered Pricing Calculation"):
        st.write("1. **Tiered Pricing Raw Calculation:**")
        st.write("   - Starts with the same raw price: **COGS / (1 - Margin)**.")
        st.write("2. **Additional Tiered Discount:**")
        st.write("   - Extra discount applied if order quantity exceeds thresholds (e.g., ≥ 400 units: extra 10%).")
    
    with st.expander("Explanation of Value-Based Pricing Calculation"):
        st.write("1. **Customer Perceived Value:**")
        st.write("   - Uses the customer’s perceived value directly as the price per unit.")
    
    with st.expander("Explanation of Risk Factor & Cash Flow Adjustments"):
        st.write("1. **Risk Factor Adjustment:**")
        st.write("   - Prices are increased by the sum of all risk factors as a percentage (e.g., 10% total risk increases price by 1.10).")
        st.write("2. **Cash Flow Adjustment:**")
        st.write(f"   - 'upfront' applies a {upfront_discount*100:.0f}% discount, 'milestone' applies a {milestone_surcharge*100:.0f}% surcharge, and 'delayed' a {delayed_surcharge*100:.0f}% surcharge.")
    
    with st.expander("Explanation of Gross Profit & Margin Calculation"):
        st.write("Gross Profit = (Final Adjusted Price per Unit - COGS) × Order Quantity")
        st.write("Final Gross Margin (%) = ((Final Price per Unit - COGS) / Final Price per Unit) × 100")
    
    st.success(f"Best Pricing Model (based on highest Gross Profit): {best_pricing_option}")

# Sensitivity Analysis Tab
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
                                      customer_value, upfront_discount, milestone_surcharge, delayed_surcharge,
                                      upfront_payment_pct, milestone_payment_pct, final_payment_pct)
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
                                      customer_value, upfront_discount, milestone_surcharge, delayed_surcharge,
                                      upfront_payment_pct, milestone_payment_pct, final_payment_pct)
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
        quantities = np.arange(100, 1001, 50)
        margins = np.linspace(0.1, 0.9, 20)
        heat_data = np.zeros((len(margins), len(quantities)))
        for i, m in enumerate(margins):
            for j, qty in enumerate(quantities):
                temp_model = PricingModel(base_cost, m, volume_discount, risk_factors, cashflow_model,
                                          customer_value, upfront_discount, milestone_surcharge, delayed_surcharge,
                                          upfront_payment_pct, milestone_payment_pct, final_payment_pct)
                _, _, _, gross, _ = temp_model.evaluate_deal(qty)
                heat_data[i, j] = gross["Cost-Plus Pricing"]
        fig, ax = plt.subplots()
        sns.heatmap(heat_data, xticklabels=quantities, yticklabels=np.round(margins,2),
                    cmap="YlGnBu", ax=ax)
        ax.set_xlabel("Order Quantity")
        ax.set_ylabel("Margin")
        ax.set_title("Cost-Plus Gross Profit Heatmap")
        st.pyplot(fig)

# Supply Chain Tab
with tabs[2]:
    st.title("Supply Chain EOQ Calculation")
    st.write("Supply Chain inputs and EOQ result are provided in the sidebar.")
    st.write(f"Calculated Economic Order Quantity (EOQ): {eoq:.2f} units")
    
    with st.expander("How EOQ Impacts Pricing"):
        st.write("Optimizing your order quantity using the EOQ model helps minimize ordering and holding costs.")
        st.write("A lower EOQ allows for more frequent replenishment with lower holding costs, enabling better volume pricing negotiations.")
        st.write("Optimized inventory costs support more competitive pricing and higher gross margins.")

# Cash Conversion Cycle Tab
with tabs[3]:
    st.title("Cash Conversion Cycle")
    st.write("This tab shows a timeline for the cash conversion cycle based on working capital inputs.")
    # Calculate Cash Conversion Cycle (CCC = DIO + DSO - DPO)
    ccc = days_inventory + days_receivables - days_payables
    st.write(f"**Days Inventory Outstanding (DIO):** {days_inventory} days")
    st.write(f"**Days Sales Outstanding (DSO):** {days_receivables} days")
    st.write(f"**Days Payables Outstanding (DPO):** {days_payables} days")
    st.write(f"**Cash Conversion Cycle (CCC):** {ccc} days")
    
    # Timeline visualization: Plot DIO and DSO as stacked bars with an arrow showing DPO reduction
    fig, ax = plt.subplots(figsize=(10, 3))
    total = days_inventory + days_receivables
    ax.broken_barh([(0, days_inventory)], (20, 9), facecolors='skyblue', label='DIO')
    ax.broken_barh([(days_inventory, days_receivables)], (20, 9), facecolors='lightgreen', label='DSO')
    ax.annotate('', xy=(ccc, 29), xytext=(total, 29),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text((total + ccc) / 2, 31, f'- DPO: {days_payables} days',
            color='red', ha='center', va='bottom', fontsize=10)
    
    # If the cash flow model is milestone, compute milestone payment breakdown and annotate on the chart.
    if cashflow_model == "milestone":
        # Re-evaluate the deal to get the final prices and best pricing model.
        _, final_results_tmp, _, _, best_option_tmp = pricing_model.evaluate_deal(order_quantity)
        best_final_price = final_results_tmp[best_option_tmp]
        # Compute the weighted factor used in milestone adjustments:
        weighted_factor = (upfront_payment_pct * (1 - upfront_discount) +
                           milestone_payment_pct * (1 + milestone_surcharge) +
                           final_payment_pct * (1 + delayed_surcharge))
        # Compute relative ratios for each payment component.
        ratio_upfront = (upfront_payment_pct * (1 - upfront_discount)) / weighted_factor
        ratio_milestone = (milestone_payment_pct * (1 + milestone_surcharge)) / weighted_factor
        ratio_final = (final_payment_pct * (1 + delayed_surcharge)) / weighted_factor
        # Compute payment amounts per unit.
        amount_upfront = best_final_price * ratio_upfront
        amount_milestone = best_final_price * ratio_milestone
        amount_final = best_final_price * ratio_final
        # Multiply by order quantity to get total payment amounts.
        total_amount_upfront = amount_upfront * order_quantity
        total_amount_milestone = amount_milestone * order_quantity
        total_amount_final = amount_final * order_quantity
        
        # Prepare annotation text for the breakdown.
        breakdown_text = (f"Milestone Payment Breakdown (Order Total):\n"
                          f"Upfront: €{total_amount_upfront:.2f}   |   "
                          f"Milestone: €{total_amount_milestone:.2f}   |   "
                          f"Final: €{total_amount_final:.2f}")
        # Place the breakdown annotation below the timeline.
        ax.text(total/2, 5, breakdown_text, ha='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
    
    ax.set_xlim(0, total + max(0, days_payables) + 5)
    ax.set_ylim(0, 45)
    ax.set_xlabel('Days')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Cash Conversion Cycle Timeline')
    plt.tight_layout()
    st.pyplot(fig)
    
    timeline = f"""
    **Timeline:**
    
    Inventory → Sales → Receivables Collection  
    [DIO: {days_inventory} days] → [DSO: {days_receivables} days]  
    → Subtract [DPO: {days_payables} days]  
    *Net Cash Conversion Cycle = {ccc} days*
    """
    st.markdown(timeline)
