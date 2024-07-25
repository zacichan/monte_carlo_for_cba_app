import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm, gaussian_kde

# Options data
options_data = {
    "option_1": {
        "benefit": {"low": 22.08, "central": 32.8, "high": 38.64},
        "cost": {"low": 20.5, "central": 27.6, "high": 36.4},
    },
    "option_2": {
        "benefit": {"low": 26.24, "central": 38, "high": 42.64},
        "cost": {"low": 25.3, "central": 32.8, "high": 42.2},
    },
    "option_3": {
        "benefit": {"low": 66.78, "central": 94.7, "high": 196.02},
        "cost": {"low": 45.09, "central": 63.5, "high": 86.09},
    },
}

def simulate_distribution(low, central, high, dist_type, num_sims=10000):
    np.random.seed(42)  # Set seed for reproducibility

    def uniform_1(low, high, size):
        return np.random.uniform(low, high, size)

    def normal_2(low, high, size):
        mean_x = (high + low) / 2
        sd_x = (high - low) / 4
        return np.random.normal(mean_x, sd_x, size)

    def normal_3(low, central, high, size):
        mean_x = central
        sd_x = (high - low) / 4
        return np.random.normal(mean_x, sd_x, size)

    def log_normal_4(low, central, high, size):
        def f(sigma):
            mu = np.log(central) + sigma**2
            return np.abs(
                lognorm.cdf(high, sigma, scale=np.exp(mu))
                - lognorm.cdf(low, sigma, scale=np.exp(mu))
                - 0.95
            )

        sigma_x = minimize_scalar(f, bounds=(0, 1), method="bounded").x
        mu_x = np.log(central) + sigma_x**2
        return lognorm.rvs(sigma_x, scale=np.exp(mu_x), size=size)

    if dist_type == "Uniform":
        results = uniform_1(low, high, num_sims)
    elif dist_type == "Normal (without central)":
        results = normal_2(low, high, num_sims)
    elif dist_type == "Normal (with central)":
        results = normal_3(low, central, high, num_sims)
    elif dist_type == "Log-Normal":
        results = log_normal_4(low, central, high, num_sims)

    return results

def plot_distribution(results, dist_type, title):
    density = gaussian_kde(results)
    x_vals = np.linspace(min(results), max(results), 1000)
    y_vals = density(x_vals)

    lower_bound = np.percentile(results, 2.5)
    upper_bound = np.percentile(results, 97.5)
    modal_point = x_vals[np.argmax(y_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Density Curve"))

    # Shading the area under the curve within the 95% CI
    fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                [
                    [lower_bound],
                    x_vals[(x_vals >= lower_bound) & (x_vals <= upper_bound)],
                    [upper_bound],
                ]
            ),
            y=np.concatenate(
                [[0], y_vals[(x_vals >= lower_bound) & (x_vals <= upper_bound)], [0]]
            ),
            fill="toself",
            fillcolor="rgba(255, 160, 122, 0.5)",
            line=dict(color="rgba(255, 160, 122, 0)"),
            showlegend=True,
            name="95% CI",
        )
    )

    fig.add_vline(
        x=lower_bound, line=dict(color="red", dash="dash"), name="95% CI Bound"
    )
    fig.add_vline(
        x=upper_bound, line=dict(color="red", dash="dash"), name="95% CI Bound"
    )

    # Add modal point and CI summary as annotations
    summary_text = f"Modal Point: {modal_point:.2f}\n95% CI: [{lower_bound:.2f}, {upper_bound:.2f}]"
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        showarrow=False,
        text=summary_text,
        bordercolor="black",
        borderwidth=1,
    )

    fig.update_layout(
        title=title, xaxis_title="Value", yaxis_title="Density", showlegend=True
    )
    return fig

def main():
    st.set_page_config(page_title="Monte Carlo Cost-Benefit Analysis", layout="wide")

    st.title("Monte Carlo Cost-Benefit Analysis")
    st.markdown("### Perform Monte Carlo simulations to analyze cost-benefit scenarios.")

    # Sidebar for inputs
    st.sidebar.header("Inputs")

    st.sidebar.subheader("Select Option")
    option = st.sidebar.selectbox("Option", options_data.keys())

    if option:
        data = options_data[option]
        st.sidebar.subheader("Cost Inputs")
        low_cost = st.sidebar.number_input("Low Cost", value=data["cost"]["low"])
        central_cost = st.sidebar.number_input("Central Cost", value=data["cost"]["central"])
        high_cost = st.sidebar.number_input("High Cost", value=data["cost"]["high"])
        cost_dist_type = st.sidebar.selectbox(
            "Cost Distribution Type",
            ["Uniform", "Normal (without central)", "Normal (with central)", "Log-Normal"],
            index=3,
        )

        st.sidebar.subheader("Benefit Inputs")
        low_benefit = st.sidebar.number_input("Low Benefit", value=data["benefit"]["low"])
        central_benefit = st.sidebar.number_input("Central Benefit", value=data["benefit"]["central"])
        high_benefit = st.sidebar.number_input("High Benefit", value=data["benefit"]["high"])
        benefit_dist_type = st.sidebar.selectbox(
            "Benefit Distribution Type",
            ["Uniform", "Normal (without central)", "Normal (with central)", "Log-Normal"],
            index=3,
        )

        run_simulation = st.sidebar.button("Run Simulation")

        if run_simulation:
            cost_results = simulate_distribution(low_cost, central_cost, high_cost, cost_dist_type)
            benefit_results = simulate_distribution(low_benefit, central_benefit, high_benefit, benefit_dist_type)

            cost_modal_point = np.mean(cost_results)
            cost_lower_bound = np.percentile(cost_results, 2.5)
            cost_upper_bound = np.percentile(cost_results, 97.5)

            benefit_modal_point = np.mean(benefit_results)
            benefit_lower_bound = np.percentile(benefit_results, 2.5)
            benefit_upper_bound = np.percentile(benefit_results, 97.5)

            bcr_results = benefit_results / cost_results
            bcr_modal_point = np.mean(bcr_results)
            bcr_lower_bound = np.percentile(bcr_results, 2.5)
            bcr_upper_bound = np.percentile(bcr_results, 97.5)

            st.markdown("### Results Summary")
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("Cost Modal Point", f"{cost_modal_point:.2f}")
                col1.metric("Cost 95% CI", f"{cost_lower_bound:.2f} - {cost_upper_bound:.2f}")
                col2.metric("Benefit Modal Point", f"{benefit_modal_point:.2f}")
                col2.metric("Benefit 95% CI", f"{benefit_lower_bound:.2f} - {benefit_upper_bound:.2f}")
                col3.metric("BCR Modal Point",f"{bcr_modal_point:.2f}")
                col3.metric("BCR 95% CI", f"{bcr_lower_bound:.2f} - {bcr_upper_bound:.2f}")

            with st.expander("Detailed Distributions"):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Cost Distribution")
                    cost_fig = plot_distribution(cost_results, cost_dist_type, "Cost Distribution")
                    st.plotly_chart(cost_fig, use_container_width=True)

                with col2:
                    st.subheader("Benefit Distribution")
                    benefit_fig = plot_distribution(benefit_results, benefit_dist_type, "Benefit Distribution")
                    st.plotly_chart(benefit_fig, use_container_width=True)

                st.subheader("Benefit-Cost Ratio (BCR) Distribution")
                bcr_fig = plot_distribution(bcr_results, "BCR", "Benefit-Cost Ratio Distribution")
                st.plotly_chart(bcr_fig, use_container_width=True)

            with st.expander("Methodology"):
                st.markdown(
                    """
                ### Methodology of Monte Carlo Cost-Benefit Analysis

                1. **Modeling Cost and Benefit Distributions**:
                    - Each option's cost and benefit are represented by specific probability distributions (e.g., Uniform, Normal, Log-Normal).
                    - Distribution parameters are determined based on user-provided low, central, and high estimates.

                2. **Simulation Process**:
                    - Generate a large number of random samples (e.g., 10,000) from the specified cost and benefit distributions to simulate potential outcomes.
                
                3. **Calculating Benefit-Cost Ratio (BCR)**:
                    - For each simulated sample, compute the Benefit-Cost Ratio (BCR) by dividing the sampled benefit by the sampled cost

                4. **Statistical Analysis of Results**:
                    - Determine key statistical metrics for the cost, benefit, and BCR distributions, including:
                        - **Modal Point**: The most frequent (mode) or average (mean) value.
                        - **95% Confidence Interval (CI)**: The range within which 95% of the simulated values fall, providing an estimate of uncertainty.

                For more details, please refer to the [presentation](https://zacpaynethompson.github.io/monte_carlo_cba/#/title-slide).
                """
                )

if __name__ == "__main__":
    main()
