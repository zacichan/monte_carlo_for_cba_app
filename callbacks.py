import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
from helpers import uniform_1, normal_2, normal_3, log_normal_4
from data.options import options_data

def register_callbacks(app):

    @app.callback(
        [
            Output("low_cost", "value"),
            Output("central_cost", "value"),
            Output("high_cost", "value"),
            Output("low_benefit", "value"),
            Output("central_benefit", "value"),
            Output("high_benefit", "value"),
        ],
        [Input("option_selector", "value")],
    )
    def update_inputs(option):
        if option:
            return (
                options_data[option]["cost"]["low"],
                options_data[option]["cost"]["central"],
                options_data[option]["cost"]["high"],
                options_data[option]["benefit"]["low"],
                options_data[option]["benefit"]["central"],
                options_data[option]["benefit"]["high"],
            )
        return dash.no_update

    @app.callback(
        [
            Output("costDistPlot", "figure"),
            Output("costDistPlot", "style"),
            Output("cost_validation_message", "children"),
        ],
        [Input("run_cost_sim", "n_clicks")],
        [
            State("low_cost", "value"),
            State("central_cost", "value"),
            State("high_cost", "value"),
            State("cost_dist_type", "value"),
        ],
    )
    def update_cost_simulation(n_clicks, low_cost, central_cost, high_cost, dist_type):
        num_sims = 10000  # Set default number of simulations to 10,000

        if n_clicks is None:
            return dash.no_update, {"display": "none"}, ""

        if low_cost is None or central_cost is None or high_cost is None:
            return dash.no_update, {"display": "none"}, "Please enter all cost values."

        if low_cost >= central_cost or central_cost >= high_cost:
            return (
                dash.no_update,
                {"display": "none"},
                "Ensure low < central < high for cost values.",
            )

        np.random.seed(42)  # Set seed for reproducibility

        if dist_type == "Uniform":
            results = uniform_1(low_cost, high_cost, num_sims)
        elif dist_type == "Normal (without central)":
            results = normal_2(low_cost, high_cost, num_sims)
        elif dist_type == "Normal (with central)":
            results = normal_3(low_cost, central_cost, high_cost, num_sims)
        elif dist_type == "Log-Normal":
            results = log_normal_4(low_cost, central_cost, high_cost, num_sims)

        # Calculate density curve
        density = gaussian_kde(results)
        x_vals = np.linspace(min(results), max(results), 1000)
        y_vals = density(x_vals)

        # Normalize histogram
        hist, bin_edges = np.histogram(results, bins=100, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=bin_centers, y=hist, name="Histogram", marker_color="blue", opacity=0.7
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Density Curve",
                line=dict(color="red"),
            )
        )

        fig.update_layout(
            title=f"{dist_type} Cost Distribution",
            xaxis_title="Total Cost",
            yaxis_title="Density",
            template="plotly_white",
            title_x=0.5,
        )

        return fig, {"display": "block"}, ""

    @app.callback(
        [
            Output("benefitDistPlot", "figure"),
            Output("benefitDistPlot", "style"),
            Output("benefit_validation_message", "children"),
        ],
        [Input("run_benefit_sim", "n_clicks")],
        [
            State("low_benefit", "value"),
            State("central_benefit", "value"),
            State("high_benefit", "value"),
            State("benefit_dist_type", "value"),
        ],
    )
    def update_benefit_simulation(
        n_clicks, low_benefit, central_benefit, high_benefit, dist_type
    ):
        num_sims = 10000  # Set default number of simulations to 10,000

        if n_clicks is None:
            return dash.no_update, {"display": "none"}, ""

        if low_benefit is None or central_benefit is None or high_benefit is None:
            return dash.no_update, {"display": "none"}, "Please enter all benefit values."

        if low_benefit >= central_benefit or central_benefit >= high_benefit:
            return (
                dash.no_update,
                {"display": "none"},
                "Ensure low < central < high for benefit values.",
            )

        np.random.seed(42)  # Set seed for reproducibility

        if dist_type == "Uniform":
            results = uniform_1(low_benefit, high_benefit, num_sims)
        elif dist_type == "Normal (without central)":
            results = normal_2(low_benefit, high_benefit, num_sims)
        elif dist_type == "Normal (with central)":
            results = normal_3(low_benefit, central_benefit, high_benefit, num_sims)
        elif dist_type == "Log-Normal":
            results = log_normal_4(low_benefit, central_benefit, high_benefit, num_sims)

        # Calculate density curve
        density = gaussian_kde(results)
        x_vals = np.linspace(min(results), max(results), 1000)
        y_vals = density(x_vals)

        # Normalize histogram
        hist, bin_edges = np.histogram(results, bins=100, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=bin_centers, y=hist, name="Histogram", marker_color="green", opacity=0.7
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Density Curve",
                line=dict(color="orange"),
            )
        )

        fig.update_layout(
            title=f"{dist_type} Benefit Distribution",
            xaxis_title="Total Benefit",
            yaxis_title="Density",
            template="plotly_white",
            title_x=0.5,
        )

        return fig, {"display": "block"}, ""

    @app.callback(
        [
            Output("bcrDistPlot", "figure"),
            Output("bcrDistPlot", "style"),
            Output("modal_point_value", "children"),
            Output("ci_value", "children"),
        ],
        [Input("calc_bcr", "n_clicks")],
        [State("costDistPlot", "figure"), State("benefitDistPlot", "figure")],
    )
    def calculate_bcr(n_clicks, cost_fig, benefit_fig):
        if n_clicks is None or cost_fig is None or benefit_fig is None:
            return dash.no_update, {"display": "none"}, "", ""

        # Extract cost and benefit results
        cost_hist = cost_fig["data"][0]["x"]
        benefit_hist = benefit_fig["data"][0]["x"]

        if not cost_hist or not benefit_hist:
            return dash.no_update, {"display": "none"}, "", ""

        cost_values = np.array(cost_hist)
        benefit_values = np.array(benefit_hist)

        # Calculate BCR
        bcr_values = benefit_values / cost_values

        # Calculate density curve
        density = gaussian_kde(bcr_values)
        x_vals = np.linspace(min(bcr_values), max(bcr_values), 1000)
        y_vals = density(x_vals)

        # Normalize histogram
        hist, bin_edges = np.histogram(bcr_values, bins=100, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Calculate modal point
        modal_point = x_vals[np.argmax(y_vals)]

        # Calculate 95% confidence interval
        lower_bound = np.percentile(bcr_values, 2.5)
        upper_bound = np.percentile(bcr_values, 97.5)

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=bin_centers, y=hist, name="Histogram", marker_color="purple", opacity=0.7
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Density Curve",
                line=dict(color="black"),
            )
        )

        fig.update_layout(
            title="Benefit-Cost Ratio (BCR) Distribution",
            xaxis_title="BCR",
            yaxis_title="Density",
            template="plotly_white",
            title_x=0.5,
        )

        # Update summary cards
        modal_text = f"{modal_point:.2f}"
        ci_text = f"{lower_bound:.2f} - {upper_bound:.2f}"

        return fig, {"display": "block"}, modal_text, ci_text
