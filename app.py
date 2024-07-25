import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm, gaussian_kde

# Set up the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Monte Carlo Cost-Benefit Analysis",
                            className="text-center my-4",
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [
                                        html.P(
                                            "This app performs Monte Carlo simulations to analyze cost-benefit scenarios. It allows users to input low, central, and high estimates for both costs and benefits, and then simulates distributions for each."
                                        ),
                                        html.H5("Cost Distribution:"),
                                        html.P(
                                            "For costs, we simulate a distribution based on the selected type (Uniform, Normal, or Log-Normal):"
                                        ),
                                        html.P("Uniform: X ~ U(low, high)"),
                                        html.P(
                                            "Normal (without central): X ~ N((high + low) / 2, (high - low) / 4)"
                                        ),
                                        html.P(
                                            "Normal (with central): X ~ N(central, (high - low) / 4)"
                                        ),
                                        html.P(
                                            "Log-Normal: X ~ LogN(mu, sigma), where mu and sigma are calculated based on the central value such that 95% of values lie within [low, high]."
                                        ),
                                        html.H5("Benefit Distribution:"),
                                        html.P(
                                            "For benefits, the process is similar to costs with the same distribution options."
                                        ),
                                        html.H5("Benefit-Cost Ratio (BCR):"),
                                        html.P(
                                            "The BCR is calculated as: BCR = Benefit / Cost, where the distributions of benefit and cost are used to generate the distribution of the BCR."
                                        ),
                                    ],
                                    title="How the Calculations Work",
                                )
                            ],
                            start_collapsed=True,
                        )
                    ]
                )
            ]
        ),
        html.Br(),  # Add space between accordion and the first dropdown
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id="option_selector",
                            options=[
                                {"label": "Option 1", "value": "option_1"},
                                {"label": "Option 2", "value": "option_2"},
                                {"label": "Option 3", "value": "option_3"},
                            ],
                            value="option_1",  # Set default value to Option 1
                            className="mb-2",
                        )
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Cost Inputs"),
                        dbc.Input(
                            id="low_cost",
                            type="number",
                            value=10,
                            placeholder="Low Cost",
                            step=0.01,
                            className="mb-2",
                        ),
                        dbc.Input(
                            id="central_cost",
                            type="number",
                            value=50,
                            placeholder="Central Cost",
                            step=0.01,
                            className="mb-2",
                        ),
                        dbc.Input(
                            id="high_cost",
                            type="number",
                            value=200,
                            placeholder="High Cost",
                            step=0.01,
                            className="mb-2",
                        ),
                        dcc.Dropdown(
                            id="cost_dist_type",
                            options=[
                                {"label": "Uniform", "value": "Uniform"},
                                {
                                    "label": "Normal (without central)",
                                    "value": "Normal (without central)",
                                },
                                {
                                    "label": "Normal (with central)",
                                    "value": "Normal (with central)",
                                },
                                {"label": "Log-Normal", "value": "Log-Normal"},
                            ],
                            value="Log-Normal",  # Set default value to Log-Normal
                            className="mb-2",
                        ),
                        dbc.Button(
                            "Run Cost Simulation",
                            id="run_cost_sim",
                            color="primary",
                            className="mb-2",
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H3("Benefit Inputs"),
                        dbc.Input(
                            id="low_benefit",
                            type="number",
                            value=20,
                            placeholder="Low Benefit",
                            step=0.01,
                            className="mb-2",
                        ),
                        dbc.Input(
                            id="central_benefit",
                            type="number",
                            value=100,
                            placeholder="Central Benefit",
                            step=0.01,
                            className="mb-2",
                        ),
                        dbc.Input(
                            id="high_benefit",
                            type="number",
                            value=400,
                            placeholder="High Benefit",
                            step=0.01,
                            className="mb-2",
                        ),
                        dcc.Dropdown(
                            id="benefit_dist_type",
                            options=[
                                {"label": "Uniform", "value": "Uniform"},
                                {
                                    "label": "Normal (without central)",
                                    "value": "Normal (without central)",
                                },
                                {
                                    "label": "Normal (with central)",
                                    "value": "Normal (with central)",
                                },
                                {"label": "Log-Normal", "value": "Log-Normal"},
                            ],
                            value="Log-Normal",  # Set default value to Log-Normal
                            className="mb-2",
                        ),
                        dbc.Button(
                            "Run Benefit Simulation",
                            id="run_benefit_sim",
                            color="primary",
                            className="mb-2",
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id="cost_validation_message", className="text-danger mb-2"
                        )
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id="benefit_validation_message",
                            className="text-danger mb-2",
                        )
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id="costDistPlot", style={"display": "none"}
                        )  # Initially hidden
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id="benefitDistPlot", style={"display": "none"}
                        )  # Initially hidden
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Calculate BCR",
                            id="calc_bcr",
                            color="success",
                            className="mb-2",
                        )
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id="bcrDistPlot", style={"display": "none"}
                        )  # Initially hidden
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Modal Point", className="card-title"),
                                    html.P(
                                        id="modal_point_value", className="card-text"
                                    ),
                                ]
                            ),
                            className="mb-2",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5(
                                        "95% Confidence Interval",
                                        className="card-title",
                                    ),
                                    html.P(id="ci_value", className="card-text"),
                                ]
                            ),
                            className="mb-2",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
    ]
)

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


# Callback to update the input fields based on the selected option
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


# Define callback to update cost plot
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


# Define callback to update benefit plot
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


# Define callback to update BCR plot
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


if __name__ == "__main__":
    app.run_server(debug=True)
