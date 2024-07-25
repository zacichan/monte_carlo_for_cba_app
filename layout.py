from dash import dcc, html
import dash_bootstrap_components as dbc


def create_layout():
    return dbc.Container(
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
                                id="cost_validation_message",
                                className="text-danger mb-2",
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
                                            id="modal_point_value",
                                            className="card-text",
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
