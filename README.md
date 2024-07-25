# Monte Carlo Cost-Benefit Analysis

This Streamlit application performs Monte Carlo simulations to analyze cost-benefit scenarios for different options. Users can input low, central, and high estimates for costs and benefits and choose from various distribution types to model these uncertainties. The app generates random samples to simulate potential outcomes and calculates the Benefit-Cost Ratio (BCR) for each option, providing statistical insights through visualisations and summaries.

## Features

- **Input Configuration**: Define cost and benefit estimates for different options.
- **Distribution Selection**: Choose from Uniform, Normal (with and without central), and Log-Normal distributions.
- **Monte Carlo Simulation**: Generate a large number of random samples to simulate outcomes.
- **Statistical Analysis**: Calculate and display modal points and 95% confidence intervals for cost, benefit, and BCR distributions.
- **Interactive Visualizations**: View detailed density plots for cost, benefit, and BCR distributions.

The app is hosted on streamlit cloud and should be available at this [link](https://montecarloforcbaapp-xsu4ple9usmhrq6xrevewo.streamlit.app/)

## Developers

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/monte-carlo-cba.git
    cd monte-carlo-cba
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:

    ```bash
    streamlit run streamlit/main.py
    ```

## Usage

1. **Launch the application** by running the Streamlit app. Or access via the [link](https://montecarloforcbaapp-xsu4ple9usmhrq6xrevewo.streamlit.app/).
2. **Navigate to the sidebar** to configure inputs:
   - Select an option from the available options.
   - Enter low, central, and high estimates for cost and benefit.
   - Choose the distribution type for cost and benefit.
   - Click the "Run Simulation" button.
3. **View the Results Summary** for a concise overview of the simulations.
4. **Explore Detailed Distributions** by expanding the detailed sections.

## Methodology

The Monte Carlo Cost-Benefit Analysis uses the following methodology:

1. **Modeling Cost and Benefit Distributions**:
   - Each option's cost and benefit are represented by specific probability distributions (e.g., Uniform, Normal, Log-Normal).
   - Distribution parameters are determined based on user-provided low, central, and high estimates.

2. **Simulation Process**:
   - Generate a large number of random samples (e.g., 10,000) from the specified cost and benefit distributions to simulate potential outcomes.

3. **Calculating Benefit-Cost Ratio (BCR)**:
   - For each simulated sample, compute the Benefit-Cost Ratio (BCR) by dividing the sampled benefit by the sampled cost:
     \[
     \text{BCR} = \frac{\text{Benefit}}{\text{Cost}}
     \]

4. **Statistical Analysis of Results**:
   - Determine key statistical metrics for the cost, benefit, and BCR distributions, including:
     - **Modal Point**: The most frequent (mode) or average (mean) value.
     - **95% Confidence Interval (CI)**: The range within which 95% of the simulated values fall, providing an estimate of uncertainty.

## Project Structure

- `streamlit/main.py`: Main Streamlit application file.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation (this file).

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and passes all tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the contributors and the Streamlit community for their support and resources.

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Monte Carlo Simulation](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Cost-Benefit Analysis](https://en.wikipedia.org/wiki/Cost%E2%80%93benefit_analysis)

