# PokerCoach_FinalProject
GTO poker coach for Gen AI (GSB570) final project
Author: Jack Muoio
Date: 06/05/2025

Youtube: https://youtu.be/A8lCOY13LgM?si=NZeL8g-5lECIKh0C

Poker Coach: Full Hand Strategy Trainer

Overview:
This Streamlit-based application serves as an interactive poker coaching tool, guiding users through each stage of a Texas Hold'em hand—preflop, flop, turn, and river. It combines Monte Carlo simulations for win probability estimation with real-time strategic advice generated by a large language model (LLM) via AWS Bedrock. The app also includes bankroll tracking and visualizations of expected value (EV) to aid in decision-making.

Features:
- User Input: Enter your hand, table position, number of players, and pot size.
- Win Probability Simulation: Estimates your chances of winning against random opponent hands using Monte Carlo simulations.
- LLM-Based Coaching: Provides strategic recommendations based on Game Theory Optimal (GTO) principles, considering factors like position, hand strength, and board texture.
- EV Visualization: Displays a chart showing expected value across different pot sizes.
- Bankroll Management: Tracks your bankroll over multiple hands, allowing you to log outcomes and financial changes.

Usage:
1. Launch the app using Streamlit (in VS Code terminal run: streamlit run /File/Location/.py).
2. Input your preflop hand (e.g., 'Ah 10s'), position, number of players, and current pot size.
3. The app simulates your win probability and provides LLM-based strategic advice.
4. As the hand progresses, input the flop, turn, and river cards when prompted.
5. After each stage, the app updates your win probability, offers new strategic insights, and displays an EV chart.
6. At the end of the hand, log the outcome and any changes to your bankroll.

Installation:
1. Clone the repository.
2. Install the required packages using pip.
3. Set up your AWS credentials and region in a .env file.
4. Run the app with Streamlit.

Note:
- Ensure you have valid AWS credentials with access to Bedrock services.
- The app uses the 'treys' library for hand evaluation and 'matplotlib' for plotting.

Disclaimer:
This tool is intended for educational and entertainment purposes. It does not guarantee success in real-money poker games.
