# Poker Coach Streamlit App with Full Game Logic and Strategy
import streamlit as st
from treys import Card, Evaluator, Deck
import boto3
import os
from dotenv import load_dotenv
import json
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np

# Load env vars
load_dotenv("api-keys")
ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = os.getenv("REGION_NAME")

# AWS Bedrock client

def get_bedrock_client(
    runtime: Optional[bool] = True,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None
):
    if runtime:
        service_name = 'bedrock-runtime'
    else:
        service_name = 'bedrock'

    bedrock_runtime = boto3.client(
        service_name=service_name,
        region_name=REGION_NAME, 
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        aws_session_token=aws_session_token  # Optional
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_runtime._endpoint)
    return bedrock_runtime

bedrock = get_bedrock_client()

def call_bedrock(prompt: str, model_id: str = "anthropic.claude-instant-v1", max_tokens: int = 400) -> str:
    payload = {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": max_tokens,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 0.999,
        "stop_sequences": ["\n\nHuman:"]
    }
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(payload).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read().decode("utf-8"))
    return result.get("completion", "").strip()


# Game logic

evaluator = Evaluator()

# State
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 1000.0
if "history" not in st.session_state:
    st.session_state.history = []
if "hand" not in st.session_state:
    st.session_state.hand = {}

st.title("♠️ Poker Coach: Full Hand Strategy Trainer")

st.sidebar.header("💰 Bankroll Management")
st.sidebar.write(f"Current Bankroll: ${st.session_state.bankroll:.2f}")
st.sidebar.markdown("""
**What are Pot Odds?** Pot odds represent the ratio of the current size of the pot to the cost of a contemplated call. It helps you determine whether a call has positive expected value. For example, if the pot is 90 and you have to call 10, you're getting 9:1 pot odds, meaning you only need to win more than 10% of the time for a call to be profitable.

**What is Fold Equity?** Fold equity is the added value you get when your opponent folds to your bet. The more likely your opponent is to fold, the more profitable bluffing or semi-bluffing becomes.

**What is Expected Value (EV)?** EV is the average amount you can expect to win or lose with a certain play over the long run. Positive EV means profit over time, while negative EV suggests a losing play.
""")

# Setup
pot = st.number_input("Current pot size", 0.0, 100000.0, 100.0, 10.0)
num_players = st.slider("Number of players", 2, 10, 6)
position = st.selectbox("Your position", ["Small Blind", "Big Blind", "Early", "Middle", "Late"])
user_hand_input = st.text_input("Pre-flop Hand (e.g. 'Ah 10s' for Ace hearts and Ten spades)")

# --- Utility functions ---

def parse_hand(text):
    cards = text.strip().split()
    normalized = []
    used_cards = set()

    for card in cards:
        card = card.strip().lower()

        if card.startswith("10") and len(card) == 3:
            rank = 'T'
            suit = card[2]
        elif len(card) == 2:
            rank = card[0].upper()
            suit = card[1]
        else:
            raise ValueError(f"Card '{card}' must be 2 or 3 characters like 'Ah', '10s'.")

        if rank not in "23456789TJQKA" or suit not in "shdc":
            raise ValueError(f"Invalid card: '{card}'")

        card_str = rank + suit
        try:
            parsed = Card.new(card_str)
        except Exception:
            raise ValueError(f"Card.new() failed to parse '{card_str}'")

        # Confirm round-trip matches input
        if Card.int_to_str(parsed).lower() != card_str.lower():
            raise ValueError(f"Treys parsed '{card_str}' incorrectly (got '{Card.int_to_str(parsed)}')")

        if parsed in used_cards:
            raise ValueError(f"Duplicate card '{card_str}' entered.")

        used_cards.add(parsed)
        normalized.append(parsed)

    return normalized

def simulate_odds(user_hand, known_board, num_opponents, iterations=500):
    win = tie = 0
    for _ in range(iterations):
        deck = Deck()
        for card in user_hand + known_board:
            deck.cards.remove(card)
        board = known_board + deck.draw(5 - len(known_board))
        villains = [deck.draw(2) for _ in range(num_opponents)]
        if any(len(v) < 2 for v in villains):
            raise ValueError("Deck ran out of cards during simulation. Reduce the number of opponents or iterations.")
        user_score = evaluator.evaluate(board, user_hand)
        scores = [evaluator.evaluate(board, v) for v in villains]
        all_scores = scores + [user_score]
        if user_score == min(all_scores):
            if all_scores.count(user_score) > 1:
                tie += 1
            else:
                win += 1
    return (win / iterations * 100, tie / iterations * 100)

# --- Preflop ---

def plot_ev_chart(win_pct):
    ev_values = [((win_pct / 100) * pot - (1 - win_pct / 100) * pot / 2) for pot in range(10, 110, 10)]
    plt.figure()
    plt.plot(range(10, 110, 10), ev_values, marker='o')
    plt.xlabel('Pot Size')
    plt.ylabel('Expected Value ($)')
    plt.title('Expected Value vs Pot Size')
    st.pyplot(plt)

if user_hand_input:
    try:
        user_hand = parse_hand(user_hand_input)
        preflop_win_pct, tie_pct = simulate_odds(user_hand, [], num_players - 1)
        dummy_board = Deck().draw(5)
        hand_score = evaluator.evaluate(dummy_board, user_hand)
        strength = evaluator.class_to_string(evaluator.get_rank_class(hand_score)) + f" (score: {hand_score})"

        # Improved decision logic based on EV thresholding and estimated fold equity
        pot_odds = 1 / (num_players + 1)
        ev_threshold = 0.5 * pot_odds * 100  # Pot odds as baseline expectation

#        if preflop_win_pct > 60:
#            action = "Raise"
#        elif preflop_win_pct > ev_threshold:
#            action = "Call"
#        elif 15 < preflop_win_pct <= ev_threshold:
#            action = "Bluff"
#        else:
#            action = "Fold"

        st.session_state.hand = {
            "user_hand": user_hand,
            "board": [],
            "num_players": num_players,
            "position": position,
            "preflop_win": preflop_win_pct,
            "pot": pot,
            "stage": "flop"
        }

        prompt = f"You are an aggressive Game Theory Optimal (GTO) poker coach with deep knowledge of exploitative and optimal strategies. The user holds {user_hand_input} in {position} position at a {num_players}-handed table. The pre-flop win rate is {preflop_win_pct:.2f}%. Given this, provide a technically grounded recommendation to Raise, Call, Bluff, or Fold. Justify the action using GTO concepts such as hand range dominance, position equity, fold equity, and expected value (EV). Also evaluate if this is a good spot for a bluff based on the user's image and position."
        explanation = call_bedrock(prompt)
        st.write(f"Preflop Odds: {preflop_win_pct:.2f}%, Tie: {tie_pct:.2f}%, Strength: {strength}")
#        st.success(f"Coach Recommendation: {action}")
        st.info(explanation)
        plot_ev_chart(preflop_win_pct)

    except Exception as e:
        st.error(f"Error: {e}")

# --- Flop ---
if st.session_state.hand.get("stage") == "flop":
    st.subheader("🪙 Update Pot Size")
    pot = st.number_input("Enter pot size after preflop:", min_value=0.0, step=10.0, value=pot)
    flop_input = st.text_input("Enter Flop (e.g. '7d Jc 2h')")
    if flop_input:
        try:
            board = parse_hand(flop_input)
            st.session_state.hand["board"] = board
            win_pct, tie_pct = simulate_odds(st.session_state.hand["user_hand"], board, num_players - 1)
            st.write(f"Flop Win %: {win_pct:.2f}%")
#           action = "Raise" if win_pct > 60 else ("Call" if win_pct > 30 else "Fold")
            prompt = f"User has {user_hand_input} in {position} position. Pot after flop is {pot} Flop is {flop_input}. Win chance: {win_pct:.2f}%. Pot: ${pot}. Provide a technically sound recommendation to Raise, Call, or Fold. Justify with concepts like range interaction, board texture, fold equity, and expected value. Is this a spot for semi-bluffing based on the board dynamics?"
            st.session_state.hand["stage"] = "turn"
#            st.success(f"Coach Recommendation: {action}")
            st.info(call_bedrock(prompt))
        except Exception as e:
            st.error(f"Error: {e}")
        plot_ev_chart(win_pct)

# --- Turn ---
if st.session_state.hand.get("stage") == "turn":
    st.subheader("🪙 Update Pot Size")
    pot = st.number_input("Enter pot size after flop:", min_value=0.0, step=10.0, value=pot)
    turn_card = st.text_input("Enter Turn (e.g. 'Qc')")
    if turn_card:
        try:
            st.session_state.hand["board"].append(Card.new(turn_card))
            win_pct, tie_pct = simulate_odds(st.session_state.hand["user_hand"], st.session_state.hand["board"], num_players - 1)
            st.write(f"Turn Win %: {win_pct:.2f}%")
#            action = "Raise" if win_pct > 70 else ("Call" if win_pct > 35 else "Fold")
            board_str = ' '.join(Card.int_to_str(c) for c in st.session_state.hand['board'])
            prompt = f"User has {user_hand_input} on turn. Board: {board_str}. Pot: ${pot}. Win %: {win_pct:.2f}%. Provide a detailed technical recommendation using hand strength vs range, pot odds, and bluff equity. Should the user semi-bluff or slowplay?"
            st.session_state.hand["stage"] = "river"
#            st.success(f"Coach Recommendation: {action}")
            st.info(call_bedrock(prompt))
        except Exception as e:
            st.error(f"Error: {e}")
        plot_ev_chart(win_pct)

# --- River ---
if st.session_state.hand.get("stage") == "river":
    st.subheader("🪙 Update Pot Size")
    pot = st.number_input("Enter pot size after turn:", min_value=0.0, step=10.0, value=pot)
    river_card = st.text_input("Enter River (e.g. 'Th')")
    if river_card:
        try:
            st.session_state.hand["board"].append(Card.new(river_card))
            win_pct, tie_pct = simulate_odds(st.session_state.hand["user_hand"], st.session_state.hand["board"], num_players - 1)
            final_board = ' '.join(Card.int_to_str(c) for c in st.session_state.hand["board"])
            st.write(f"River Win %: {win_pct:.2f}%")
#            action = "All-in" if win_pct > 85 else ("Raise" if win_pct > 60 else ("Bluff" if win_pct < 20 else "Check/Fold"))
            prompt = f"User's hand: {user_hand_input}. Final board: {final_board}. Final pot size is {pot}. Win %: {win_pct:.2f}%. Pot: ${pot}. Provide a technical recommendation for post-river play. Consider opponent ranges, bet sizing, bluff catching, and whether this is a profitable bluff spot. Justify using GTO principles and EV calculations."
#            st.success(f"Coach Recommendation: {action}")
            st.info(call_bedrock(prompt))
            plot_ev_chart(win_pct)
            
            outcome = st.radio("Did you win the hand?", ["Won", "Lost", "Folded"])
            change = st.number_input("Change in bankroll for this hand", -10000.0, 10000.0, 0.0)
            if st.button("Submit Result"):
                st.session_state.bankroll += change
                st.session_state.history.append({"hand": user_hand_input, "result": outcome, "change": change})
                st.success("Hand logged. Ready for next round.")
                st.session_state.hand = {}

        except Exception as e:
            st.error(f"Error: {e}")
