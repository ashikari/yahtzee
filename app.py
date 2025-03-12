import gradio as gr
import pandas as pd

from policy_model import State

from game_model import Yahtzee


def create_yahtzee_scoresheet(num):
    # Create a DataFrame with Yahtzee scoring categories
    categories = [
        "Ones",
        "Twos",
        "Threes",
        "Fours",
        "Fives",
        "Sixes",
        "Upper Section Bonus",
        "Three of a Kind",
        "Four of a Kind",
        "Full House",
        "Small Straight",
        "Large Straight",
        "Yahtzee",
        "Chance",
        "Yahtzee Bonus",
        "Total Score",
    ]

    df = pd.DataFrame(
        {
            "Category": categories,
            "Description": [
                "Sum of all ones",
                "Sum of all twos",
                "Sum of all threes",
                "Sum of all fours",
                "Sum of all fives",
                "Sum of all sixes",
                "35 points if upper section ≥ 63",
                "Sum of all dice if 3+ of one number",
                "Sum of all dice if 4+ of one number",
                "25 points for 3 of one number and 2 of another",
                "30 points for sequence of 4",
                "40 points for sequence of 5",
                "50 points for 5 of a kind",
                "Sum of all dice",
                "100 points per extra Yahtzee",
                "Sum of all scores",
            ],
            "Selected": [num for _ in categories],
            "Scores": [num for _ in categories],
            "Current Dice Scores": [num for _ in categories],
        }
    )
    return df


def display_scoresheet(state: State):
    return create_yahtzee_scoresheet(state.round_index.item()).to_html()


def setup_app():
    gr.Markdown("# Yahtzee Scoresheet")
    output = gr.HTML()
    refresh_btn = gr.Button("Show Scoresheet")
    return output, refresh_btn


def main():
    """
    Main entry point for the Yahtzee application.
    Initializes and launches the Gradio interface.
    """
    game_model = Yahtzee(1)

    states = game_model()

    with gr.Blocks() as demo:
        output, refresh_btn = setup_app()

        # Create a state variable to track the index
        index = gr.State(value=1)

        # Initialize the scoresheet when the app starts
        output.value = display_scoresheet(states[0])

        # Update the scoresheet each time the button is clicked
        def refresh_scoresheet(current_index):
            next_index = current_index + 1
            state = states[current_index]
            return display_scoresheet(state), next_index

        refresh_btn.click(fn=refresh_scoresheet, inputs=index, outputs=[output, index])

    demo.launch()


if __name__ == "__main__":
    main()
