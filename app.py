import gradio as gr
import pandas as pd

from policy_model import State

from game_model import Yahtzee


def create_yahtzee_scoresheet(state: State):
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
            "Selected": [
                *state.upper_section_used.squeeze().tolist(),
                0,
                *state.lower_section_used.squeeze().tolist(),
                0,
                0,
            ],
            "Scores": [
                *state.upper_section_scores.squeeze().tolist(),
                state.upper_bonus.item(),
                *state.lower_section_scores.squeeze().tolist(),
                state.lower_bonus.item(),
                state.total_score.item(),
            ],
            "Current Dice Scores": [
                *state.upper_section_current_dice_scores.squeeze().tolist(),
                0,  # No current dice score for upper bonus
                *state.lower_section_current_dice_scores.squeeze().tolist(),
                0,  # No current dice score for Yahtzee bonus
                0,  # No current dice score for total
            ],
        }
    )
    # Apply styling to highlight non-zero values and format as integers
    styled_df = df.style.map(
        lambda x: "background-color: #90EE90"
        if isinstance(x, (int, float)) and x != 0
        else "background-color: black",
        subset=["Scores", "Current Dice Scores", "Selected"],
    ).format("{:.0f}", subset=["Scores", "Current Dice Scores", "Selected"])

    return styled_df


def display_scoresheet(state: State):
    # Calculate roll number (3 - rolls_remaining)
    roll_number = 3 - state.rolls_remaining.item()
    round_number = state.round_index.item() + 1  # Add 1 for human-readable round number

    # Create info text for roll and round
    game_info = f"<h3>Round: {round_number}/13 | Roll: {roll_number}/3</h3>"

    # Create dice display
    dice_values = state.dice_state.squeeze().tolist()
    dice_histogram = state.dice_histogram.squeeze().tolist()

    # Format dice display as HTML
    dice_html = f"""
    <div style='margin-top: 20px; margin-bottom: 20px;'>
        <h4>Current Dice:</h4>
        <div style='display: flex; flex-direction: column; gap: 20px;'>
            
            <!-- Dice values section -->
            <div>
                <h5>Dice Values:</h5>
                <div style='display: flex; gap: 10px;'>
                    {
        "".join(
            [
                f'<div style="width: 40px; height: 40px; background-color: blue; border: 1px solid black; display: flex; justify-content: center; align-items: center; font-size: 20px; font-weight: bold;">{int(die)}</div>'
                for die in dice_values
            ]
        )
    }
                </div>
            </div>
            
            <!-- Dice histogram section -->
            <div>
                <h5>Dice Histogram:</h5>
                <div style='display: flex; gap: 10px;'>
                    {
        "".join(
            [
                f'''
                    <div style='display: flex; flex-direction: column; align-items: center;'>
                        <div style='width: 30px; height: 30px; background-color: blue; border: 1px solid black; display: flex; justify-content: center; align-items: center;'>{int(count)}</div>
                        <div>{i + 1}</div>
                    </div>'''
                for i, count in enumerate(dice_histogram)
            ]
        )
    }
                </div>
            </div>
            
        </div>
    </div>
    """

    # Wrap the scoresheet and dice display in a flex container
    combined_html = f"""
    {game_info}
    <div style='display: flex; gap: 20px; align-items: flex-start;'>
        <div style='flex: 2;'>
            {create_yahtzee_scoresheet(state).to_html()}
        </div>
        <div style='flex: 1;'>
            {dice_html}
        </div>
    </div>
    """
    return combined_html


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
            # Check if we've reached the end of the states list
            if next_index >= len(states):
                next_index = 0  # Reset to beginning
                gr.Warning("Reached the end of game states. Starting over.")

            combined_html = display_scoresheet(state)

            return combined_html, next_index

        refresh_btn.click(fn=refresh_scoresheet, inputs=index, outputs=[output, index])

    demo.launch()


if __name__ == "__main__":
    main()
