import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from state import State

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
    # Convert the styled DataFrame to HTML with specific styling
    html_table = (
        df.style.map(
            lambda x: "background-color: #2E8B57"
            if isinstance(x, (int, float)) and x != 0
            else "background-color: black",
            subset=["Scores", "Current Dice Scores", "Selected"],
        )
        .format("{:.0f}", subset=["Scores", "Current Dice Scores", "Selected"])
        .to_html()
    )

    # Wrap the table in a div with custom styling
    styled_html = f"""
    <div style="font-size: 0.9em; line-height: 1.2;">
        {html_table}
    </div>
    """
    return styled_html


def create_dice_histogram(dice_histogram, dice_values):
    # Create figure using plotly
    fig = go.Figure(
        data=[
            go.Bar(
                x=[str(i + 1) for i in range(6)],
                y=dice_histogram,
                text=dice_histogram,
                textposition="auto",
            )
        ]
    )

    # Update layout with more reasonable dimensions and better text contrast
    fig.update_layout(
        title={"text": "Dice Histogram", "font": {"color": "white", "size": 20}},
        xaxis_title={"text": "Dice Value", "font": {"color": "white", "size": 14}},
        yaxis_title={"text": "Count", "font": {"color": "white", "size": 14}},
        xaxis={"tickfont": {"color": "white"}},  # X-axis tick labels
        yaxis={"tickfont": {"color": "white"}},  # Y-axis tick labels
        width=350,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    # Update bar appearance with white text on bars
    fig.update_traces(marker_color="#4169E1", textfont={"size": 14, "color": "white"})

    return fig


def display_scoresheet(state: State):
    # Calculate roll number (3 - rolls_remaining)
    roll_number = 3 - state.rolls_remaining.item()
    round_number = state.round_index.item() + 1  # Add 1 for human-readable round number

    # Create info text for roll and round
    game_info = f"<h3>Round: {round_number}/13 | Roll: {roll_number}/3</h3>"

    # Create dice display
    dice_values = state.dice_state.squeeze().tolist()
    dice_histogram = state.dice_histogram.squeeze().tolist()

    # Generate histogram plot
    histogram_plot = create_dice_histogram(dice_histogram, dice_values)

    # Format dice display as HTML with more vertical padding
    dice_html = f"""
    <div style='margin-top: 40px; margin-bottom: 40px; padding: 20px 0;'>
        <h4>Current Dice:</h4>
        <div style='display: flex; flex-direction: column; gap: 40px;'>
            
            <!-- Dice values section -->
            <div>
                <h5>Dice Values:</h5>
                <div style='display: flex; gap: 10px;'>
                    {
        "".join(
            [
                f'<div style="width: 40px; height: 40px; background-color: blue; border: 1px solid black; display: flex; justify-content: center; align-items: center; font-size: 20px; font-weight: bold; color: white;">{int(die)}</div>'
                for die in dice_values
            ]
        )
    }
                </div>
            </div>
        </div>
    </div>
    """

    return game_info, dice_html, histogram_plot


def setup_app(initial_state):
    # Get initial values
    game_info, dice_html, histogram_plot = display_scoresheet(initial_state)
    scoresheet_html = create_yahtzee_scoresheet(initial_state)

    gr.Markdown("# Yahtzee Scoresheet")
    with gr.Row():
        # Left column for scoresheet
        with gr.Column(scale=3):
            output_info = gr.HTML(value=game_info)
            output_scoresheet = gr.HTML(value=scoresheet_html)  # Changed to HTML
        # Right column for dice and histogram
        with gr.Column(scale=2):
            output_dice = gr.HTML(value=dice_html)
            output_histogram = gr.Plot(value=histogram_plot)
            next_action_btn = gr.Button("Next Action")
    return (
        output_info,
        output_scoresheet,
        output_dice,
        output_histogram,
        next_action_btn,
    )


def main():
    """
    Main entry point for the Yahtzee application.
    Initializes and launches the Gradio interface.
    """
    game_model = Yahtzee(1)
    states, actions, values, rewards = game_model()
    initial_state = states[0]

    with gr.Blocks(theme=gr.themes.Default()) as demo:
        outputs = setup_app(initial_state)
        (
            output_info,
            output_scoresheet,
            output_dice,
            output_histogram,
            next_action_btn,
        ) = outputs

        # Create a state variable to track the index
        index = gr.State(value=0)

        # Update the scoresheet each time the button is clicked
        def refresh_scoresheet(current_index):
            next_index = current_index + 1
            state = states[current_index]
            if next_index >= len(states):
                next_index = 0
                gr.Warning("Reached the end of game states. Starting over.")

            game_info, dice_html, histogram_plot = display_scoresheet(state)
            scoresheet_html = create_yahtzee_scoresheet(state)  # Get HTML version

            return game_info, scoresheet_html, dice_html, histogram_plot, next_index

        next_action_btn.click(
            fn=refresh_scoresheet,
            inputs=index,
            outputs=[
                output_info,
                output_scoresheet,
                output_dice,
                output_histogram,
                index,
            ],
        )

    demo.launch(show_error=True)


if __name__ == "__main__":
    main()
