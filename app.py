import gradio as gr
import pandas as pd

def create_yahtzee_scoresheet(num):
    # Create a DataFrame with Yahtzee scoring categories
    categories = [
        'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
        'Upper Section Bonus', 'Three of a Kind', 'Four of a Kind',
        'Full House', 'Small Straight', 'Large Straight', 'Yahtzee',
        'Chance', 'Yahtzee Bonus', 'Total Score'
    ]
    
    df = pd.DataFrame({
        'Category': categories,
        'Description': [
            'Sum of all ones', 'Sum of all twos', 'Sum of all threes',
            'Sum of all fours', 'Sum of all fives', 'Sum of all sixes',
            '35 points if upper section ≥ 63',
            'Sum of all dice if 3+ of one number',
            'Sum of all dice if 4+ of one number',
            '25 points for 3 of one number and 2 of another',
            '30 points for sequence of 4',
            '40 points for sequence of 5',
            '50 points for 5 of a kind',
            'Sum of all dice',
            '100 points per extra Yahtzee',
            'Sum of all scores'
        ],
        "Scores": [num for _ in categories]
    })
    return df

def display_scoresheet(num):
    return create_yahtzee_scoresheet(num).to_html()

with gr.Blocks() as demo:
    gr.Markdown("# Yahtzee Scoresheet")
    output = gr.HTML()
    refresh_btn = gr.Button("Show Scoresheet")
    
    # Initialize the scoresheet when the app starts
    output.value = display_scoresheet(1)
    
    # Update the scoresheet each time the button is clicked
    def refresh_scoresheet(num):
        return display_scoresheet(num)
    
    refresh_btn.click(fn=refresh_scoresheet, inputs=gr.Number(), outputs=output)

demo.launch()