from collections import Counter
from dataclasses import dataclass


def score(state):
    score = 0
    
    upper_section_score = 0
    upper_section_categories = [
        compute_counts_score,
    ]

    for f in upper_section_categories:
        upper_section_score += f(state)

    upper_section_score += compute_bonus(upper_section_score)

    lower_section_score = 0
    lower_section_categories = [

    ]

    for f in lower_section_categories:
        lower_section_score += f(state)
    
    return upper_section_score + lower_section_score

def compute_bonus(upper_section_score):
    return 35 if upper_section_score >= 63 else 0

# Upper Section



