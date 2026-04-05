# -*- coding: utf-8 -*-
"""English display titles and student-facing advice per predicted obesity class (API + UI)."""

LEVEL_TITLE_EN = {
    "Insufficient_Weight": "Insufficient weight",
    "Normal_Weight": "Normal weight",
    "Overweight_Level_I": "Overweight (level I)",
    "Overweight_Level_II": "Overweight (level II)",
    "Obesity_Type_I": "Obesity (type I)",
    "Obesity_Type_II": "Obesity (type II)",
    "Obesity_Type_III": "Obesity (type III)",
}

LEVEL_ADVICE_EN = {
    "Insufficient_Weight": (
        "Your result suggests body weight may be lower than what is healthy for your height. "
        "Please do not skip meals to lose weight, and talk with a parent, guardian, or school nurse "
        "about balanced meals and snacks. If you often feel tired or dizzy, it is important to ask "
        "a health professional for advice. Building steady, healthy eating habits now helps you grow "
        "and learn at your best."
    ),
    "Normal_Weight": (
        "Great job! You seem to be keeping a balanced lifestyle. Keep eating regular meals with fruit "
        "and vegetables, staying active, and limiting long stretches of screen time. Small habits every "
        "day (sleep, movement, and mindful snacking) add up. Keep it up: you are on a healthy track!"
    ),
    "Overweight_Level_I": (
        "This pattern suggests a little extra weight compared with healthy growth for many students. "
        "Try smaller portions of sugary drinks and ultra-processed snacks, add one more active break "
        "each day (walking, sports, or play), and keep screens from replacing sleep. Sticking with these "
        "habits over time can lower future health risks. Ask your family or PE teacher for ideas you "
        "can stick with."
    ),
    "Overweight_Level_II": (
        "Your answers point to habits that may raise weight-related health risks if they continue for "
        "years. Focus on regular meals, more water instead of sweet drinks, and daily movement you enjoy. "
        "Long hours sitting and frequent high-calorie snacks can slowly push weight upward and later "
        "affect blood pressure or blood sugar. Small steady changes now are easier than big crash plans; "
        "consider talking with a trusted adult about a realistic plan."
    ),
    "Obesity_Type_I": (
        "This level suggests obesity-related risk tied to everyday habits. Eating patterns, screen time, "
        "and low activity can, over time, strain your heart, joints, and metabolism. Please take this "
        "seriously: work with your family on meal planning, cut back on sugary snacks and fast food, "
        "and aim for consistent physical activity. A school nurse or doctor can help you set safe, "
        "age-appropriate goals. Early changes can prevent more serious problems later."
    ),
    "Obesity_Type_II": (
        "The result indicates a higher level of obesity risk from your current lifestyle pattern. "
        "If similar habits continue, the chance of problems such as type 2 diabetes, high blood pressure, "
        "sleep issues, and joint pain can increase, not only in adulthood but sometimes during the teen "
        "years. You deserve support: talk openly with parents or caregivers and ask a clinician for "
        "guidance. Sustainable changes to food choices, sleep, and daily movement matter more than "
        "short-term dieting."
    ),
    "Obesity_Type_III": (
        "This is the most serious category in this survey: your answers suggest very high obesity-related "
        "risk. Continuing the same lifestyle can strongly affect long-term health, including heart disease, "
        "diabetes, and mobility. Please reach out to a doctor or specialist as soon as possible; they can "
        "help with safe nutrition, activity, and emotional support. This tool is only a classroom model and "
        "not a diagnosis, but it is a signal to take action with trusted adults and professionals."
    ),
}


def advice_for_level(level_key: str) -> tuple[str, str]:
    """Return (title, advice paragraph). Fallback if key unknown."""
    k = str(level_key)
    title = LEVEL_TITLE_EN.get(k, k.replace("_", " "))
    advice = LEVEL_ADVICE_EN.get(
        k,
        "Please discuss these results with a parent, teacher, or health professional for guidance suited to you.",
    )
    return title, advice
