# Part 2: Instruction templates
instruction_templates = [
    "{Action_capitalized} the {target} in the {location} that is {relation} the {reference}.",
    "Go to the {location} and {action} the {target} you find {relation} the {reference}.",
    "Could you please {action} the {target} located {relation} the {reference} in the {location}?",
    "The {target} {relation} the {reference} in the {location} needs your attention, please {action} it.",
    "First, head to the {location}. Once there, please {action} the {target} sitting {relation} the {reference}.",
    "In the {location}, near the {reference}, {action} the {target}.",
    "There is a {target} in the {location}. It's the one {relation} the {reference}. Please {action} it.",
    "I need your help. In the {location}, can you find the {reference} and then {action} the {target} {relation} it?",
    "Find the {reference} in the {location}, then {action} the {target} positioned {relation} it.",
    "Please go to the {location} and {action} the {target} by the {reference}."
]

smart_templates = [
    # Template 1: Uses third-person singular form
    "{Action_capitalized} the {target} in the {location} that {rel_3rd_person} the {reference}.",

    # Template 2: Uses present participle form
    "Go to the {location} and find the {target} {rel_present_participle} the {reference}, then {action} it.",

    # Template 3: Polite request style
    "Could you please {action} the {target} in the {location}? I'm looking for the one {rel_present_participle} the {reference}.",

    # Template 4: Task description style
    "In the {location}, there is a {target} that {rel_3rd_person} the {reference}. It needs to be {action}ed.",

    # Template 5: Two-step instruction style
    "First, head to the {location}. You should see a {target} {rel_present_participle} the {reference}. Please {action} it.",

    # Template 6: Location-first (does not directly describe target-reference relation)
    "In the {location}, near the {reference}, {action} the {target}.",

    # Template 7: Object existence confirmation style
    "There is a {target} in the {location}. It's the one that {rel_3rd_person} the {reference}. Please {action} it.",

    # Template 8: Conversational, asking for help
    "I need your help. In the {location}, can you find the {target} {rel_present_participle} the {reference} and {action} it?",

    # Template 9: Starting from finding the reference object
    "Find the {reference} in the {location}, then {action} the {target} that {rel_3rd_person} it.",

    # Template 10: Concise command style
    "Please go to the {location} and {action} the {target} you see {rel_present_participle} the {reference}."
]

smart_templates_v3 = [
    # Template 1: Uses smart verb phrase
    "{Action_capitalized} the {target} in the {location} that {rel_verb_phrase} the {reference}.",

    # Template 2: Uses smart participle phrase
    "Go to the {location} and find the {target} {rel_participle_phrase} the {reference}, then {action} it.",

    # Template 3: Polite request style
    "Could you please {action} the {target} in the {location}? I'm looking for the one {rel_participle_phrase} the {reference}.",

    # Template 4: Safe task description
    "In the {location}, there is a {target} that {rel_verb_phrase} the {reference}. The required task for it is to {action}.",

    # Template 5: Two-step instruction style
    "First, head to the {location}. You should see a {target} {rel_participle_phrase} the {reference}. Please {action} it.",

    # Template 6: Location-first (safe, no changes needed)
    "In the {location}, near the {reference}, {action} the {target}.",

    # Template 7: Object existence confirmation style
    "There is a {target} in the {location}. It's the one that {rel_verb_phrase} the {reference}. Please {action} it.",

    # Template 8: Conversational, asking for help
    "I need your help. In the {location}, can you find the {target} {rel_participle_phrase} the {reference} and {action} it?",

    # Template 9: Starting from finding the reference object
    "Find the {reference} in the {location}, then {action} the {target} that {rel_verb_phrase} it.",

    # Template 10: Concise command style
    "Please go to the {location} and {action} the {target} you see {rel_participle_phrase} the {reference}."
]

smart_templates_v4 = [
    # Uses the phrase with the full target name
    "{action_with_target_capitalized} in the {location} that {rel_verb_phrase} the {reference}.",

    # Uses the phrase with the pronoun "it"
    "Go to the {location} and find the {target} {rel_participle_phrase} the {reference}, then {action_with_pronoun}.",

    # A mix of both
    "Could you please {action_with_target} in the {location}? I'm looking for the one {rel_participle_phrase} the {reference}.",

    # Uses the pronoun "it"
    "In the {location}, there is a {target} that {rel_verb_phrase} the {reference}. Please {action_with_pronoun}.",

    # Uses the pronoun "it"
    "First, head to the {location}. You should see a {target} {rel_participle_phrase} the {reference}. Please {action_with_pronoun}.",

    # Uses the phrase with the full target name
    "In the {location}, near the {reference}, {action_with_target}.",

    # Uses the pronoun "it"
    "There is a {target} in the {location}. It's the one that {rel_verb_phrase} the {reference}. Please {action_with_pronoun}.",

    # Uses the pronoun "it"
    "I need your help. In the {location}, can you find the {target} {rel_participle_phrase} the {reference} and {action_with_pronoun}?",

    # Uses the phrase with the full target name
    "Find the {reference} in the {location}, then {action_with_target} that {rel_verb_phrase} it.",

    # Uses the pronoun "it"
    "Please go to the {location}, find the {target} you see {rel_participle_phrase} the {reference}, and {action_with_pronoun}."
]
