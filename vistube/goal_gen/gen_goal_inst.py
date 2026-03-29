import wordninja
import random
import spacy
import pyinflect # Step 1: The worker is hired.
from vistube.goal_gen.target_action import target_to_actions_final
from vistube.goal_gen.action_category import action_categories_final
from vistube.goal_gen.template import smart_templates, smart_templates_v3, smart_templates_v4

# Preposition
PREPOSITIONS = {'above', 'below', 'in', 'near', 'on', 'under', 'out of', 'next to', \
                'by', 'in front of', 'behind', 'to the right of', 'to the left of', 'with'}

# Load the NLP pipeline
nlp = spacy.load("en_core_web_sm")

def correct_description(description: str) -> str:
    """
    Automatically correct and format an English object description.

    Processing logic:
    1. Replace all underscores '_' with a single space ' '.
    2. Use the wordninja library to intelligently insert spaces between
       concatenated words (e.g. 'coffeemaker' -> 'coffee maker').

    Args:
        description: A possibly incorrectly formatted English description string.

    Returns:
        A cleaned and formatted string.
    """
    if not isinstance(description, str) or not description:
        return ""

    # Step 1: Replace underscores with spaces
    # 'remote_control' -> 'remote control'
    description = description.replace('_', ' ')
    description = description.replace('-', ' ')

    # Step 2: Use wordninja to split words. It handles already-spaced strings correctly.
    # 'washingmachine' -> ['washing', 'machine']
    # 'remote control' -> ['remote', 'control']
    split_words = wordninja.split(description)

    # Step 3: Rejoin the split words with spaces
    cleaned_description = ' '.join(split_words)

    return cleaned_description

# v2: Added manual fix dictionary for special cases
def correct_description_v2(description: str) -> str:
    """Automatically correct and format an English object description, handling known special cases."""
    manual_fixes = {
        'sofachair': 'sofa chair',
        'sofa_chair': 'sofa chair',
        'sofa': 'sofa',
        'chestofdrawers': 'chest of drawers',
        'tvstand': 'tv stand',
        'tv_stand': 'tv stand',
        'closestool': 'closestool',
        'cookware': 'cookware',
        'builtin_oven': 'built-in oven',
        'dish_washer': 'dishwasher',
        'spot_lig': 'spotlight',
        'spot_light': 'spotlight',
        'night_stand': 'nightstand',
        'nightstand': 'nightstand',
        'doorsil': 'doorsill',
        'doorsill': 'doorsill',
        # Add more words that wordninja handles incorrectly here
    }

    # Convert to lowercase and check against manual fixes
    cleaned_desc = description.lower()
    if cleaned_desc in manual_fixes:
        return manual_fixes[cleaned_desc]

    # If not a special case, use standard processing
    cleaned_desc = cleaned_desc.replace('_0000', '')
    cleaned_desc = cleaned_desc.replace('_000', '')
    cleaned_desc = cleaned_desc.replace('_00', '')
    cleaned_desc = cleaned_desc.replace('_0', '')
    temp_description = cleaned_desc.replace('_', ' ')
    split_words = wordninja.split(temp_description)
    return ' '.join(split_words)



# ==============================================================================
# Main generation functions
# ==============================================================================

def generate_instruction_smart(target: str, reference: str, relation: str, location: str) -> str:
    """
    Intelligently generate a random instruction based on input parameters,
    correctly handling verbs and prepositions.
    """
    cleaned_target = correct_description(target)
    action = random.choice(target_to_actions_final.get(cleaned_target, ['interact with']))

    # --- Core logic: process relation ---
    rel_3rd_person = relation
    rel_present_participle = relation

    if relation.lower() not in PREPOSITIONS:
        # If relation is a verb, conjugate it
        doc = nlp(relation)
        token = doc[0]
        # Get third-person singular form (VBZ), e.g., holds
        rel_3rd_person = token._.inflect('VBZ') or f"{relation}s" # fallback
        # Get present participle form (VBG), e.g., holding
        rel_present_participle = token._.inflect('VBG') or f"{relation}ing" # fallback

    # Randomly select a template
    template = random.choice(smart_templates)

    # Fill in the template
    instruction = template.format(
        Action_capitalized=action.capitalize(),
        action=action,
        target=cleaned_target,
        reference=reference,
        location=location,
        rel_3rd_person=rel_3rd_person,
        rel_present_participle=rel_present_participle
    ).replace("needs to be " + action + "ed", f"needs to be {action}ed (e.g., cleaned, moved)")

    return instruction

def is_preposition_spacy(word: str) -> bool:
    """
    Use spaCy's POS tagging to determine if a word or phrase is a preposition.
    """
    # Process the input word with the NLP model
    doc = nlp(word.lower())

    # Check if the first token's POS tag is 'ADP' (adposition/preposition)
    # This works for both single-word and phrasal prepositions (e.g. 'in front of')
    if doc[0].pos_ == 'ADP':
        return True

    return False

def generate_instruction_v3(target: str, reference: str, relation: str, location: str) -> str:
    """
    Generate instructions using smart grammar components to support complex
    and grammatically correct sentence structures.
    """
    cleaned_target = correct_description(target)
    action = random.choice(target_to_actions_final.get(cleaned_target, ['interact with']))

    # --- Core upgrade logic ---
    rel_verb_phrase = ""
    rel_participle_phrase = ""

    doc = nlp(relation.lower())
    pos_tag = doc[0].pos_

    if is_preposition_spacy(relation):
        # If it's a preposition (e.g., "next to")
        rel_verb_phrase = f"is {relation}"          # e.g., "is next to"
        rel_participle_phrase = relation            # e.g., "next to"
    else:
        # If it's a verb (e.g., "hold")
        doc = nlp(relation)
        token = doc[0]
        # Conjugate to third-person singular (VBZ)
        rel_verb_phrase = token._.inflect('VBZ') or f"{relation}s" # e.g., "holds"
        # Conjugate to present participle (VBG)
        rel_participle_phrase = token._.inflect('VBG') or f"{relation}ing" # e.g., "holding"

    if pos_tag == 'VERB':
        token = doc[0]
        # Conjugate to third-person singular (VBZ), e.g., "holds"
        rel_verb_phrase = token._.inflect('VBZ') or f"{relation}s"
        # Conjugate to present participle (VBG), e.g., "holding"
        rel_participle_phrase = token._.inflect('VBG') or f"{relation}ing"
    else:
        # Otherwise (for prepositions ADP, adverbs ADV, and other cases), treat as preposition
        # e.g., "is next to", "is behind"
        rel_verb_phrase = f"is {relation}"
        # e.g., "next to", "behind"
        rel_participle_phrase = relation

    # Randomly select a v3 template
    template = random.choice(smart_templates_v3)

    # Fill in the template
    instruction = template.format(
        Action_capitalized=action.capitalize(),
        action=action,
        target=cleaned_target,
        reference=reference,
        location=location,
        rel_verb_phrase=rel_verb_phrase,                 # Use new smart placeholders
        rel_participle_phrase=rel_participle_phrase    # Use new smart placeholders
    )

    return instruction

def generate_instruction_v5(target: str, reference: str, relation: str, location: str) -> str:
    """
    Generates an instruction with intelligent handling of phrasal verbs.
    """
    cleaned_target = correct_description_v2(target)
    actions_list = target_to_actions_final.get(cleaned_target) or ['interact with']
    action = random.choice(actions_list)

    # --- New Intelligent Logic ---
    action_parts = action.split()

    # Phrase with the full target name, e.g., "pick up the pillow"
    action_with_target = f"{action} the {cleaned_target}"

    # Phrase with the pronoun "it" correctly placed
    if action == 'bring me':
        action_with_pronoun = "bring it to me" # Special handling
    elif len(action_parts) == 2:
        # For phrasal verbs like "pick up" -> "pick it up"
        action_with_pronoun = f"{action_parts[0]} it {action_parts[1]}"
    else:
        # For simple verbs like "clean" -> "clean it"
        action_with_pronoun = f"{action} it"

    # (The logic for relation phrases from v4 remains the same)
    doc = nlp(relation.lower())
    pos_tag = doc[0].pos_

    if pos_tag == 'VERB':
        token = doc[0]
        rel_verb_phrase = token._.inflect('VBZ') or f"{relation}s"
        rel_participle_phrase = token._.inflect('VBG') or f"{relation}ing"
    else:
        rel_verb_phrase = f"is {relation}"
        rel_participle_phrase = relation

    # Choose a template from the new v4 list
    template = random.choice(smart_templates_v4)

    # Fill the template with the new placeholders
    instruction = template.format(
        action_with_target_capitalized=action_with_target.capitalize(),
        action_with_target=action_with_target,
        action_with_pronoun=action_with_pronoun,
        target=cleaned_target,
        reference=reference,
        location=location,
        rel_verb_phrase=rel_verb_phrase,
        rel_participle_phrase=rel_participle_phrase
    )

    return instruction

# v7: Added handling for target == 'person' to select the correct pronoun (it/them)
def generate_instruction_v7(target: str, reference: str, relation: str, location: str) -> str:
    """
    Generate instructions with intelligent handling of phrasal verbs
    and special target pronouns (e.g. person -> them).
    """
    cleaned_target = correct_description_v2(target) # Use v2 cleaning function
    actions_list = target_to_actions_final.get(cleaned_target) or ['interact with']
    action = random.choice(actions_list)

    # --- Core upgrade logic ---
    # Determine the correct pronoun based on the target
    pronoun = "it"
    if cleaned_target == "person":
        pronoun = "them"  # Use them as gender-neutral singular pronoun

    action_parts = action.split()
    action_with_target = f"{action} the {cleaned_target}"

    action_with_pronoun = ""
    if action == 'bring me':
        action_with_pronoun = f"bring {pronoun} to me" # Use pronoun variable
    elif len(action_parts) == 2 and action_parts[0] == 'sit':
        # NEW: Special case for inseparable "sit" verbs
        action_with_pronoun = f"{action} {pronoun}" # e.g., "sit on it"
    elif len(action_parts) == 2:
        action_with_pronoun = f"{action_parts[0]} {pronoun} {action_parts[1]}" # Use pronoun variable
    else:
        action_with_pronoun = f"{action} {pronoun}" # Use pronoun variable

    # (Relation phrase logic remains the same)
    doc = nlp(relation.lower())
    pos_tag = doc[0].pos_
    if pos_tag == 'VERB':
        token = doc[0]
        rel_verb_phrase = token._.inflect('VBZ') or f"{relation}s"
        rel_participle_phrase = token._.inflect('VBG') or f"{relation}ing"
    else:
        rel_verb_phrase = f"is {relation}"
        rel_participle_phrase = relation

    template = random.choice(smart_templates_v4)

    instruction = template.format(
        action_with_target_capitalized=action_with_target.capitalize(),
        action_with_target=action_with_target,
        action_with_pronoun=action_with_pronoun,
        target=cleaned_target,
        reference=reference,
        location=location,
        rel_verb_phrase=rel_verb_phrase,
        rel_participle_phrase=rel_participle_phrase
    )
    return instruction

# The final, complete set based on your full list
inseparable_phrasal_verbs = {
    'talk to',
    'wave at',
    'sit at',
    'sit in',
    'sit on',
    'connect to',
    'play with'
}

# --- Final version V8 ---

def generate_instruction_v8(target: str, reference: str, relation: str, location: str) -> str:
    """
    Generate instructions using an "inseparable phrasal verb list"
    to perfectly handle all types of phrasal verbs.
    """
    cleaned_target = correct_description_v2(target)
    actions_list = target_to_actions_final.get(cleaned_target) or ['interact with']
    action = random.choice(actions_list)

    pronoun = "it"
    if cleaned_target == "person":
        pronoun = "them"

    action_parts = action.split()
    action_with_target = f"{action} the {cleaned_target}"

    action_with_pronoun = ""
    if action == 'bring me':
        # Special case 1: "bring me"
        action_with_pronoun = f"bring {pronoun} to me"
    elif action in inseparable_phrasal_verbs:
        # Special case 2: All inseparable verbs (talk to, sit on, etc.)
        action_with_pronoun = f"{action} {pronoun}" # e.g., "talk to them", "sit on it"
    elif len(action_parts) == 2:
        # Standard case: Separable verbs (pick up, turn on, etc.)
        action_with_pronoun = f"{action_parts[0]} {pronoun} {action_parts[1]}" # e.g., "pick it up"
    else:
        # Standard case: Single verbs
        action_with_pronoun = f"{action} {pronoun}"

    # (Relation phrase logic remains the same)
    doc = nlp(relation.lower())
    pos_tag = doc[0].pos_
    if pos_tag == 'VERB':
        token = doc[0]
        rel_verb_phrase = token._.inflect('VBZ') or f"{relation}s"
        rel_participle_phrase = token._.inflect('VBG') or f"{relation}ing"
    else:
        rel_verb_phrase = f"is {relation}"
        rel_participle_phrase = relation

    instructions = []
    for template in smart_templates_v4:
        instruction = template.format(
            action_with_target_capitalized=action_with_target.capitalize(),
            action_with_target=action_with_target,
            action_with_pronoun=action_with_pronoun,
            target=cleaned_target,
            reference=reference,
            location=location,
            rel_verb_phrase=rel_verb_phrase,
            rel_participle_phrase=rel_participle_phrase
        )
        instructions.append(instruction)
    return instructions
