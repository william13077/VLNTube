action_categories_final = {
    "Cleaning & Tidying": [
        "arrange", "clean", "clean off", "clear", "clear off", "disinfect",
        "dust", "dust off", "fold", "make", "organize", "pair", "polish",
        "refold", "reline", "reorganize", "rinse", "sanitize", "scrub",
        "smooth", "stack", "straighten", "sweep", "tidy up", "vacuum",
        "wash", "wipe", "wipe down", 'mop', 'fluff'
    ],
    "Movement & Relocation": [
        "bring me", "carry", "enter", "fetch", "gather", "get", "lift",
        "lift up", "move", "pick up", "pull", "pull out", "push", "push in",
        "put away", "remove", "retrieve", "rotate", "spin", "stow",
        "store", "take", "take down", "approach"
    ],
    "State Change & Operation": [
        "activate", "adjust", "change", "close", "deactivate", "empty",
        "fill", "fix", "flush", "focus", "hang up", "light", "load",
        "lock", "open", "operate", "preheat", "recycle", "refill",
        "replace", "restart", "restock", "sharpen", "spread", "toggle",
        "turn off", "turn on", "unfold", "unload", "unplug"
    ],
    "Inspection & Examination": [
        "admire", "browse", "check", "display", "examine", "find",
        "inspect", "inventory", "locate", "read", "smell", "test"
    ],
    "General Interaction & Use": [
        "answer", "connect to", "decorate", "eat", "feed", "paint", "peel",
        "pet", "play", "play with", "prepare", "set", "sit at", "sit in",
        "sit on", "talk to", "touch", "trim", "use", "water", "wave at"
    ]
}

if __name__ == '__main__':
    from vistube.goal_gen.target_action import target_to_actions_final
    # Check if there are overlapping actions in different categories
    t_action = []
    for actions in target_to_actions_final.values():
        t_action += actions
    t_action = set(t_action)
    breakpoint()
    cat_actions = []
    for cat, actions in action_categories_final.items():
        cat_actions += actions
    cat_actions = set(cat_actions)

    print('Should be empty', t_action - cat_actions)  # actions in target_action but not in action_categories
    print('Could be redundant', cat_actions - t_action)  # actions in action_categories but not in target_action
