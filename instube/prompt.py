# NOTE: Second-person imperative instead of video caption style.
VLN_PROMPT_VIDEO = """You are an expert navigation assistant. Your goal is to watch a first-person video of a path and generate a clear, step-by-step instruction for a person or a robot to follow that exact path.

The instruction MUST be a direct command written in the second-person imperative voice (e.g., "Walk forward," "Turn left"). You are telling an agent what to do.

Your instruction must include these four elements:
1.  **Landmarks:** Refer to areas, rooms, or furniture the agent moves through (e.g., "enter the living room", "move towards the bathroom", "pass by a couch").
2.  **Spatial Information:** Identify the direction of landmarks relative to the agent (e.g., "with the table on your left", "the chairs are on the right-hand side").
3.  **Actions:** Detail the agent's movements using command verbs (e.g., "turn left at the hallway", "walk straight past the sofa", "go through the doorway").
4.  **End Point:** Clearly describe the final stopping position (e.g., "Stop near the sink in front of you", "Stay in front of the TV", "Stop to the left of the bed").

**EXAMPLE OF A PERFECT INSTRUCTION:**
"Walk past a dining table on the left and a living room on the right, then turn right into a hallway. Proceed straight down the hallway, and then turn left to enter a bathroom, stopping in front of the window."

**IMPORTANT RULES:**
-   **DO NOT** use third-person descriptive words like "Walks," "Moves," "Enters," or "Proceeds."
-   **DO NOT** narrate what is happening like a video caption. Give direct commands.
-   Avoid using "Move backward."

Now, generate the navigation instruction for the following video."""



#=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*
# NOTE: Image sequence
VLN_PROMPT_IMAGE_SEQUENCE = """You are an expert navigation assistant. Your goal is to analyze a **sequence of images**, which represents a continuous first-person path, and generate a clear, step-by-step instruction for a person or a robot to follow that exact path. Your highest priority is the accuracy of the actions.

The images are provided in chronological order and should be treated as frames from a video.

The instruction MUST be a direct command written in the second-person imperative voice (e.g., "Walk forward," "Turn left"). You are telling an agent what to do.

Your instruction must include these four elements:
1.  **Landmarks:** Refer to areas, rooms, or furniture the agent moves through (e.g., "enter the living room", "move towards the bathroom", "pass by a couch").
2.  **Spatial Information:** Identify the direction of landmarks relative to the agent (e.g., "with the table on your left", "the chairs are on the right-hand side").
3.  **Actions:** Detail the agent's movements using command verbs (e.g., "turn left at the hallway", "walk straight past the sofa", "go through the doorway").
4.  **End Point:** Clearly describe the final stopping position (e.g., "Stop near the sink in front of you", "Stay in front of the TV", "Stop to the left of the bed").

**EXAMPLE OF A PERFECT INSTRUCTION:**
"Walk past a dining table on the left and a living room on the right, then turn right into a hallway. Proceed straight down the hallway, and then turn left to enter a bathroom, stopping in front of the window."

**IMPORTANT RULES:**
-   **DO NOT** use third-person descriptive words like "Walks," "Moves," "Enters," or "Proceeds."
-   **DO NOT** narrate what is happening like a video caption. Give direct commands.
-   Avoid using "Move backward."

Now, generate the navigation instruction for the following image sequence."""

# NOTE Describe the goal
CAPTION_GENERATION_PROMPT = """
Analyze the provided image, presumably from room [{room_id}]. Focus specifically on the [{target_object_name}].
Describe its visual attributes (e.g., color, material if obvious) and its current state (e.g., open/closed, full/empty, on/off if applicable and visible).
Also, describe its spatial relationship to the [{reference_object_name}] if visible and relevant. Mention its relationship to any other significant nearby objects as well.
If the target object or reference object isn't clearly visible, the relationship is ambiguous, or the image itself is problematic (e.g., black, blurry), state that clearly (e.g., 'Target object [target_object_name] not clearly visible.' or 'Image is unclear.').
Respond ONLY with the concise description or the unclear status. Be factual.
"""

# NOTE Fuse text instructions and image caption to rewrite instructions
REWRITE_PROMPT_FUSION = """
You are an expert in natural language generation, specializing in data augmentation for goal-oriented navigation instructions by fusing text and image-derived information.

Your task is to analyze 10 rigid text instructions AND an image caption describing the scene. The 10 text instructions share the same semantic goal. The caption provides a targeted description of the goal object, its state/attributes, and its spatial relationships based on an image.

**Core Task:** Generate three new, high-quality, and distinct instructions (Formal, Natural, Casual) in English based primarily on the TEXT instructions' goal (action, target object, reference object).

**Fusion Rule for Spatial Relationship & Details:**
- Compare the spatial relationship described in the IMAGE CAPTION (specifically between the target and reference objects) with the one consistently described in the TEXT instructions.
- **IF** the caption is informative (not 'unclear', 'not visible', etc.) AND describes a relationship between the correct objects AND this relationship seems physically plausible AND it differs from the text relationship (especially if the text relationship seems implausible): **PRIORITIZE the spatial relationship from the IMAGE CAPTION.** Incorporate relevant attributes/state details from the caption (e.g., color, open/closed) naturally into the generated instructions.
- **ELSE** (caption is uninformative, describes wrong objects, relationship is unclear, or matches the text): **Use the spatial relationship from the TEXT instructions.** You may still incorporate object attributes/state details from the caption if available and relevant.

**Output Styles:**
1.  **Formal:** Precise, objective, complete sentences. Use details sparingly.
2.  **Natural:** Clear, polite, conversational. Use relevant details naturally.
3.  **Casual:** Very informal, colloquial, uses fragments. Use key details concisely.

You MUST return ONLY a valid JSON object (and nothing else) with the following structure:
{{
  "formal": "The formal instruction you generated.",
  "natural": "The natural instruction you generated.",
  "casual": "The casual instruction you generated."
}}

**[Text Instructions]**
{instructions_text}
**[/Text Instructions]**

**[Image Caption (Targeted Description)]**
{image_caption}
**[/Image Caption]**
"""



