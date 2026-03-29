import json
import random
import numpy as np
import os
import natsort
from vlntube.goal_gen.gen_goal_inst import correct_description_v2, correct_description
from vlntube.goal_gen.target_action import target_to_actions_final
from vlntube.tube_utils import extract_object_type_outer


def find_unique_objects(scene_data: dict, n: int) -> list:
    """
    Analyzes a scene graph to find N unique object relationships.

    Uniqueness is determined in two ways:
    1.  Relational Uniqueness: The target is the only object with a specific
        relationship to the source (e.g., the only thing "under" the couch).
    2.  Semantic Uniqueness: The target is the only object of its CATEGORY
        with a specific relationship (e.g., the only "couch" on the blanket).

    Args:
        scene_data: A dictionary representing the scene graph, where keys are
                    object instance_ids.
        n: The maximum number of unique relationships to find.

    Returns:
        A list of strings, with each string describing a found unique
        relationship.
    """
    found_uniques = []
    # Get a list of all object IDs and shuffle them randomly
    all_object_ids = list(scene_data.keys())
    random.shuffle(all_object_ids)

    # Iterate through the shuffled objects to find N unique relationships
    for source_id in all_object_ids:
        if len(found_uniques) >= n:
            break  # Stop if we have found enough unique objects

        source_object = scene_data[source_id]
        source_category = source_object.get("category", "unknown")
        nearby_objects = source_object.get("nearby_objects", {})

        # --- Check for both types of uniqueness ---

        # 1. Relational uniqueness check
        # Count relationships by type (e.g., {'on': 5, 'under': 1, 'near': 8})
        relation_counts = {}
        for details in nearby_objects.values():
            relation = details[0]
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        # Find relationships that only appear once
        for target_id, details in nearby_objects.items():
            relation = details[0]
            if relation_counts[relation] == 1:
                target_category = scene_data.get(target_id, {}).get("category", "unknown")
                description = f"'{target_category}' is the only object '{relation}' '{source_category}' (relationally unique)"
                # To avoid duplicates, check if a similar relationship isn't already found
                if description not in found_uniques:
                    found_uniques.append(description)
                    if len(found_uniques) >= n:
                        return found_uniques

        # 2. Semantic uniqueness check
        # Group objects by relationship AND category
        # e.g., {('on', 'couch'): 1, ('on', 'curtain'): 3}
        semantic_counts = {}
        for target_id, details in nearby_objects.items():
            relation = details[0]
            # Look up the category of the target object
            target_category = scene_data.get(target_id, {}).get("category", "unknown")
            key = (relation, target_category)
            semantic_counts[key] = semantic_counts.get(key, 0) + 1

        # Find semantic groups that only have one member
        for (relation, target_category), count in semantic_counts.items():
            if count == 1:
                description = f"'{target_category}' is the only object '{relation}' '{source_category}' (semantically unique)"
                if description not in found_uniques:
                     found_uniques.append(description)
                     if len(found_uniques) >= n:
                        return found_uniques


    return found_uniques


def find_unique_objects_with_ids(
    scene_data: dict,
    n: int,
    source_blacklist: list = None,
    target_blacklist: list = None
) -> list:
    """
    Analyzes a scene graph to find N unique object relationships, using separate
    blacklists for source and target objects.

    Args:
        scene_data: A dictionary representing the scene graph.
        n: The maximum number of unique relationships to find.
        source_blacklist: A list of strings. If a string appears in a starting
                          object's ID, that object is ignored.
        target_blacklist: A list of strings. If a string appears in a target
                          object's ID, that relationship is ignored.

    Returns:
        A list of strings describing the found unique relationships.
    """
    if source_blacklist is None:
        source_blacklist = []
    if target_blacklist is None:
        target_blacklist = []

    found_uniques = []

    # Filter the list of all possible SOURCE objects
    all_object_ids = list(scene_data.keys())
    if source_blacklist:
        all_object_ids = [
            obj_id for obj_id in all_object_ids
            if not any(keyword in obj_id for keyword in source_blacklist)
            # if any(keyword in obj_id for keyword in source_blacklist)
        ]

    random.shuffle(all_object_ids)

    # Iterate through the filtered source objects
    for source_id in all_object_ids:
        if len(found_uniques) >= n:
            break

        source_object = scene_data.get(source_id)
        if not source_object:
            continue

        nearby_objects = source_object.get("nearby_objects", {})

        # Filter the nearby (TARGET) objects
        if target_blacklist:
            nearby_objects = {
                target_id: details for target_id, details in nearby_objects.items()
                if not any(keyword in target_id for keyword in target_blacklist)
                # if any(keyword in target_id for keyword in target_blacklist)
            }

        if not nearby_objects:
            continue

        # --- Uniqueness analysis (operates on doubly filtered data) ---

        # 1. Relational uniqueness
        relation_counts = {}
        for details in nearby_objects.values():
            relation = details[0]
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        for target_id, details in nearby_objects.items():
            relation = details[0]
            if relation_counts.get(relation) == 1:
                description = f"'{target_id}' is the only object '{relation}' '{source_id}' (relationally unique)"
                if description not in found_uniques:
                    found_uniques.append(description)
                    if len(found_uniques) >= n: return found_uniques

        # 2. Semantic uniqueness
        semantic_counts = {}
        for target_id, details in nearby_objects.items():
            relation = details[0]
            target_object = scene_data.get(target_id)
            if not target_object: continue
            target_category = target_object.get("category", "unknown")
            key = (relation, target_category)
            semantic_counts[key] = semantic_counts.get(key, 0) + 1

        for target_id, details in nearby_objects.items():
            relation = details[0]
            target_object = scene_data.get(target_id)
            if not target_object: continue
            target_category = target_object.get("category", "unknown")
            key = (relation, target_category)
            if semantic_counts.get(key) == 1:
                description = f"'{target_id}' is the only object '{relation}' '{source_id}' (semantically unique)"
                if description not in found_uniques:
                    found_uniques.append(description)
                    if len(found_uniques) >= n: return found_uniques

    return found_uniques

def _is_relationship_unique(source_id: str, target_id: str, scene_data: dict) -> tuple:
    """
    A helper function to check for one-way uniqueness (relational and semantic).

    Args:
        source_id: The starting object's ID.
        target_id: The target object's ID.
        scene_data: The full scene graph.

    Returns:
        A tuple (is_unique, uniqueness_type, relationship_name).
        Returns (False, None, None) if not unique.
    """
    source_object = scene_data.get(source_id)
    if not source_object:
        return False, None, None

    nearby_objects = source_object.get("nearby_objects", {})
    target_details = nearby_objects.get(target_id)
    if not target_details:
        return False, None, None # No direct relationship found

    # This is the relationship from the source's perspective (e.g., target is "on" source)
    relation = target_details[0]

    # 1. Relational Check: Is this the ONLY object with this relationship?
    relation_count = sum(1 for r in nearby_objects.values() if r[0] == relation)
    if relation_count == 1:
        return True, "relationally unique", relation

    # 2. Semantic Check: Is this the ONLY object of its CATEGORY with this relationship?
    target_object = scene_data.get(target_id)
    if not target_object:
        return False, None, None
    # target_category = target_object.get("category")
    target_category = target_object.get("category").split('_')[0] # Use only the main category part

    semantic_count = 0
    for obj_id, details in nearby_objects.items():
        if details[0] == relation:
            obj_data = scene_data.get(obj_id)
            # if obj_data and obj_data.get("category")== target_category:
            if obj_data and obj_data.get("category").split('_')[0] == target_category:
                semantic_count += 1

    if semantic_count == 1:
        return True, "semantically unique", relation

    return False, None, None

def find_bidirectionally_unique_objects(
    scene_data: dict,
    n: int,
    source_blacklist: list = None,
    target_blacklist: list = None
) -> list:
    """
    Finds N bidirectionally unique object relationships in a scene graph.

    A relationship between Object A and Object B is unique only if:
    1. The relationship from A to B is unique (relationally or semantically).
    2. The reverse relationship from B to A is also unique.

    Args:
        scene_data: The dictionary representing the scene graph.
        n: The maximum number of unique relationships to find.
        source_blacklist: Blacklist for starting objects.
        target_blacklist: Blacklist for target objects.

    Returns:
        A list of strings describing the found bidirectional relationships.
    """
    if source_blacklist is None: source_blacklist = []
    if target_blacklist is None: target_blacklist = []

    found_uniques = []

    all_object_ids = list(scene_data.keys())
    if source_blacklist:
        all_object_ids = [
            obj_id for obj_id in all_object_ids
            if not any(keyword in obj_id for keyword in source_blacklist)
        ]

    random.shuffle(all_object_ids)

    for source_id in all_object_ids:
        if len(found_uniques) >= n:
            break

        source_object = scene_data.get(source_id)
        if not source_object: continue

        nearby_objects = source_object.get("nearby_objects", {})

        # Iterate through potential targets
        for target_id in nearby_objects.keys():
            # Apply target blacklist
            if any(keyword in target_id for keyword in target_blacklist):
                continue

            # Step 1: Forward Check (Source -> Target)
            is_forward_unique, f_type, f_relation = _is_relationship_unique(source_id, target_id, scene_data)

            if is_forward_unique:
                # Step 2: Reverse Check (Target -> Source)
                is_reverse_unique, r_type, r_relation = _is_relationship_unique(target_id, source_id, scene_data)

                if is_reverse_unique:
                    # Step 3: Bidirectional uniqueness confirmed!
                    desc = (
                        f"bidirectionally unique: ('{target_id}' [{f_relation}, {f_type}]) <=> "
                        f"('{source_id}' [{r_relation}, {r_type}])"
                    )

                    # Add checks to prevent adding inverse duplicates
                    reverse_desc = (
                        f"bidirectionally unique: ('{source_id}' [{r_relation}, {r_type}]) <=> "
                        f"('{target_id}' [{f_relation}, {f_type}])"
                    )

                    if desc not in found_uniques and reverse_desc not in found_uniques:
                        found_uniques.append(desc)
                        if len(found_uniques) >= n:
                            return found_uniques
    return found_uniques

# def find_bidirectionally_unique_objects_debug(
#     scene_data: dict,
#     n: int,
#     source_blacklist: list = None,
#     target_blacklist: list = None,
#     debug: bool = False
# ) -> list:
#     """
#     Finds N bidirectionally unique relationships or, if debug is True, finds
#     relationships that are unique in only one direction.
#
#     Args:
#         scene_data: The dictionary representing the scene graph.
#         n: The maximum number of unique relationships to find (in normal mode).
#         source_blacklist: Blacklist for starting objects.
#         target_blacklist: Blacklist for target objects.
#         debug: If True, returns a list of one-way unique but not
#                bidirectionally unique relationships.
#
#     Returns:
#         A list of strings describing the found relationships.
#     """
#     if source_blacklist is None: source_blacklist = []
#     if target_blacklist is None: target_blacklist = []
#
#     found_uniques = []
#     debug_findings = [] # New list for debug mode results
#
#     all_object_ids = list(scene_data.keys())
#     # The blacklist for source objects is applied here
#     if source_blacklist:
#         all_object_ids = [
#             obj_id for obj_id in all_object_ids
#             if not any(keyword in obj_id for keyword in source_blacklist)
#         ]
#
#     random.shuffle(all_object_ids)
#
#     for source_id in all_object_ids:
#         # Stop condition for normal mode
#         if not debug and len(found_uniques) >= n:
#             break
#
#         source_object = scene_data.get(source_id)
#         if not source_object: continue
#
#         nearby_objects = source_object.get("nearby_objects", {})
#
#         # Iterate through potential targets
#         for target_id in nearby_objects.keys():
#             # Apply target blacklist
#             if any(keyword in target_id for keyword in target_blacklist):
#                 continue
#
#             # Step 1: Forward Check (Source -> Target)
#             is_forward_unique, f_type, f_relation = _is_relationship_unique(source_id, target_id, scene_data)
#
#             if is_forward_unique:
#                 # Step 2: Reverse Check (Target -> Source)
#                 is_reverse_unique, _, _ = _is_relationship_unique(target_id, source_id, scene_data)
#
#                 # --- LOGIC BRANCHING BASED ON DEBUG FLAG ---
#
#                 if debug:
#                     # In debug mode, we are looking for relationships that FAIL the reverse check.
#                     if not is_reverse_unique:
#                         desc = (
#                             f"one-way unique (reverse check failed): '{target_id}' and '{source_id}' "
#                             f"have relationship '{f_relation}' ({f_type}), but the reverse check is not unique."
#                         )
#                         if desc not in debug_findings:
#                             debug_findings.append(desc)
#                 else:
#                     # In normal mode, we are looking for relationships that PASS the reverse check.
#                     if is_reverse_unique:
#                         r_type, r_relation = _is_relationship_unique(target_id, source_id, scene_data)[1:]
#                         desc = (
#                             f"bidirectionally unique: ('{target_id}' [{f_relation}, {f_type}]) <=> "
#                             f"('{source_id}' [{r_relation}, {r_type}])"
#                         )
#                         reverse_desc = (
#                             f"bidirectionally unique: ('{source_id}' [{r_relation}, {r_type}]) <=> "
#                             f"('{target_id}' [{f_relation}, {f_type}])"
#                         )
#
#                         if desc not in found_uniques and reverse_desc not in found_uniques:
#                             found_uniques.append(desc)
#                             if len(found_uniques) >= n:
#                                 # Break outer loop if N is reached
#                                 break
#         if not debug and len(found_uniques) >= n:
#             break
#
#     # Return the appropriate list based on the mode
#     return debug_findings if debug else found_uniques

def find_bidirectionally_unique_objects_debug(
    scene_data: dict,
    n: int,
    source_blacklist: list = None,
    target_blacklist: list = None,
    debug: bool = False
) -> tuple | list:
    """
    Finds bidirectionally unique relationships and returns them in two formats.

    In normal mode, returns a tuple: (list_of_strings, structured_dictionary).
    In debug mode, returns a single list of debug strings.

    Args:
        scene_data, n, source_blacklist, target_blacklist, debug...

    Returns:
        A tuple containing a list of descriptive strings and a structured
        dictionary, or a single list if in debug mode.
    """
    if source_blacklist is None: source_blacklist = []
    if target_blacklist is None: target_blacklist = []

    # --- INITIALIZE RETURN STRUCTURES ---
    descriptive_list = []
    structured_dict = {
        "sem": [],
        "rel": []
    }
    rel_mapping = {"semantically unique":"sem","relationally unique":"rel"}
    debug_findings = []

    all_object_ids = list(scene_data.keys())
    if source_blacklist:
        all_object_ids = [
            obj_id for obj_id in all_object_ids
            if not any(keyword in obj_id for keyword in source_blacklist)
        ]

    random.shuffle(all_object_ids)

    for source_id in all_object_ids:
        if not debug and len(descriptive_list) >= n: break

        source_object = scene_data.get(source_id)
        if not source_object: continue

        nearby_objects = source_object.get("nearby_objects", {})

        for target_id in nearby_objects.keys():
            if any(keyword in target_id for keyword in target_blacklist): continue

            # Forward Check (Source -> Target)
            is_forward_unique, f_type, f_relation = _is_relationship_unique(source_id, target_id, scene_data)

            if is_forward_unique:
                # Reverse Check (Target -> Source)
                is_reverse_unique, r_type, r_relation = _is_relationship_unique(target_id, source_id, scene_data)

                if debug:
                    if not is_reverse_unique:
                        desc = f"one-way unique (reverse check failed): '{target_id}' and '{source_id}' have relationship '{f_relation}' ({f_type.split()[0]} unique), but the reverse check is not unique."
                        if desc not in debug_findings: debug_findings.append(desc)
                elif is_reverse_unique:
                    # --- BUILD DESCRIPTIVE STRING ---
                    desc = f"bidirectionally unique: ('{target_id}' [{f_relation}, {f_type.split()[0]}]) <=> ('{source_id}' [{r_relation}, {r_type.split()[0]}])"
                    reverse_desc = f"bidirectionally unique: ('{source_id}' [{r_relation}, {r_type.split()[0]}]) <=> ('{target_id}' [{f_relation}, {f_type.split()[0]}])"

                    if desc not in descriptive_list and reverse_desc not in descriptive_list:
                        descriptive_list.append(desc)

                        # --- BUILD STRUCTURED DICTIONARY ITEM ---
                        structured_item = {
                            "object_1_id": target_id,
                            "object_1_relation_to_2": f_relation,
                            "object_2_id": source_id,
                            "object_2_relation_to_1": r_relation
                        }
                        # Add to the correct list based on the forward check type
                        structured_dict[ rel_mapping[f_type]].append(structured_item)

                        if len(descriptive_list) >= n:
                            return descriptive_list, structured_dict

    if debug:
        return debug_findings
    else:
        return descriptive_list, structured_dict

def find_bidirectionally_unique_objects_exact(
    scene_data: dict,
    n: int,
    source_blacklist: list = None,
    target_blacklist: list = None,
    debug: bool = False
) -> tuple | list:
    """
    Improved bidirectional unique object relationship finder using exact matching to avoid false positives.

    Main improvements:
    1. Uses exact object type matching instead of substring matching
    2. Correctly extracts object type (strips numeric suffix _0001 etc.)
    3. Avoids false positives with similar names (e.g., table_la vs table_lamp)

    Args:
        scene_data: Scene graph data dictionary
        n: Maximum number of results to return
        source_blacklist: Source object blacklist (exact object type matching)
        target_blacklist: Target object blacklist (exact object type matching)
        debug: Whether to use debug mode

    Returns:
        Normal mode: (description list, structured dictionary)
        Debug mode: debug information list
    """
    if source_blacklist is None:
        source_blacklist = []
    if target_blacklist is None:
        target_blacklist = []

    # Helper function: extract object type
    def get_object_type(obj_id: str) -> str:
        """Extract object type from object ID (strip numeric suffix)"""
        if '/' in obj_id:
            # Format: table_lamp_0001/xxx -> table_lamp
            base_name = obj_id.split('/')[0]
        else:
            base_name = obj_id

        # Strip the trailing _XXXX numeric suffix
        if base_name and len(base_name) > 5 and base_name[-5] == '_':
            return base_name[:-5]
        return base_name

    # Helper function: check if object is in blacklist (exact matching)
    def is_in_blacklist(obj_id: str, blacklist: list) -> bool:
        """Check if object is in blacklist using exact matching"""
        obj_type = get_object_type(obj_id)
        return obj_type in blacklist

    # Initialize return structures
    descriptive_list = []
    structured_dict = {
        "sem": [],
        "rel": []
    }
    rel_mapping = {"semantically unique": "sem", "relationally unique": "rel"}
    debug_findings = []

    # Get all object IDs and apply source blacklist filter
    all_object_ids = list(scene_data.keys())

    # Filter source objects using exact matching
    if source_blacklist:
        filtered_ids = []
        for obj_id in all_object_ids:
            if not is_in_blacklist(obj_id, source_blacklist):
                filtered_ids.append(obj_id)
        all_object_ids = filtered_ids

    random.shuffle(all_object_ids)

    # Main loop: iterate through all source objects
    for source_id in all_object_ids:
        if not debug and len(descriptive_list) >= n:
            break

        source_object = scene_data.get(source_id)
        if not source_object:
            continue

        nearby_objects = source_object.get("nearby_objects", {})

        # Iterate through all target objects
        for target_id in nearby_objects.keys():
            # Check target blacklist using exact matching
            if target_blacklist and is_in_blacklist(target_id, target_blacklist):
                continue

            # Forward check (Source -> Target)
            is_forward_unique, f_type, f_relation = _is_relationship_unique(source_id, target_id, scene_data)

            if is_forward_unique:
                # Reverse check (Target -> Source)
                is_reverse_unique, r_type, r_relation = _is_relationship_unique(target_id, source_id, scene_data)

                if debug:
                    # Debug mode: record one-way unique relationships
                    if not is_reverse_unique:
                        source_type = get_object_type(source_id)
                        target_type = get_object_type(target_id)
                        desc = (
                            f"one-way unique (reverse check failed): '{target_type}' and '{source_type}' "
                            f"have relationship '{f_relation}' ({f_type}), but the reverse check is not unique."
                        )
                        if desc not in debug_findings:
                            debug_findings.append(desc)

                elif is_reverse_unique:
                    # Normal mode: record bidirectionally unique relationships
                    # Build description string (using object type instead of full ID)
                    source_type = get_object_type(source_id)
                    target_type = get_object_type(target_id)

                    desc = (
                        f"bidirectionally unique: ('{target_type}' [{f_relation}, {f_type}]) <=> "
                        f"('{source_type}' [{r_relation}, {r_type}])"
                    )
                    reverse_desc = (
                        f"bidirectionally unique: ('{source_type}' [{r_relation}, {r_type}]) <=> "
                        f"('{target_type}' [{f_relation}, {f_type}])"
                    )

                    # Avoid adding duplicates
                    if desc not in descriptive_list and reverse_desc not in descriptive_list:
                        descriptive_list.append(desc)

                        # Build structured dictionary item (using full ID)
                        structured_item = {
                            "object_1_id": target_id,
                            "object_1_relation_to_2": f_relation,
                            "object_2_id": source_id,
                            "object_2_relation_to_1": r_relation,
                            # Additionally include object type info for debugging
                            "object_1_type": target_type,
                            "object_2_type": source_type
                        }

                        # Add to the correct list based on the forward check type
                        structured_dict[rel_mapping[f_type]].append(structured_item)

                        if len(descriptive_list) >= n:
                            return descriptive_list, structured_dict

    # Return results
    if debug:
        return debug_findings
    else:
        # Print statistics
        # total_found = len(structured_dict['sem']) + len(structured_dict['rel'])
        # print(f"Found {total_found} bidirectionally unique relationships")
        # print(f"  - Semantically unique: {len(structured_dict['sem'])}")
        # print(f"  - Relationally unique: {len(structured_dict['rel'])}")

        return descriptive_list, structured_dict

if __name__ == '__main__':
    # path = '/home/sihao/lsh/dataset/assets/benchmark/meta/MV7J6NIKTKJZ2AABAAAAADA8_usd/object_dict.json'
    path = '/home/sihao/lsh/dataset/meta/MV7J6NIKTKJZ2AABAAAAADA8_usd/object_dict.json'
    with open(path,'r') as f: obj_dict = json.load(f)
    # obj_list = find_unique_objects_with_ids(obj_dict,
    #                                         n=100000000,
    #                                         # source_blacklist=['bowl','decoration',],
    #                                         source_blacklist=[],
    #                                         # target_blacklist=['decoration']
    #                                         # target_blacklist=['plate','bowl','cup']
    #                                         target_blacklist=[]
    #                                         )
    # for p in obj_list:
    #     if 'oven' in p.split(' ')[3]:
    #         print(p)

    # gruroot = '/data/lsh/dataset/meta/' #gru
    gruroot = '/data/lsh/scene_summary/scene_summary' #kujiale
    temp_dir = os.listdir(gruroot)
    dir = natsort.natsorted([i for i in temp_dir if os.path.isdir(os.path.join(gruroot,i))])
    object_1 =[]
    object_2 =[]
    rel = []
    target = []
    debug = []
    for scene_id in dir:
    # for scene_id in ['kujiale_0003']:
        # if scene_id in ['demo_0000']: #gru
        if scene_id in ['kujiale', 'demo_0000', 'hello_world','003','test003']: #kujiale
            continue
        # if scene_id not in ['kujiale_0121']: #kujiale
        #     continue
        path = os.path.join(gruroot,scene_id,'object_dict.json')
        with open(path,'r') as f: obj_data = json.load(f)
        source_blacklist=['ceiling','doorsil','ornament','daily_equipment','tooling','unknown','menorah','celling']
        obj_list,obj_dict = find_bidirectionally_unique_objects_debug(obj_data,
                                                n=100000000,
                                                debug = False,
                                                # source_blacklist=['decoration','plate','bowl'],
                                                # source_blacklist=['plate','bowl'],
                                                # source_blacklist=[],
                                                # target_blacklist=['decoration']
                                                # target_blacklist=['plate','bowl','cup']
                                                # target_blacklist=[]
                                                source_blacklist=source_blacklist,
                                                target_blacklist=source_blacklist,
                                                )
        # s_blacklist = ['ceiling', 'doorsil', 'ornament', 'daily_equipment', 'tooling',
        #                 'unknown', 'decorat','decoration', 'menorah', 'stora', 'table_la',
        #                 'cabin', 'chandelier', 'floor','celling']
        # t_blacklist = s_blacklist

        # # Use the new exact matching function to avoid false positives with similar names like table_lamp
        # obj_list, obj_dict = find_bidirectionally_unique_objects_exact(
        #     obj_data, n=100000, debug=False,
        #     source_blacklist=s_blacklist, target_blacklist=t_blacklist
        # )

        print(f'Processing scene {scene_id}: {len(obj_list)}\n')

        for p in obj_dict['sem']+obj_dict['rel']:
            # Debug check with proper extraction
            o1_full = p['object_1_id'].split('/')[0]
            o1 = extract_object_type_outer(o1_full)
            if 'unknown' in o1:
                debug.append(p['object_1_id'])
                print('ERROR: bedding_0003 extracted, should be just bedding!')
                print(f'  Full ID: {p["object_1_id"]}')
                print(f'  Extracted: {o1}')

        for p in obj_dict['sem']+obj_dict['rel']:
            # o1 = p['object_1_id'].split('/')[0].split('_')[0]
            # o2 = p['object_2_id'].split('/')[0].split('_')[0]
            # Smart extraction: keep only non-numeric parts
            o1_full = p['object_1_id'].split('/')[0]
            o2_full = p['object_2_id'].split('/')[0]

            # Extract object type by keeping parts until first numeric part
            def extract_object_type(full_id):
                parts = full_id.split('_')
                non_numeric_parts = []
                for part in parts:
                    if part.isdigit():
                        break  # Stop at first numeric part
                    non_numeric_parts.append(part)
                return '_'.join(non_numeric_parts) if non_numeric_parts else full_id

            o1 = extract_object_type(o1_full)
            o2 = extract_object_type(o2_full)

            object_1.append(o1)
            object_1 = np.unique(object_1).tolist()
            object_2.append(o2)
            object_2 = np.unique(object_2).tolist()
            rel.append(p['object_1_relation_to_2'])
            # print(f'{o1}_{rel}_{o2}')

        a,b = np.unique(object_1,return_counts=True)
        p = sorted([(x,y) for x,y in zip(b,a)])
        target+=a.tolist()

    unique_obj = np.unique(target)
    print([i.item() for i in unique_obj])
    print('================================')
    # print([correct_description(i) for i in unique_obj])

    key1 = set([correct_description_v2(i) for i in unique_obj])
    breakpoint()
    # print(f'Unique objects found: {key1}')
    key2 = set(target_to_actions_final.keys())
    print('Only in unique objects, should be empty:',key1-key2)
    print('Only in target_to_actions_final:',key2-key1)
