#!/usr/bin/env python3
"""
=============================================================================
Team-Draft Interleaving Engine (Netflix / Pinterest inspired)
Blends two separate model ranking lists into a single unbiased presentation 
list, tracing attribution to see which model users prefer in real-time.
=============================================================================
"""

import random
from typing import List, Dict, Tuple

class TeamDraftInterleaver:
    """
    Implements Team-Draft Interleaving for high-velocity online model evaluation.
    Randomly assigns selection priority at each step to prevent positional bias.
    """
    
    @staticmethod
    def interleave(list_a: List[str], list_b: List[str], max_size: int = 10) -> Tuple[List[str], Dict[str, str]]:
        """
        Blends two recommendation lists using a randomized coin-flip team draft.
        
        Args:
            list_a: Recommendations from Model A (Production)
            list_b: Recommendations from Model B (Candidate)
            max_size: Maximum length of the final blended list
            
        Returns:
            interleaved_list: The final list of items to show the user
            assignment_map: Dictionary mapping each item to the model that provided it
        """
        interleaved_list = []
        assignment_map = {}
        
        # Create local queue copies to pop from
        queue_a = [item for item in list_a if item]
        queue_b = [item for item in list_b if item]
        selected_set = set()
        
        while len(interleaved_list) < max_size and (queue_a or queue_b):
            # If Queue A is empty, exhaust Queue B
            if not queue_a:
                item = queue_b.pop(0)
                if item not in selected_set:
                    interleaved_list.append(item)
                    assignment_map[item] = "model_B"
                    selected_set.add(item)
                continue
                
            # If Queue B is empty, exhaust Queue A
            if not queue_b:
                item = queue_a.pop(0)
                if item not in selected_set:
                    interleaved_list.append(item)
                    assignment_map[item] = "model_A"
                    selected_set.add(item)
                continue
                
            # Coin flip to determine priority for the current slot to avoid bias
            if random.choice(["A", "B"]) == "A":
                item = queue_a.pop(0)
                if item not in selected_set:
                    interleaved_list.append(item)
                    assignment_map[item] = "model_A"
                    selected_set.add(item)
            else:
                item = queue_b.pop(0)
                if item not in selected_set:
                    interleaved_list.append(item)
                    assignment_map[item] = "model_B"
                    selected_set.add(item)
                    
        return interleaved_list[:max_size], assignment_map


if __name__ == "__main__":
    print("[*] Running Team-Draft Interleaving Simulation...")
    
    # Mock recommendation streams
    production_list = ["item_1", "item_2", "item_3", "item_4", "item_8"]
    candidate_list = ["item_5", "item_2", "item_6", "item_7", "item_9"] # Note: item_2 exists in both
    
    print(f"List A (Production): {production_list}")
    print(f"List B (Candidate):  {candidate_list}")
    
    blended_feed, attribution_map = TeamDraftInterleaver.interleave(
        production_list, candidate_list, max_size=6
    )
    
    print(f"\n[+] Blended Presentation List: {blended_feed}")
    print("[+] Attribution Map (For downstream click tracking):")
    for item in blended_feed:
        print(f"    - {item}  ->  Credit to: {attribution_map[item]}")