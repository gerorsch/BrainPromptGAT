"""
Generate ROI-specific descriptions using ChatGPT API.

This script generates detailed descriptions for each ROI in the AAL116 atlas
using ChatGPT, following the approach described in the BrainPrompt paper.

The query sent to ChatGPT is:
"Given the ROI labels for AAL116 atlas, generate a sentence to describe 
each of them by the given order: Precentral_L, Precentral_R, Frontal_Sup_L ..."

Usage:
    python generate_roi_descriptions.py
    
    Or with custom API key:
    OPENAI_API_KEY=your_key python generate_roi_descriptions.py
"""

import os
import json
import sys
from generate_prompts import _aal_labels_offline

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. Install with: pip install openai")


def get_roi_labels():
    """Get list of ROI labels from AAL116 atlas."""
    try:
        from nilearn import datasets
        dataset = datasets.fetch_atlas_aal(version="SPM12")
        labels = list(dataset.labels)
        if len(labels) > 116:
            labels = labels[:116]
        return labels
    except Exception:
        return _aal_labels_offline()


def generate_descriptions_via_chatgpt(labels, api_key=None):
    """
    Generate ROI descriptions using ChatGPT API.
    
    Args:
        labels: List of ROI labels (116 labels)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
    
    Returns:
        dict: Mapping from ROI label to description string
    """
    if not HAS_OPENAI:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    client = OpenAI(api_key=api_key)
    
    # Build query as specified in the paper
    labels_str = ", ".join(labels)
    query = (
        f"Given the ROI labels for AAL116 atlas, generate a sentence to describe "
        f"each of them by the given order: {labels_str}. "
        f"Each description should describe the general structural and functional "
        f"features of the brain region. Format your response as a JSON object where "
        f"keys are the ROI labels and values are the descriptions."
    )
    
    print(f"Querying ChatGPT for {len(labels)} ROI descriptions...")
    print("This may take a few minutes...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency, can use gpt-4 for better quality
            messages=[
                {
                    "role": "system",
                    "content": "You are a neuroscience expert. Generate concise, accurate descriptions of brain regions based on their anatomical names."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        # Parse JSON response
        content = response.choices[0].message.content
        descriptions = json.loads(content)
        
        # Verify all labels are present
        missing = [label for label in labels if label not in descriptions]
        if missing:
            print(f"Warning: {len(missing)} labels missing from response. Regenerating...")
            # Try to get missing descriptions individually
            for label in missing:
                individual_query = (
                    f"Generate a sentence describing the brain region '{label}' "
                    f"from the AAL116 atlas. Describe its general structural and functional features."
                )
                individual_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a neuroscience expert."},
                        {"role": "user", "content": individual_query}
                    ],
                    temperature=0.7
                )
                descriptions[label] = individual_response.choices[0].message.content.strip()
        
        return descriptions
        
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        raise


def save_descriptions(descriptions, save_path):
    """Save descriptions to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    print(f"Saved descriptions to: {save_path}")


def main():
    """Main function to generate and save ROI descriptions."""
    # Get ROI labels
    print("1) Loading AAL116 atlas labels...")
    labels = get_roi_labels()
    print(f"   Found {len(labels)} ROI labels")
    
    # Determine save path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "data", "roi_descriptions.json")
    
    # Check if descriptions already exist
    if os.path.exists(save_path):
        response = input(f"Descriptions file already exists at {save_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Generate descriptions
    print("\n2) Generating descriptions via ChatGPT...")
    try:
        descriptions = generate_descriptions_via_chatgpt(labels)
        print(f"   Generated {len(descriptions)} descriptions")
        
        # Show example
        if labels:
            example_label = labels[0]
            if example_label in descriptions:
                print(f"\n   Example ({example_label}):")
                print(f"   {descriptions[example_label]}")
        
        # Save descriptions
        print("\n3) Saving descriptions...")
        save_descriptions(descriptions, save_path)
        
        print("\n✓ Successfully generated and saved ROI descriptions!")
        print(f"  File: {save_path}")
        print(f"  Next step: Run generate_prompts.py to create embeddings")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nAlternative: You can manually create roi_descriptions.json with the format:")
        print('  {"Precentral_L": "The left precentral gyrus, associated with motor control and planning.", ...}')
        sys.exit(1)


if __name__ == "__main__":
    main()
